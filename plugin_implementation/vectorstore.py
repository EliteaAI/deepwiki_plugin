"""
FAISS-based vector store manager with single HuggingFace embedding model and code_graph retriever
"""

import copy
import hashlib
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Global cache for HuggingFace embeddings instances
_EMBEDDINGS_CACHE = {}

def _get_embeddings_cache_key(model_name: str, cache_folder: str) -> str:
    """Generate a cache key for embeddings based on model name and cache folder."""
    return f"{model_name}:{cache_folder}"

def get_cached_embeddings(model_name: str, cache_folder: str):
    """Get or create a cached HuggingFaceEmbeddings instance."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    cache_key = _get_embeddings_cache_key(model_name, cache_folder)
    
    if cache_key not in _EMBEDDINGS_CACHE:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=cache_folder
        )
        _EMBEDDINGS_CACHE[cache_key] = embeddings
    
    return _EMBEDDINGS_CACHE[cache_key]

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logging.warning("HuggingFace embeddings not available")

logger = logging.getLogger(__name__)


class DummyEmbeddings(Embeddings):
    """Dummy embeddings class for loading cached FAISS indexes"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return dummy embeddings for documents"""
        return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Return dummy embedding for query"""
        return [0.0] * 384


class VectorStoreManager:
    """Manager for FAISS vector stores with single HuggingFace embedding model"""
    
    def __init__(self, cache_dir: Optional[str] = None,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model_cache_dir: Optional[str] = None):
        """
        Initialize VectorStoreManager with single embedding model
        
        Args:
            cache_dir: Directory for caching vector stores
            model_name: HuggingFace model name for embeddings
            model_cache_dir: Directory for caching HuggingFace models
        """
        
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.alita/wiki_indexes")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up model cache directory
        if model_cache_dir is None:
            model_cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize single embedding model using cache
        if HUGGINGFACE_AVAILABLE:
            self.embeddings = get_cached_embeddings(model_name, str(self.model_cache_dir))
        else:
            raise ImportError("HuggingFace embeddings not available")
        
        # Vector store and documents
        self.vectorstore: Optional[FAISS] = None
        self.documents: List[Document] = []
        self.document_ids: Dict[str, int] = {}  # UUID to index mapping
        
    def load_or_build(self, documents: List[Document], repo_path: str, 
                     force_rebuild: bool = False) -> Tuple[FAISS, List[Document]]:
        """
        Load existing vector store or build new one
        
        Args:
            documents: List of documents to index
            repo_path: Path to repository for cache key generation
            force_rebuild: Force rebuilding even if cache exists
            
        Returns:
            - FAISS vector store
            - List of indexed documents
        """
        # Don't mutate input list
        docs_copy = copy.deepcopy(documents)
        
        # Generate cache key based on repository and documents
        repo_hash = self._generate_repo_hash(repo_path, docs_copy)
        cache_file = self.cache_dir / f"{repo_hash}.faiss"
        docs_file = self.cache_dir / f"{repo_hash}.docs.pkl"
        
        # Try to load from cache
        if not force_rebuild and cache_file.exists() and docs_file.exists():
            try:
                logger.info(f"Loading vector store from cache: {cache_file}")
                return self._load_from_cache(cache_file, docs_file)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}, rebuilding...")
        
        # Build new vector store
        logger.info("Building new vector store...")
        return self._build_and_save(docs_copy, cache_file, docs_file)
    
    def _generate_repo_hash(self, repo_path: str, documents: List[Document]) -> str:
        """Generate hash for repository content"""
        hasher = hashlib.md5()
        
        # Include repo path
        hasher.update(repo_path.encode())
        
        # Include document count and some content samples
        hasher.update(f"docs:{len(documents)}".encode())
        
        # Sample some content for hash
        for i, doc in enumerate(documents[:10]):
            hasher.update(f"doc_{i}:{doc.page_content[:100]}".encode())
        
        return hasher.hexdigest()
    
    def _load_from_cache(self, cache_file: Path, docs_file: Path) -> Tuple[FAISS, List[Document]]:
        """Load vector store and documents from cache"""
        # Load documents
        with open(docs_file, 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load vector store
        self.vectorstore = FAISS.load_local(
            str(cache_file.parent),
            embeddings=self.embeddings,
            index_name=cache_file.stem,
            allow_dangerous_deserialization=True
        )
        
        # Rebuild document ID mapping
        self._rebuild_document_mapping()
        
        logger.info(f"Loaded {len(self.documents)} documents from cache")
        return self.vectorstore, self.documents
    
    def _build_and_save(self, documents: List[Document], cache_file: Path, 
                       docs_file: Path) -> Tuple[FAISS, List[Document]]:
        """Build vector store and save to cache"""
        if not documents:
            raise ValueError("No documents to index")
        
        # Assign UUIDs to documents if not present
        for doc in documents:
            if 'uuid' not in doc.metadata:
                doc.metadata['uuid'] = str(uuid.uuid4())
        
        self.documents = documents
        
        # Extract text content for embedding
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        logger.info(f"Embedding {len(texts)} documents...")
        
        # Build FAISS vector store
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Build document ID mapping
        self._rebuild_document_mapping()
        
        # Save to cache
        try:
            logger.info(f"Saving vector store to cache: {cache_file}")
            self.vectorstore.save_local(
                str(cache_file.parent),
                index_name=cache_file.stem
            )
            
            # Save documents
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
                
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
        
        logger.info(f"Built vector store with {len(self.documents)} documents")
        return self.vectorstore, self.documents
    
    def _rebuild_document_mapping(self):
        """Rebuild the UUID to index mapping"""
        self.document_ids = {}
        for i, doc in enumerate(self.documents):
            if 'uuid' in doc.metadata:
                self.document_ids[doc.metadata['uuid']] = i
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add new documents to the vector store"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        # Don't mutate input list
        docs_copy = copy.deepcopy(documents)
        
        # Assign UUIDs if not present
        doc_ids = []
        for doc in docs_copy:
            if 'uuid' not in doc.metadata:
                doc_id = str(uuid.uuid4())
                doc.metadata['uuid'] = doc_id
            else:
                doc_id = doc.metadata['uuid']
            doc_ids.append(doc_id)
        
        # Extract texts and metadatas
        texts = [doc.page_content for doc in docs_copy]
        metadatas = [doc.metadata for doc in docs_copy]
        
        # Add to FAISS
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        
        # Update document lists and mapping
        start_idx = len(self.documents)
        self.documents.extend(docs_copy)
        
        for i, doc_id in enumerate(doc_ids):
            self.document_ids[doc_id] = start_idx + i
        
        return doc_ids
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents by UUID using efficient FAISS deletion"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        # Find docstore IDs to delete
        docstore_ids_to_delete = []
        indices_to_remove = []
        
        for i, doc in enumerate(self.documents):
            doc_uuid = doc.metadata.get('uuid')
            if doc_uuid in document_ids:
                # Find the docstore ID for this document
                # Index in self.documents corresponds to index in vectorstore
                if i in self.vectorstore.index_to_docstore_id:
                    docstore_id = self.vectorstore.index_to_docstore_id[i]
                    docstore_ids_to_delete.append(docstore_id)
                    indices_to_remove.append(i)
        
        if not docstore_ids_to_delete:
            # No documents to delete
            return
        
        # Delete from FAISS vectorstore (efficient, no rebuilding)
        self.vectorstore.delete(docstore_ids_to_delete)
        
        # Remove documents from our list (in reverse order to maintain indices)
        for idx in sorted(indices_to_remove, reverse=True):
            del self.documents[idx]
        
        # Rebuild document mapping to match new vectorstore state
        self._rebuild_document_mapping()
    
    def search(self, query: str, k: int = 10, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Search documents with optional filtering"""
        if not self.vectorstore:
            return []
        
        if filter_dict:
            return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            return self.vectorstore.similarity_search(query, k=k)
    
    def search_with_score(self, query: str, k: int = 10, 
                         filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Search with similarity scores"""
        if not self.vectorstore:
            return []
        
        if filter_dict:
            results = self.vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Convert numpy.float32 to float for JSON serialization
        return [(doc, float(score)) for doc, score in results]
    
    def search_by_type(self, query: str, chunk_type: str, k: int = 10) -> List[Document]:
        """Search documents by chunk type (code, text, etc.)"""
        filter_dict = {"chunk_type": {"$eq": chunk_type}}
        return self.search(query, k=k, filter_dict=filter_dict)
    
    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None):
        """Convert to LangChain retriever"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        if search_kwargs is None:
            search_kwargs = {"k": 10}
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def get_vectorstore(self) -> Optional[FAISS]:
        """Get the FAISS vector store"""
        return self.vectorstore
    
    def get_all_documents(self) -> List[Document]:
        """Get all indexed documents"""
        return self.documents.copy()  # Return copy to prevent mutation
    
    def clear_cache(self, repo_path: Optional[str] = None):
        """Clear cached vector stores"""
        if repo_path:
            # Clear specific repository cache
            docs = []  # Empty for hash calculation
            repo_hash = self._generate_repo_hash(repo_path, docs)
            cache_file = self.cache_dir / f"{repo_hash}.faiss"
            docs_file = self.cache_dir / f"{repo_hash}.docs.pkl"
            
            for file_path in [cache_file, docs_file]:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed cache file: {file_path}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.faiss"):
                cache_file.unlink()
            for docs_file in self.cache_dir.glob("*.docs.pkl"):
                docs_file.unlink()
            logger.info("Cleared all vector store cache")
