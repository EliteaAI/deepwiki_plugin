"""
Retriever stack with ensemble retrievers and cross-encoder reranking
"""

import logging
import traceback
from typing import List, Optional, Dict, Any, Union
import asyncio

from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableConfig

# Import content expander for post-retrieval enhancement
from .content_expander import ContentExpander

try:
    from langchain_community.retrievers import BM25Retriever
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("BM25Retriever not available")

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logging.warning("TavilySearchResults not available")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("CrossEncoder not available")

import networkx as nx
from .vectorstore import VectorStoreManager
from .content_expander import ContentExpander

logger = logging.getLogger(__name__)


class WebRetriever:
    """Web search retriever using Tavily"""
    
    def __init__(self, k: int = 10, api_key: Optional[str] = None):
        self.k = k
        self.api_key = api_key
        self.search_tool = None
        
        if TAVILY_AVAILABLE and api_key:
            try:
                self.search_tool = TavilySearchResults(
                    max_results=k,
                    search_depth="advanced",
                    api_key=api_key
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily search: {e}")
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get web search results as documents"""
        if not self.search_tool:
            logger.debug("Web search tool not available")
            return []
        
        try:
            results = self.search_tool.run(query)
            documents = []
            
            if isinstance(results, list):
                for i, result in enumerate(results):
                    if isinstance(result, dict):
                        content = result.get('content', '')
                        url = result.get('url', f'web_result_{i}')
                        title = result.get('title', 'Web Result')
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': url,
                                'title': title,
                                'chunk_type': 'web',
                                'search_query': query
                            }
                        )
                        documents.append(doc)
            
            logger.debug(f"Retrieved {len(documents)} web search results")
            return documents
            
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []


class WikiRetrieverStack:
    """Complete retriever stack with ensemble and reranking"""
    
    def __init__(self, 
                 vectorstore_manager: VectorStoreManager,
                 relationship_graph: Optional[nx.DiGraph] = None,
                 tavily_api_key: Optional[str] = None,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 llm_client=None,
                 use_enhanced_graph: bool = True):
        
        self.vectorstore_manager = vectorstore_manager
        self.relationship_graph = relationship_graph
        self.tavily_api_key = tavily_api_key
        self.cross_encoder_model = cross_encoder_model
        self.llm_client = llm_client
        self.use_enhanced_graph = use_enhanced_graph
        
        # Initialize content expander for intelligent post-processing
        self.content_expander = ContentExpander(relationship_graph)
        
        # Core retrievers
        self.dense_retriever = None
        self.bm25_retriever = None
        self.graph_retriever = None
        self.web_retriever = None
        
        # Ensemble retrievers
        self.repo_retriever = None
        self.research_retriever = None
        
        # Reranker
        self.reranker = None
        
        # Initialize components
        self._initialize_retrievers()
    
    def _initialize_retrievers(self):
        """Initialize all retriever components"""
        try:
            # Get vector store and documents
            vectorstore = self.vectorstore_manager.get_vectorstore()
            all_documents = self.vectorstore_manager.get_all_documents()
            
            if not vectorstore or not all_documents:
                logger.warning("No vector store or documents available")
                return
            
            # 1. Dense retriever (FAISS)
            self.dense_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 40}
            )
            logger.info("Initialized dense retriever")
            
            # 2. BM25 retriever
            if BM25_AVAILABLE and all_documents:
                try:
                    self.bm25_retriever = BM25Retriever.from_documents(
                        all_documents,
                        k=40
                    )
                    logger.info("Initialized BM25 retriever")
                except Exception as e:
                    logger.warning(f"Failed to initialize BM25 retriever: {e}")
            
            # 5. Cross-encoder reranker
            if CROSS_ENCODER_AVAILABLE:
                try:
                    model_kwargs = {'device': 'cpu'}
                    ce_model = HuggingFaceCrossEncoder(
                        model_name=self.cross_encoder_model,
                        model_kwargs=model_kwargs
                    )
                    self.reranker = CrossEncoderReranker(
                        model=ce_model,
                        top_n=20
                    )
                    logger.info(f"Initialized cross-encoder reranker: {self.cross_encoder_model}")
                except Exception as e:
                    logger.warning(f"Failed to initialize cross-encoder reranker: {e}")
            
            # Build ensemble retrievers
            self._build_ensemble_retrievers()
            
        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")
    
    def _build_ensemble_retrievers(self):
        """Build ensemble retrievers with simplified architecture"""
        try:
            # Repository retriever - flatten all retrievers into one ensemble
            all_retrievers = []
            all_weights = []
            
            if self.dense_retriever:
                all_retrievers.append(self.dense_retriever)
                all_weights.append(0.6)
            
            if self.bm25_retriever:
                all_retrievers.append(self.bm25_retriever)
                all_weights.append(0.4)

            # Create single flat ensemble instead of nested ones
            if len(all_retrievers) >= 2:
                self.repo_retriever = EnsembleRetriever(
                    retrievers=all_retrievers,
                    weights=all_weights
                )
                logger.info(f"Built flat ensemble with {len(all_retrievers)} retrievers")

                if self.reranker:
                    self.repo_retriever = ContextualCompressionRetriever(
                        base_retriever=self.repo_retriever,
                        base_compressor=self.reranker
                    )
                
            elif len(all_retrievers) == 1:
                # If only one retriever, use it directly
                self.repo_retriever = all_retrievers[0]
                logger.info("Using single retriever (no ensemble needed)")
            else:
                logger.warning("No retrievers available for ensemble")
                return
                
        except Exception as e:
            logger.error(f"Failed to build ensemble retrievers: {e}")
            # Fallback to dense retriever only
            if self.dense_retriever:
                self.repo_retriever = self.dense_retriever
                self.research_retriever = self.dense_retriever
                logger.info("Fallback: Using dense retriever only")
    
    def build_repo_retriever(self, k: int = 20) -> Optional[Any]:
        """
        Build repository-focused retriever
        
        Returns retriever configured for repository-only search
        """
        if not self.repo_retriever:
            logger.warning("Repository retriever not available")
            return None
        
        # Configure k parameter
        if hasattr(self.repo_retriever, 'search_kwargs'):
            self.repo_retriever.search_kwargs = {"k": k}
        
        return self.repo_retriever
    
    def search_repository(self, query: str, k: int = 20) -> List[Document]:
        """Retriever with code_graph expansion"""
        raw_result = self.repo_retriever.invoke(query)
        # Apply content expansion even in streamlit safe mode
        expanded_result = self.content_expander.expand_retrieved_documents(
            raw_result or []
        )
        return expanded_result[:k] if expanded_result else []

