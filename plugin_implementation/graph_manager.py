"""
Graph manager for saving and loading NetworkX graphs
"""

import os
import gzip
import pickle
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

import networkx as nx

logger = logging.getLogger(__name__)


class GraphManager:
    """Manager for persisting and loading NetworkX graphs"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.alita/wiki_indexes")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_graph(self, graph: nx.DiGraph, repo_path: str, graph_type: str = "combined") -> str:
        """
        Save code_graph to compressed pickle file
        
        Args:
            graph: NetworkX code_graph to save
            repo_path: Repository path for generating cache key
            graph_type: Type of code_graph (combined, import, call)
            
        Returns:
            Cache file path
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(repo_path, graph_type)
            cache_file = self.cache_dir / f"{cache_key}.code_graph.gz"
            
            # Save code_graph with compression
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved {graph_type} code_graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            logger.error(f"Failed to save code_graph: {e}")
            raise
    
    def load_graph(self, repo_path: str, graph_type: str = "combined") -> Optional[nx.DiGraph]:
        """
        Load code_graph from cache
        
        Args:
            repo_path: Repository path for generating cache key
            graph_type: Type of code_graph to load
            
        Returns:
            Loaded NetworkX code_graph or None if not found
        """
        try:
            cache_key = self._generate_cache_key(repo_path, graph_type)
            cache_file = self.cache_dir / f"{cache_key}.code_graph.gz"
            
            if not cache_file.exists():
                logger.debug(f"Graph cache file not found: {cache_file}")
                return None
            
            # Load code_graph
            with gzip.open(cache_file, 'rb') as f:
                graph = pickle.load(f)
            
            logger.info(f"Loaded {graph_type} code_graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges from {cache_file}")
            return graph
            
        except Exception as e:
            logger.warning(f"Failed to load code_graph from cache: {e}")
            return None
    
    def _generate_cache_key(self, repo_path: str, graph_type: str) -> str:
        """Generate cache key for code_graph"""
        # Use repo path and modification time if available
        hasher = hashlib.md5()
        hasher.update(repo_path.encode())
        hasher.update(graph_type.encode())
        
        # Include repo modification time if path exists
        if os.path.exists(repo_path):
            try:
                mtime = os.path.getmtime(repo_path)
                hasher.update(str(mtime).encode())
            except OSError:
                pass
        
        return hasher.hexdigest()
    
    def graph_exists(self, repo_path: str, graph_type: str = "combined") -> bool:
        """Check if code_graph cache exists"""
        cache_key = self._generate_cache_key(repo_path, graph_type)
        cache_file = self.cache_dir / f"{cache_key}.code_graph.gz"
        return cache_file.exists()
    
    def clear_cache(self, repo_path: str = None):
        """Clear code_graph cache"""
        if repo_path:
            # Clear specific repository graphs
            for graph_type in ["combined", "import", "call"]:
                cache_key = self._generate_cache_key(repo_path, graph_type)
                cache_file = self.cache_dir / f"{cache_key}.code_graph.gz"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Removed code_graph cache: {cache_file}")
        else:
            # Clear all code_graph cache
            for cache_file in self.cache_dir.glob("*.code_graph.gz"):
                cache_file.unlink()
            logger.info("Cleared all code_graph cache")
    
    def export_graph_data(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Export code_graph to serializable format"""
        return {
            'nodes': [
                {
                    'id': node_id,
                    **data
                }
                for node_id, data in graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    **data
                }
                for source, target, data in graph.edges(data=True)
            ],
            'graph_info': {
                'number_of_nodes': graph.number_of_nodes(),
                'number_of_edges': graph.number_of_edges(),
                'is_directed': graph.is_directed(),
                'is_multigraph': graph.is_multigraph()
            }
        }
    
    def import_graph_data(self, graph_data: Dict[str, Any]) -> nx.DiGraph:
        """Import code_graph from serializable format"""
        graph = nx.DiGraph()
        
        # Add nodes
        for node_data in graph_data.get('nodes', []):
            node_id = node_data.pop('id')
            graph.add_node(node_id, **node_data)
        
        # Add edges
        for edge_data in graph_data.get('edges', []):
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            graph.add_edge(source, target, **edge_data)
        
        return graph
    
    def analyze_graph(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze code_graph properties"""
        analysis = {
            'basic_stats': {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'is_connected': nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph)
            }
        }
        
        if graph.number_of_nodes() > 0:
            # Degree statistics
            degrees = dict(graph.degree())
            if degrees:
                analysis['degree_stats'] = {
                    'average_degree': sum(degrees.values()) / len(degrees),
                    'max_degree': max(degrees.values()),
                    'min_degree': min(degrees.values())
                }
            
            # Try to compute centrality measures for small graphs
            if graph.number_of_nodes() < 1000:
                try:
                    centrality = nx.degree_centrality(graph)
                    analysis['top_central_nodes'] = sorted(
                        centrality.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]
                except Exception as e:
                    logger.debug(f"Could not compute centrality: {e}")
            
            # Component analysis
            if graph.is_directed():
                components = list(nx.weakly_connected_components(graph))
            else:
                components = list(nx.connected_components(graph))
            
            analysis['components'] = {
                'count': len(components),
                'largest_size': max(len(comp) for comp in components) if components else 0,
                'sizes': [len(comp) for comp in components]
            }
        
        return analysis
    
    def find_shortest_path(self, graph: nx.DiGraph, source: str, target: str) -> Optional[list]:
        """Find shortest path between two nodes"""
        try:
            if source in graph and target in graph:
                return nx.shortest_path(graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return None
    
    def get_node_neighbors(self, graph: nx.DiGraph, node: str, hops: int = 1) -> set:
        """Get neighbors of a node within specified hops"""
        if node not in graph:
            return set()
        
        neighbors = set()
        current_nodes = {node}
        
        for _ in range(hops):
            next_nodes = set()
            for n in current_nodes:
                # Get both predecessors and successors
                next_nodes.update(graph.predecessors(n))
                next_nodes.update(graph.successors(n))
            
            neighbors.update(next_nodes)
            current_nodes = next_nodes - neighbors
        
        return neighbors - {node}  # Exclude the original node
    
    def filter_graph_by_file(self, graph: nx.DiGraph, file_path: str) -> nx.DiGraph:
        """Filter code_graph to nodes from specific file"""
        filtered_nodes = [
            node for node, data in graph.nodes(data=True)
            if data.get('file_path') == file_path
        ]
        return graph.subgraph(filtered_nodes).copy()
    
    def filter_graph_by_language(self, graph: nx.DiGraph, language: str) -> nx.DiGraph:
        """Filter code_graph to nodes of specific language"""
        filtered_nodes = [
            node for node, data in graph.nodes(data=True)
            if data.get('language') == language
        ]
        return graph.subgraph(filtered_nodes).copy()
    
    def get_files_in_graph(self, graph: nx.DiGraph) -> set:
        """Get unique file paths in code_graph"""
        files = set()
        for node, data in graph.nodes(data=True):
            file_path = data.get('file_path')
            if file_path:
                files.add(file_path)
        return files
    
    def get_languages_in_graph(self, graph: nx.DiGraph) -> set:
        """Get unique languages in code_graph"""
        languages = set()
        for node, data in graph.nodes(data=True):
            language = data.get('language')
            if language:
                languages.add(language)
        return languages
