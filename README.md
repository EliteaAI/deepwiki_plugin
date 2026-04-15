# DeepWiki Plugin

Advanced repository documentation generator with AI-powered analysis and multiple indexing strategies.

## 🚀 Quick Start

### Filesystem-based Indexing (Recommended)

```python
from plugin_implementation.hybrid_wiki_toolkit_wrapper import HybridWikiToolkitWrapper

# Fast, local repository analysis with authentication
wrapper = HybridWikiToolkitWrapper(
    github_repository="microsoft/vscode",
    github_access_token="ghp_your_token",  # Required for private repos, optional for public
    indexing_method="filesystem"  # Default
)

result = wrapper.generate_wiki(
    query="Explain the VS Code extension system",
    include_diagrams=True
)

print(result['content'])
```

### Authentication Support

The filesystem indexer supports both public and private repositories:

```python
# Public repository (no token needed)
wrapper = HybridWikiToolkitWrapper(
    github_repository="microsoft/vscode",
    indexing_method="filesystem"
)

# Private repository (token required)
wrapper = HybridWikiToolkitWrapper(
    github_repository="your-org/private-repo",
    github_access_token="ghp_your_personal_access_token",
    indexing_method="filesystem"
)

# Using environment variable (recommended)
import os
wrapper = HybridWikiToolkitWrapper(
    github_repository="your-org/private-repo", 
    github_access_token=os.getenv("GITHUB_ACCESS_TOKEN"),
    indexing_method="filesystem"
)
```

### GitHub API-based Indexing (Legacy)

```python
# Traditional API approach (slower due to rate limits)
wrapper = HybridWikiToolkitWrapper(
    github_repository="owner/repo",
    indexing_method="github_api",
    rate_limit_delay=0.1
)
```

## 🏗️ Architecture

### Two Indexing Methods

1. **Filesystem-based** (⭐ Recommended)
   - Clones repositories locally for direct filesystem access
   - Eliminates GitHub API rate limiting
   - Faster processing with Enhanced Unified Graph Builder
   - Supports offline operation after initial clone

2. **GitHub API-based** (Legacy)
   - Uses GitHub API with parallel processing workarounds
   - Subject to API rate limits and network latency
   - Complex error handling for API failures

### Performance Comparison

| Feature | Filesystem | GitHub API |
|---------|------------|------------|
| Speed | 🚀 Fast | 🐌 Slow (rate limited) |
| Reliability | ✅ High | ⚠️ Network dependent |
| Complexity | ✅ Simple | ❌ Complex (parallelization) |
| Offline Support | ✅ Yes | ❌ No |

## 📦 Components

### Core Components
- **HybridWikiToolkitWrapper** - Main interface supporting both indexing methods
- **FilesystemRepositoryIndexer** - Local repository analysis with EUGB
- **GitHubIndexer** - Legacy GitHub API-based indexer
- **LocalRepositoryManager** - Git clone and repository management
- **OptimizedWikiGenerationAgent** - LangGraph-based wiki generation

### Enhanced Features
- **Enhanced Unified Graph Builder (EUGB)** - Advanced code analysis
- **Creative Freedom Prompts** - AI-optimized wiki generation
- **Multi-source Context** - Code + documentation analysis
- **Mermaid Diagram Generation** - Technical visualization
- **Caching & Performance** - Optimized for repeated usage

## 🔧 Installation

```bash
git clone <repository>
cd deepwiki_plugin
pip install -r requirements.txt
```

## 📚 Usage Examples

### Simple Usage

```python
from plugin_implementation.hybrid_wiki_toolkit_wrapper import create_filesystem_wrapper

# Quick filesystem-based wrapper
wrapper = create_filesystem_wrapper(
    github_repository="owner/repo",
    github_access_token="ghp_token"
)

result = wrapper.generate_wiki("Document the architecture")
```

### Advanced Configuration

```python
with HybridWikiToolkitWrapper(
    github_repository="large/repo",
    indexing_method="filesystem",
    
    # Performance tuning
    max_workers=16,
    max_files=5000,
    
    # Repository management
    cleanup_repos_on_exit=True,
    force_reclone=False,
    
    # Caching
    cache_dir="/custom/cache",
    force_rebuild_index=False
    
) as wrapper:
    
    # Multiple wikis using same index
    overview = wrapper.generate_wiki("Architectural overview")
    apis = wrapper.generate_wiki("Document the APIs")
    setup = wrapper.generate_wiki("Installation guide")

# Auto-cleanup on exit
```

### GitHub Token Setup

For private repositories, you need a GitHub Personal Access Token:

```bash
# 1. Generate token at: https://github.com/settings/tokens
# 2. Select 'repo' scope for full repository access
# 3. Set environment variable
export GITHUB_ACCESS_TOKEN="ghp_your_personal_access_token"

# 4. Verify token works
python examples/authentication_examples.py
```

**Token Requirements:**
- **Public repos**: No token needed
- **Private repos**: Token with `repo` scope required
- **Organization repos**: Token with appropriate organization access

### Performance Benchmark

```python
# Compare both methods
from examples.filesystem_vs_api_demo import benchmark_indexing_methods

results = benchmark_indexing_methods(
    repository="microsoft/vscode",
    query="Explain the extension system",
    github_token="ghp_token"
)

print(f"Speedup: {results['speedup']:.1f}x faster")
```

## 🔄 Migration Guide

### From Legacy Wrapper

```python
# DEPRECATED (legacy API-only)
from plugin_implementation.wiki_toolkit_wrapper import OptimizedWikiToolkitWrapper  # legacy
legacy_wrapper = OptimizedWikiToolkitWrapper(...)

# CURRENT (recommended, supports filesystem + API fallback)
from plugin_implementation.hybrid_wiki_toolkit_wrapper import HybridWikiToolkitWrapper
wrapper = HybridWikiToolkitWrapper(..., indexing_method="filesystem")  # default & fastest
```

### Gradual Migration

1. **Phase 1**: Use `HybridWikiToolkitWrapper` with `indexing_method="github_api"`
2. **Phase 2**: Switch to `indexing_method="filesystem"` 
3. **Phase 3**: Remove GitHub API fallbacks

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration steps.

## 🎯 Features

### Wiki Generation
- 📄 **Comprehensive Documentation** - Architecture, APIs, guides
- 🎨 **Mermaid Diagrams** - Visual system representations  
- 🔍 **Deep Code Analysis** - Symbol relationships and dependencies
- 🎭 **Creative Freedom** - AI-optimized content generation vs rigid templates

### Repository Analysis
- 🕸️ **Relationship Graphs** - Code dependency mapping
- 📊 **Multi-language Support** - Python, JavaScript, TypeScript, Java, etc.
- 🔄 **Enhanced Unified Graph Builder** - Advanced AST analysis
- 📈 **Performance Metrics** - Indexing and generation statistics

### Performance & Reliability
- ⚡ **Parallel Processing** - Multi-threaded analysis
- 🗄️ **Intelligent Caching** - Vector stores and relationship graphs
- 🧹 **Resource Management** - Auto-cleanup of temporary repositories
- 📱 **Context Management** - Proper resource handling

## 🛠️ Configuration

### Environment Variables

```bash
export GITHUB_ACCESS_TOKEN="ghp_your_github_token"  # For private repos
export DEEPWIKI_CACHE_DIR="/custom/cache/path"      # Optional
export DEEPWIKI_MAX_WORKERS="8"                     # Optional
```

### Repository Filters

Create `repo.json` for custom filtering:

```json
{
  "file_filters": {
    "excluded_dirs": [".git", "node_modules", "__pycache__"],
    "excluded_files": ["*.pyc", "*.log", ".DS_Store"],
    "included_extensions": [".py", ".js", ".ts", ".md"]
  }
}
```

## 🚦 Examples

Run the included examples:

```bash
# Filesystem vs API performance comparison
python examples/filesystem_vs_api_demo.py

# Authentication examples (public/private repos)
python examples/authentication_examples.py

# Set environment variable for private repos
export GITHUB_ACCESS_TOKEN="ghp_your_token"
python examples/filesystem_vs_api_demo.py
python examples/authentication_examples.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test both indexing methods
4. Submit a pull request

## 📄 License

See LICENSE file for details.

## 🔗 Related

- [Enhanced Unified Graph Builder](plugin_implementation/code_graph/) - Advanced code analysis
- [Wiki Generation Agents](plugin_implementation/agents/) - LangGraph-based generation
- [Content Expansion System](plugin_implementation/content_expander.py) - Post-retrieval enhancement
