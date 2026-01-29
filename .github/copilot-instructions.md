# DeepWiki Plugin - AI Assistant Instructions

## Directory Structure Overview

This is a **multi-layered workspace** with the following structure:

```
deepwiki_plugin/                    # Infrastructure/runner repository
├── .github/                        # THIS FILE - workspace-wide AI instructions
├── docker-compose.yml              # Local development infrastructure
├── .venv                           # Python virtual environment (gitignored). Use it for dev and running tests and scripts.  
├── envs/                           # Environment configurations
│   ├── default.env
│   └── override.env
├── pylon_deepwiki/                 # Pylon platform integration
│   ├── pylon.yml                   # Pylon configuration
│   ├── configs/                    # Plugin bootstrap configs
│   ├── requirements/               # Pylon requirements
│   ├── wiki_builder/               # Wiki build utilities
│   └── plugins/                    # Plugin submodules
│       ├── bootstrap/              # Platform bootstrap plugin
│       └── deepwiki_plugin/        # ⭐ MAIN PLUGIN CODE (git submodule)
│           ├── .github/            # Plugin-specific CI/CD
│           ├── plugin_implementation/  # ⭐ ACTUAL IMPLEMENTATION
│           │   ├── agents/         # Agent implementations
│           │   ├── code_graph/     # Graph building & querying
│           │   ├── parsers/        # Language-specific parsers
│           │   ├── deep_research/  # Research tool implementation
│           │   ├── wiki_structure_planner/  # Structure planning
│           │   └── ...
│           ├── methods/            # RPC method handlers
│           ├── routes/             # HTTP routes
│           ├── static/ui/          # Place for the plugin UI implementation based on react 
│           │   ├── template/       # Plugin UI implementation based on the React
│           │   ├── dist/           # Compiled react application. It is mandatory to compile it after the feature development
│           └── tests/              # Plugin unit tests
├── PLANNING_*.md                   # Planning documents
└── tests/                          # Infrastructure-level tests
```

## Key Understanding: Two Git Repositories

1. **Infrastructure repo** (`deepwiki_plugin/`):
   - Branch: `centry` (or other)
   - Contains: Docker configs, Pylon setup, planning docs, test scripts
   - Purpose: Run and develop the plugin locally

2. **Plugin repo** (`pylon_deepwiki/plugins/deepwiki_plugin/`):
   - Branch: `feat/capability-based-naming` (or other feature branches)
   - Contains: Actual plugin source code
   - Purpose: The DeepWiki implementation
   - **This is a git submodule with its own commit history**

## Working with the Code

### Where to Make Changes

| Task                             | Location                                                                               |
|----------------------------------|----------------------------------------------------------------------------------------|
| Add/modify wiki generation logic | `pylon_deepwiki/plugins/deepwiki_plugin/plugin_implementation/agents/`                 |
| Add/modify graph operations      | `pylon_deepwiki/plugins/deepwiki_plugin/plugin_implementation/code_graph/`             |
| Add/modify language parsers      | `pylon_deepwiki/plugins/deepwiki_plugin/plugin_implementation/parsers/`                |
| Add/modify structure planning    | `pylon_deepwiki/plugins/deepwiki_plugin/plugin_implementation/wiki_structure_planner/` |
| Add/modify RPC methods           | `pylon_deepwiki/plugins/deepwiki_plugin/methods/`                                      |
| Add/modify HTTP routes           | `pylon_deepwiki/plugins/deepwiki_plugin/routes/`                                       |
| Add plugin tests                 | `pylon_deepwiki/plugins/deepwiki_plugin/tests/`                                        |
| Modify Docker/infrastructure     | `deepwiki_plugin/docker-compose.yml`, `envs/`                                          |
| Add planning documents           | `deepwiki_plugin/PLANNING_*.md`                                                        |
| Add quick test scripts           | `deepwiki_plugin/test_*.py` (temporary, don't commit)                                  |
| Plugin UI development            | `deepwiki_plugin/static/ui/template/`                                                  |

### Committing Changes

**For plugin code changes:**
```bash
cd pylon_deepwiki/plugins/deepwiki_plugin
git add -A
git commit -m "feat: your change description"
git push origin feat/capability-based-naming
```

**For plugin UI code changes and development:**

> **IMPORTANT: Only commit source files, NOT build artifacts!**
> - ✅ Commit: `static/ui/template/src/**` (source files)
> - ❌ Do NOT commit: `static/ui/dist/**` (build output)
> - The `dist/` folder is automatically built by GitHub Actions CI on push to `main`
> - Run `npm run build` locally only for testing purposes

```bash
cd pylon_deepwiki/plugins/deepwiki_plugin
# Only add source files, not dist/
git add static/ui/template/src/
git commit -m "feat: your UI change description"
git push origin main
# GitHub CI will automatically build and commit dist/ artifacts
```

For local testing only (do not commit build output):
```bash
cd pylon_deepwiki/plugins/deepwiki_plugin/static/ui/template/
npm run build  # Creates dist/ for local testing only
```

**For infrastructure changes:**
```bash
# From deepwiki_plugin root
git add docker-compose.yml envs/
git commit -m "chore: infrastructure update"
```

### Running the Plugin Locally

```bash
# From deepwiki_plugin root
docker-compose up -d

# Or with specific services
docker-compose up -d pylon redis minio
```

### Running Tests

```bash
# Plugin unit tests
cd pylon_deepwiki/plugins/deepwiki_plugin
python -m pytest tests/

# Quick diagnostic scripts (from infra root)
python test_structure_tools.py
python test_expansion_quality_v2.py
```

### Python Environment (MANDATORY)

**Always use the existing `.venv` at the repository root** for development, running tests, and scripts:

```bash
# From deepwiki_plugin root
source .venv/bin/activate

# Then run any Python commands
python test_structure_tools.py
python -m pytest pylon_deepwiki/plugins/deepwiki_plugin/tests/
```

Do NOT create a new virtual environment. The `.venv` is pre-configured with all dependencies.

## Code Patterns

### Plugin Implementation Structure

```python
# plugin_implementation/agents/some_agent.py

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Constants at module level
SOME_CONSTANT = "value"

# Main implementation classes
class SomeAgent:
    """
    Docstring explaining purpose.
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    async def execute(self, input_data: Dict) -> Dict:
        """Main execution method."""
        ...
```

### Graph Operations

The graph system uses NetworkX with these conventions:

```python
# Node structure
{
    'id': 'unique_node_id',
    'symbol': {
        'name': 'ClassName',
        'kind': 'class',  # class, function, method, etc.
        'signature': 'class ClassName:',
    },
    'location': {
        'file': 'relative/path/to/file.py',
        'start_line': 10,
        'end_line': 50,
    },
    'docstring': 'Class docstring if any',
    'content': '... source code ...',
}

# Edge types
- 'calls': function/method calls
- 'imports': module imports
- 'inherits': class inheritance
- 'uses': type usage
- 'contains': parent-child (class contains method)
```

### Key Files Reference

| File | Purpose |
|------|---------|
| `wiki_graph_optimized.py` | Main wiki generation agent with graph |
| `graph_builder.py` | Code graph construction, ARCHITECTURAL_SYMBOLS |
| `structure_tools.py` | LangChain tools for structure planning |
| `structure_prompts.py` | Prompts for capability-based naming |
| `content_expander.py` | Content expansion with graph context |
| `retrievers.py` | Document retrieval strategies |
| `vectorstore.py` | Vector store operations |

### ARCHITECTURAL_SYMBOLS

Source of truth is `code_graph/graph_builder.py`:

```python
ARCHITECTURAL_SYMBOLS = {
    # Core structural elements
    'class', 'interface', 'struct', 'enum', 'trait',
    'function',  # Standalone functions
    'constant',  # Module-level constants
    'macro',     # Language macros (included in graph, excluded from expansion)
    # Documentation
    'module_doc', 'file_doc',
}
```

For graph expansion filtering, use subset without 'macro' and 'method'.

## Debugging Tips

### Graph Inspection

```python
# Quick graph stats
from plugin_implementation.code_graph.graph_builder import CodeGraphBuilder

builder = CodeGraphBuilder()
graph = builder.build_graph(repo_path)
print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

# Find specific symbols
for node_id, data in graph.nodes(data=True):
    if 'ClassName' in data.get('symbol', {}).get('name', ''):
        print(node_id, data)
```

### Cache Locations

When running locally, caches are stored in:
- Graph cache: `wiki_builder/cached_graphs/`
- Vector store: `wiki_builder/vector_stores/`
- Cloned repos: `wiki_builder/repos/`

## Related Workspace Components

This workspace includes other related projects:

| Project | Purpose |
|---------|---------|
| `alita-sdk/` | LangChain-based agent SDK |
| `elitea_core/` | Platform core APIs |
| `provider_worker/` | Toolkit operations shim |
| `indexer_worker/` | Agent execution runner |
| `AlitaUI/` | React frontend application |
| `shared/` | DB clients, S3/Minio clients, config singleton, RPC helpers |
| `artifacts/` | S3-compatible storage handler with SigV4 auth |
Each has its own git repository and development patterns. See their respective README.md files.

## Current Feature Branches

- `feat/capability-based-naming`: Enhanced wiki structure with capability-based section naming
- Implements smart graph expansion with ARCHITECTURAL_TYPES filtering
- Fixes file_path → rel_path bug in structure_tools.py
