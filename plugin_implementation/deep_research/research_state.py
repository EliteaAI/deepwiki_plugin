"""
Research Configuration - Configuration for deep research.

Note: DeepAgents handles todos internally via TodoListMiddleware.
We don't need to duplicate state tracking - events are streamed directly.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ResearchConfig:
    """Configuration for deep research"""
    max_iterations: int = 30       # Maximum research iterations
    max_concurrent_tools: int = 3  # Max parallel tool calls
    enable_subagents: bool = True  # Enable specialist subagents
    research_type: str = "exploration"  # exploration, deep_dive, security_audit
    
    # Subagent configuration
    enable_code_analyzer: bool = True
    enable_architecture_expert: bool = True
    enable_improvement_planner: bool = True

