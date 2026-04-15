"""
Deep Research Engine - Using langchain-ai/deepagents

This is the main orchestration for deep research using the ACTUAL DeepAgents library.
It leverages:
- TodoListMiddleware for planning and progress tracking (write_todos, read_todos)
- FilesystemMiddleware for context offloading (outputs >20K tokens → files)
- SubAgentMiddleware for delegation to specialist agents (task tool)
- Custom tools wrapping WikiRetrieverStack and GraphManager

Events are captured via LangGraph's astream with stream_mode=["messages", "updates"]
rather than LangChain callbacks, as DeepAgents is built on LangGraph.
"""

import logging
from typing import Any, Optional, Callable, Dict, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from deepagents import create_deep_agent
from deepagents.backends import StateBackend

from .research_tools import create_codebase_tools
from .research_prompts import (
    RESEARCH_INSTRUCTIONS,
    get_research_prompt
)

logger = logging.getLogger(__name__)

# Safety limits
MAX_ITERATIONS = 15


@dataclass
class ResearchConfig:
    """Configuration for a research session"""
    max_iterations: int = MAX_ITERATIONS
    max_search_results: int = 15
    enable_graph_analysis: bool = True
    research_type: str = 'general'
    enable_subagents: bool = False  # Disabled - subagents add complexity without clear benefit


class DeepResearchEngine:
    """
    Main engine for conducting deep research using the actual DeepAgents library.
    
    This creates a proper DeepAgents agent with:
    - TodoListMiddleware for planning (write_todos, read_todos)
    - FilesystemMiddleware for context offloading (read_file, write_file, etc.)
    - Custom tools for codebase search and graph analysis (search_codebase, get_symbol_relationships, think)
    
    Events are captured using LangGraph's native astream with dual stream mode
    (messages + updates) - NOT LangChain callbacks.
    
    Usage:
        engine = DeepResearchEngine(
            retriever_stack=wiki_retriever_stack,
            graph_manager=graph_manager,
            code_graph=loaded_graph,
            llm_settings=llm_config
        )
        
        async for event in engine.research("How does auth work?"):
            handle_event(event)
    """
    
    def __init__(
        self,
        retriever_stack: Any,  # WikiRetrieverStack
        graph_manager: Any,    # GraphManager
        code_graph: Any,       # NetworkX graph
        repo_analysis: Optional[Dict] = None,
        llm_client: Optional[BaseChatModel] = None,
        backend: Optional[object] = None,
        llm_settings: Optional[Dict] = None,
        config: Optional[ResearchConfig] = None
    ):
        """
        Initialize the research engine.
        
        Args:
            retriever_stack: WikiRetrieverStack for codebase search
            graph_manager: GraphManager for relationship analysis
            code_graph: Loaded NetworkX code graph
            repo_analysis: Pre-loaded repository analysis
            llm_settings: LLM configuration dict
            config: Research configuration
        """
        self.retriever_stack = retriever_stack
        self.graph_manager = graph_manager
        self.code_graph = code_graph
        self.repo_analysis = repo_analysis or {}
        self.llm_client = llm_client
        self.backend = backend
        self.llm_settings = llm_settings or {}
        self.config = config or ResearchConfig()
        
        # Session tracking (simplified - no redundant state)
        self.session_id: Optional[str] = None
        self.question: Optional[str] = None
        self.final_report: str = ""
        self.current_todos: list = []
        self.status: str = "idle"  # idle, running, completed, failed
        self.error: Optional[str] = None
    
    def _build_model(self) -> BaseChatModel:
        """Build the LLM from settings.
        
        Always uses ChatOpenAI because we're going through a LiteLLM proxy
        that handles auth with JWT tokens and routes to the appropriate backend
        (OpenAI, Anthropic, etc.) based on the model name.
        """
        api_base = self.llm_settings.get('api_base') or self.llm_settings.get('openai_api_base')
        api_key = self.llm_settings.get('api_key') or self.llm_settings.get('openai_api_key')
        model_name = self.llm_settings.get('model_name', 'gpt-4o-mini')
        max_tokens = self.llm_settings.get('max_tokens', 16384)  # Default to 16k for comprehensive reports
        
        # Always use ChatOpenAI for proxy-based setups
        # The proxy handles routing to Anthropic/OpenAI/etc based on model name
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=api_base,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    
    def _create_agent(self):
        """
        Create the DeepAgents agent with all middleware and tools.
        
        Uses create_deep_agent which provides:
        - TodoListMiddleware (write_todos, read_todos)
        - FilesystemMiddleware (ls, read_file, write_file, edit_file, glob, grep)
        - SummarizationMiddleware (auto-summarize at 170K tokens)
        """
        model = self.llm_client or self._build_model()
        
        # Create our custom codebase tools (no callback - we get events from stream)
        # Pass FTS5 index from graph_manager if available (uses property for feature-flag gating)
        fts_index = getattr(self.graph_manager, 'fts_index', None) if self.graph_manager else None
        custom_tools = create_codebase_tools(
            retriever_stack=self.retriever_stack,
            graph_manager=self.graph_manager,
            code_graph=self.code_graph,
            repo_analysis=self.repo_analysis,
            event_callback=None,  # Not needed - using LangGraph stream
            graph_text_index=fts_index,
        )
        
        # Create the DeepAgents agent (no subagents - they add complexity without benefit)
        agent = create_deep_agent(
            model=model,
            tools=custom_tools,
            system_prompt=RESEARCH_INSTRUCTIONS,
            subagents=[],  # Disabled - single agent with parallel tool calling is more efficient
            backend=(self.backend if self.backend is not None else (lambda rt: StateBackend(rt))),
        )
        
        return agent
    
    async def research(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Conduct deep research on a question.
        
        Uses LangGraph's astream with stream_mode=["messages", "updates"] to capture:
        - Tool calls and results (messages stream)
        - Todo updates from TodoListMiddleware (updates stream)
        - State changes (updates stream)
        
        Args:
            question: The research question
            session_id: Optional session identifier
            
        Yields:
            Event dictionaries for UI updates
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.question = question
        self.status = "running"
        self.final_report = ""
        self.current_todos = []
        self.error = None
        
        step_count = 0
        
        # Yield start event
        yield {
            'event_type': 'research_start',
            'data': {
                'session_id': self.session_id,
                'question': question,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        try:
            # Create the agent
            agent = self._create_agent()
            
            # Build research prompt with context
            repo_context = self._get_repo_context()
            research_prompt = get_research_prompt(
                research_type=self.config.research_type,
                topic=question,
                context=repo_context
            )
            
            # Prepare input message
            stream_input = {"messages": [{"role": "user", "content": research_prompt}]}
            
            # Stream with dual mode: messages for tool calls, updates for todos/state
            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates"],
                # config={"recursion_limit": self.config.max_iterations * 10}
            ):
                # Handle different chunk formats
                if not isinstance(chunk, tuple) or len(chunk) != 2:
                    continue
                
                stream_mode, data = chunk
                
                # Handle UPDATES stream - todo changes and state updates
                if stream_mode == "updates":
                    if not isinstance(data, dict):
                        continue
                    
                    # Extract node updates
                    for node_name, node_data in data.items():
                        if not isinstance(node_data, dict):
                            continue
                        
                        # Check for todo updates from TodoListMiddleware
                        if "todos" in node_data:
                            new_todos = node_data["todos"]
                            if new_todos != self.current_todos:
                                self.current_todos = new_todos
                                yield {
                                    'event_type': 'todo_update',
                                    'data': {
                                        'todos': new_todos,
                                        'timestamp': datetime.now().isoformat()
                                    }
                                }
                
                # Handle MESSAGES stream - tool calls and responses
                elif stream_mode == "messages":
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue
                    
                    message, metadata = data
                    
                    # AI message with tool calls
                    if isinstance(message, AIMessage):
                        # Check for tool calls
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                step_count += 1
                                # Get tool call id for linking with result
                                tool_call_id = tool_call.get('id', f"call_{step_count}")
                                tool_name = tool_call.get('name', 'unknown')
                                yield {
                                    'event_type': 'thinking_step',
                                    'data': {
                                        'step': step_count,
                                        'type': 'tool_call',
                                        'tool': tool_name,
                                        'tool_call_id': tool_call_id,  # Include ID for linking
                                        'call_id': tool_call_id,  # Also as call_id for compatibility
                                        'input': str(tool_call.get('args', {}))[:500],
                                        'timestamp': datetime.now().isoformat()
                                    }
                                }
                        
                        # Check for final content (no tool calls = final response)
                        if message.content and not message.tool_calls:
                            self.final_report = message.content
                    
                    # Tool response message
                    elif isinstance(message, ToolMessage):
                        step_count += 1
                        content_str = str(message.content) if message.content else ""
                        # Get tool name from metadata if available
                        tool_name = getattr(message, 'name', None) or metadata.get('tool', 'tool')
                        yield {
                            'event_type': 'thinking_step',
                            'data': {
                                'step': step_count,
                                'type': 'tool_result',
                                'tool': tool_name,  # Include tool name for display
                                'tool_call_id': message.tool_call_id,
                                'call_id': message.tool_call_id,  # Also as call_id for compatibility
                                'output_length': len(content_str),
                                'output_preview': content_str[:300] + "..." if len(content_str) > 300 else content_str,
                                'output': content_str[:500],  # Also as 'output' for frontend
                                'timestamp': datetime.now().isoformat()
                            }
                        }
            
            # Research completed
            self.status = 'completed'
            
            yield {
                'event_type': 'research_complete',
                'data': {
                    'session_id': self.session_id,
                    'report': self.final_report,
                    'todos': self.current_todos,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            self.status = 'failed'
            self.error = str(e)
            
            yield {
                'event_type': 'research_error',
                'data': {
                    'session_id': self.session_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def _get_repo_context(self) -> str:
        """Get repository context string"""
        parts = []
        
        if self.repo_analysis.get('summary'):
            parts.append(f"Repository Summary: {self.repo_analysis['summary']}")
        
        if self.repo_analysis.get('key_components'):
            parts.append(f"Key Components: {self.repo_analysis['key_components']}")
        
        if self.repo_analysis.get('languages'):
            parts.append(f"Languages: {self.repo_analysis['languages']}")
        
        return "\n".join(parts) if parts else "No repository overview available."
    
    def research_sync(
        self,
        question: str,
        on_event: Optional[Callable[[Dict], None]] = None
    ) -> str:
        """
        Synchronous wrapper for deep research.
        
        Args:
            question: Research question
            on_event: Optional callback for events
            
        Returns:
            Final research report
        """
        import asyncio
        
        async def _run():
            report = ""
            async for event in self.research(question):
                if on_event:
                    on_event(event)
                if event.get('event_type') == 'research_complete':
                    report = event['data'].get('report', '')
            return report
        
        return asyncio.run(_run())
    
    def get_report(self) -> str:
        """Get the final research report"""
        return self.final_report or "Research not completed"
    
    def get_status(self) -> Dict:
        """Get current research status"""
        return {
            "session_id": self.session_id,
            "question": self.question,
            "status": self.status,
            "todos": self.current_todos,
            "error": self.error
        }


def create_deep_research_engine(
    retriever_stack: Any,
    graph_manager: Any,
    code_graph: Any,
    repo_analysis: Optional[Dict] = None,
    llm_client: Optional[BaseChatModel] = None,
    backend: Optional[object] = None,
    llm_settings: Optional[Dict] = None,
    **kwargs
) -> DeepResearchEngine:
    """
    Factory function to create a DeepResearchEngine.
    
    This is the main entry point for creating research engines.
    
    Args:
        retriever_stack: WikiRetrieverStack for codebase search
        graph_manager: GraphManager for relationship analysis
        code_graph: Loaded NetworkX code graph
        repo_analysis: Pre-loaded repository analysis
        llm_settings: LLM configuration
        **kwargs: Additional config options
        
    Returns:
        Configured DeepResearchEngine instance
    """
    config = ResearchConfig(**{k: v for k, v in kwargs.items() if hasattr(ResearchConfig, k)})
    
    return DeepResearchEngine(
        retriever_stack=retriever_stack,
        graph_manager=graph_manager,
        code_graph=code_graph,
        repo_analysis=repo_analysis,
        llm_client=llm_client,
        backend=backend,
        llm_settings=llm_settings,
        config=config
    )
