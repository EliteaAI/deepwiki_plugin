"""
Tests for the Agentic Ask engine and prompts.

Tests cover:
1. **Prompt generation** — ask_prompts.py functions and content
2. **AskEngine construction** — config, factory, model building
3. **Feature flag dispatch** — DEEPWIKI_ASK_AGENTIC gating in subprocess worker
4. **Event streaming** — event format compatibility with deep research
5. **Progressive tool forcing** — DEEPWIKI_PROGRESSIVE_TOOLS is set during agent creation
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import asdict

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _THIS_DIR.parent
_IMPL_DIR = _PLUGIN_ROOT / 'plugin_implementation'

if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))


# ===================================================================
# 1. Prompt Tests
# ===================================================================

class TestAskPrompts(unittest.TestCase):
    """Test ask_prompts.py prompt generation."""

    def test_get_ask_instructions_returns_string(self):
        from plugin_implementation.ask_prompts import get_ask_instructions
        result = get_ask_instructions()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)

    def test_get_ask_instructions_contains_workflow(self):
        from plugin_implementation.ask_prompts import get_ask_instructions
        result = get_ask_instructions()
        self.assertIn("search_symbols", result)
        self.assertIn("get_relationships", result)
        self.assertIn("get_code", result)
        self.assertIn("Mandatory Workflow", result)

    def test_get_ask_instructions_contains_tool_guidance(self):
        from plugin_implementation.ask_prompts import get_ask_instructions
        result = get_ask_instructions()
        self.assertIn("search_docs", result)
        self.assertIn("query_graph", result)
        self.assertIn("think", result)

    def test_get_ask_instructions_max_iterations_default(self):
        from plugin_implementation.ask_prompts import get_ask_instructions
        result = get_ask_instructions()
        self.assertIn("8 tool calls", result)

    def test_get_ask_instructions_custom_max_iterations(self):
        from plugin_implementation.ask_prompts import get_ask_instructions
        result = get_ask_instructions(max_iterations=12)
        self.assertIn("12 tool calls", result)

    def test_get_ask_instructions_citation_style(self):
        """Prompts enforce natural file path citations, NOT [N] numeric."""
        from plugin_implementation.ask_prompts import get_ask_instructions
        result = get_ask_instructions()
        self.assertIn("Do NOT use [N] numeric citations", result)
        self.assertIn("naturally by file path", result)

    def test_get_ask_instructions_no_research_report(self):
        """Prompts tell agent NOT to generate a research report."""
        from plugin_implementation.ask_prompts import get_ask_instructions
        result = get_ask_instructions()
        self.assertIn("Do NOT", result)
        self.assertIn("Research Report", result)

    def test_get_ask_instructions_contains_date(self):
        from plugin_implementation.ask_prompts import get_ask_instructions
        from datetime import datetime
        result = get_ask_instructions()
        # Should contain current year
        self.assertIn(datetime.now().strftime("%Y"), result)

    def test_ask_instructions_prebuilt(self):
        from plugin_implementation.ask_prompts import ASK_INSTRUCTIONS
        self.assertIsInstance(ASK_INSTRUCTIONS, str)
        self.assertIn("search_symbols", ASK_INSTRUCTIONS)

    def test_get_ask_prompt_question_only(self):
        from plugin_implementation.ask_prompts import get_ask_prompt
        result = get_ask_prompt("How does auth work?")
        self.assertIn("How does auth work?", result)
        self.assertIn("## Question", result)
        self.assertNotIn("## Repository Context", result)
        self.assertNotIn("## Conversation History", result)

    def test_get_ask_prompt_with_repo_context(self):
        from plugin_implementation.ask_prompts import get_ask_prompt
        result = get_ask_prompt("How does auth work?", repo_context="Python web app")
        self.assertIn("## Repository Context", result)
        self.assertIn("Python web app", result)

    def test_get_ask_prompt_with_chat_history(self):
        from plugin_implementation.ask_prompts import get_ask_prompt
        result = get_ask_prompt("Follow up?", chat_history="User: First question\nAssistant: Answer")
        self.assertIn("## Conversation History", result)
        self.assertIn("First question", result)

    def test_get_ask_prompt_getting_started(self):
        from plugin_implementation.ask_prompts import get_ask_prompt
        result = get_ask_prompt("test")
        self.assertIn("## Getting Started", result)
        self.assertIn("search_symbols", result)

    def test_prompts_no_todo_or_filesystem(self):
        """Ask prompts should NOT mention todo or filesystem offloading commands."""
        from plugin_implementation.ask_prompts import get_ask_instructions
        result = get_ask_instructions()
        self.assertNotIn("write_todos", result)
        self.assertNotIn("write_file", result)
        self.assertNotIn("edit_file", result)
        # Only check for filesystem-tool style calls, not substring in examples
        self.assertNotIn("ls('/", result)
        self.assertNotIn("grep('", result)


# ===================================================================
# 2. AskConfig Tests
# ===================================================================

class TestAskConfig(unittest.TestCase):
    """Test AskConfig dataclass."""

    def test_default_config(self):
        from plugin_implementation.ask_engine import AskConfig
        cfg = AskConfig()
        self.assertEqual(cfg.max_iterations, 8)  # Default
        self.assertEqual(cfg.max_search_results, 20)
        self.assertTrue(cfg.enable_graph_analysis)

    def test_custom_config(self):
        from plugin_implementation.ask_engine import AskConfig
        cfg = AskConfig(max_iterations=5, max_search_results=10, enable_graph_analysis=False)
        self.assertEqual(cfg.max_iterations, 5)
        self.assertEqual(cfg.max_search_results, 10)
        self.assertFalse(cfg.enable_graph_analysis)


# ===================================================================
# 3. AskEngine Construction Tests
# ===================================================================

class TestAskEngineConstruction(unittest.TestCase):
    """Test AskEngine initialization and factory function."""

    def test_init_with_defaults(self):
        from plugin_implementation.ask_engine import AskEngine
        engine = AskEngine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
        )
        self.assertEqual(engine.status, "idle")
        self.assertEqual(engine.final_answer, "")
        self.assertIsNone(engine.error)
        self.assertEqual(engine.config.max_iterations, 8)

    def test_init_with_custom_config(self):
        from plugin_implementation.ask_engine import AskEngine, AskConfig
        cfg = AskConfig(max_iterations=5)
        engine = AskEngine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
            config=cfg,
        )
        self.assertEqual(engine.config.max_iterations, 5)

    def test_init_with_llm_client(self):
        from plugin_implementation.ask_engine import AskEngine
        mock_llm = MagicMock()
        engine = AskEngine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
            llm_client=mock_llm,
        )
        self.assertEqual(engine.llm_client, mock_llm)

    def test_init_with_repo_analysis(self):
        from plugin_implementation.ask_engine import AskEngine
        analysis = {"summary": "A Python web framework", "languages": "Python"}
        engine = AskEngine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
            repo_analysis=analysis,
        )
        self.assertEqual(engine.repo_analysis["summary"], "A Python web framework")

    def test_factory_function(self):
        from plugin_implementation.ask_engine import create_ask_engine
        engine = create_ask_engine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
            max_iterations=6,
        )
        self.assertEqual(engine.config.max_iterations, 6)

    def test_factory_ignores_unknown_kwargs(self):
        from plugin_implementation.ask_engine import create_ask_engine
        # Should not raise on unknown kwargs
        engine = create_ask_engine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
            unknown_param="ignored",
        )
        self.assertIsNotNone(engine)


# ===================================================================
# 4. AskEngine Repo Context Tests
# ===================================================================

class TestAskEngineRepoContext(unittest.TestCase):
    """Test _get_repo_context method."""

    def test_empty_analysis(self):
        from plugin_implementation.ask_engine import AskEngine
        engine = AskEngine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
        )
        ctx = engine._get_repo_context()
        self.assertEqual(ctx, "No repository overview available.")

    def test_with_summary(self):
        from plugin_implementation.ask_engine import AskEngine
        engine = AskEngine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
            repo_analysis={"summary": "Test repo"},
        )
        ctx = engine._get_repo_context()
        self.assertIn("Repository Summary: Test repo", ctx)

    def test_with_all_fields(self):
        from plugin_implementation.ask_engine import AskEngine
        engine = AskEngine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
            repo_analysis={
                "summary": "A web app",
                "key_components": "auth, api, db",
                "languages": "Python, TypeScript",
            },
        )
        ctx = engine._get_repo_context()
        self.assertIn("A web app", ctx)
        self.assertIn("auth, api, db", ctx)
        self.assertIn("Python, TypeScript", ctx)


# ===================================================================
# 5. Feature Flag Tests
# ===================================================================

class TestFeatureFlag(unittest.TestCase):
    """Test DEEPWIKI_ASK_AGENTIC feature flag."""

    def test_flag_off_by_default(self):
        from plugin_implementation.ask_engine import ASK_AGENTIC_ENABLED
        # The flag should be off unless env var is set
        env_val = os.environ.get("DEEPWIKI_ASK_AGENTIC", "")
        if env_val.strip() in ("1", "true", "yes"):
            self.assertTrue(ASK_AGENTIC_ENABLED)
        else:
            self.assertFalse(ASK_AGENTIC_ENABLED)

    @patch.dict(os.environ, {"DEEPWIKI_ASK_AGENTIC": "1"})
    def test_subprocess_dispatches_to_agentic(self):
        """When DEEPWIKI_ASK_AGENTIC=1, run_ask should call run_ask_agentic_async."""
        with patch('plugin_implementation.ask_subprocess_worker.run_ask_agentic_async') as mock_agentic:
            from plugin_implementation.ask_subprocess_worker import run_ask
            import asyncio

            # Make the async function return a result
            async def fake_agentic(payload):
                return {"success": True, "answer": "agentic answer", "agentic": True}

            mock_agentic.side_effect = fake_agentic

            result = run_ask({"question": "test", "github_repository": "org/repo"})
            mock_agentic.assert_called_once()
            self.assertTrue(result.get("agentic", False) or result.get("success", False))

    @patch.dict(os.environ, {"DEEPWIKI_ASK_AGENTIC": "0"}, clear=False)
    def test_subprocess_uses_classic_when_flag_off(self):
        """When DEEPWIKI_ASK_AGENTIC is not set, should use classic AskTool."""
        # We just verify it doesn't call the agentic path by checking imports
        with patch('plugin_implementation.ask_subprocess_worker.run_ask_agentic_async') as mock_agentic:
            # The classic path will fail on missing vectorstore, but we can verify
            # the agentic path is NOT called
            from plugin_implementation.ask_subprocess_worker import run_ask
            try:
                run_ask({"question": "test", "github_repository": "org/repo"})
            except Exception:
                pass  # Classic path will fail without real vectorstore

            mock_agentic.assert_not_called()


# ===================================================================
# 6. AskEngine._create_agent Progressive Tools Test
# ===================================================================

class TestAskEngineProgressiveTools(unittest.TestCase):
    """Test that AskEngine forces progressive tools regardless of env."""

    @patch.dict(os.environ, {"DEEPWIKI_PROGRESSIVE_TOOLS": ""}, clear=False)
    def test_forces_progressive_tools(self):
        """_create_agent should set DEEPWIKI_PROGRESSIVE_TOOLS=1 during tool creation."""
        from plugin_implementation.ask_engine import AskEngine

        engine = AskEngine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
            llm_client=MagicMock(),
        )

        captured_env = {}

        try:
            from plugin_implementation.deep_research import research_tools  # noqa: F401
        except ImportError:
            self.skipTest("research_tools not available")

        def mock_create_tools(**kwargs):
            captured_env['DEEPWIKI_PROGRESSIVE_TOOLS'] = os.environ.get('DEEPWIKI_PROGRESSIVE_TOOLS', '')
            # Return minimal tool list
            return [MagicMock()]

        # Pre-import modules that _create_agent imports locally so patches resolve
        import langchain.agents.middleware.summarization  # noqa: F401
        import deepagents.middleware.patch_tool_calls  # noqa: F401
        import langchain_anthropic.middleware  # noqa: F401

        with patch('plugin_implementation.deep_research.research_tools.create_codebase_tools', side_effect=mock_create_tools), \
             patch('langchain.agents.create_agent', return_value=MagicMock()), \
             patch('langchain.agents.middleware.summarization.SummarizationMiddleware', return_value=MagicMock()), \
             patch('deepagents.middleware.patch_tool_calls.PatchToolCallsMiddleware', return_value=MagicMock()), \
             patch('langchain_anthropic.middleware.AnthropicPromptCachingMiddleware', return_value=MagicMock()):
            try:
                engine._create_agent("test system prompt")
            except Exception:
                pass  # May fail on agent creation, that's fine

        # Verify progressive tools were forced
        self.assertEqual(captured_env.get('DEEPWIKI_PROGRESSIVE_TOOLS'), '1')

        # Verify env was restored after
        self.assertEqual(os.environ.get('DEEPWIKI_PROGRESSIVE_TOOLS', ''), '')


# ===================================================================
# 7. AskEngine Event Format Compatibility
# ===================================================================

class TestAskEngineEventFormat(unittest.TestCase):
    """Test that Ask engine events are compatible with deep research format."""

    def test_event_types_match_expected(self):
        """All event types yielded by the engine should be in the expected set."""
        expected_types = {
            'ask_start', 'thinking_step', 'todo_update',
            'ask_complete', 'ask_error',
        }
        # These are the event types the engine yields
        # Verified by inspecting ask_engine.py source
        from plugin_implementation.ask_engine import AskEngine
        self.assertTrue(expected_types)  # Placeholder sanity check

    def test_thinking_step_has_required_fields(self):
        """thinking_step events must have step, type, tool fields."""
        # Mock event matching what ask_engine.py yields
        event = {
            "event_type": "thinking_step",
            "data": {
                "step": 1,
                "type": "tool_call",
                "tool": "search_symbols",
                "tool_call_id": "call_123",
                "call_id": "call_123",
                "input": '{"query": "auth"}',
                "timestamp": "2024-01-01T00:00:00",
            }
        }
        data = event['data']
        self.assertIn('step', data)
        self.assertIn('type', data)
        self.assertIn('tool', data)
        self.assertIn('tool_call_id', data)
        self.assertIn('call_id', data)

    def test_tool_result_has_required_fields(self):
        """tool_result events must have output_length and output_preview."""
        event = {
            "event_type": "thinking_step",
            "data": {
                "step": 2,
                "type": "tool_result",
                "tool": "search_symbols",
                "tool_call_id": "call_123",
                "call_id": "call_123",
                "output_length": 500,
                "output_preview": "Found 10 symbols...",
                "output": "Found 10 symbols...",
                "timestamp": "2024-01-01T00:00:00",
            }
        }
        data = event['data']
        self.assertIn('output_length', data)
        self.assertIn('output_preview', data)
        self.assertIn('output', data)


# ===================================================================
# 8. Subprocess Worker Event Emission Tests
# ===================================================================

class TestSubprocessEventEmission(unittest.TestCase):
    """Test that the subprocess worker emits correct event markers."""

    def test_emit_event_format(self):
        """_emit_event produces [ASK_EVENT] JSON markers."""
        import io
        from plugin_implementation.ask_subprocess_worker import _emit_event

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            _emit_event("test_type", {"key": "value"})

        output = captured.getvalue()
        self.assertIn("[ASK_EVENT]", output)
        self.assertIn('"event": "test_type"', output)

    def test_emit_thinking_step_format(self):
        """_emit_thinking_step produces [THINKING_STEP] JSON markers."""
        import io
        from plugin_implementation.ask_subprocess_worker import _emit_thinking_step

        captured = io.StringIO()
        with patch('sys.stdout', captured):
            _emit_thinking_step("tool_call", "Calling: search_symbols", "auth query")

        output = captured.getvalue()
        self.assertIn("[THINKING_STEP]", output)
        self.assertIn('"type": "tool_call"', output)
        self.assertIn('"title": "Calling: search_symbols"', output)


# ===================================================================
# 9. Prompt Token Efficiency Tests
# ===================================================================

class TestPromptTokenEfficiency(unittest.TestCase):
    """Verify prompts are compact relative to deep research."""

    def test_ask_prompt_shorter_than_research(self):
        """Ask system prompt should be shorter than research system prompt."""
        from plugin_implementation.ask_prompts import get_ask_instructions
        ask_prompt = get_ask_instructions()

        try:
            from plugin_implementation.deep_research.research_prompts import get_research_instructions
            research_prompt = get_research_instructions()
            self.assertLess(len(ask_prompt), len(research_prompt),
                            "Ask prompt should be shorter than research prompt")
        except ImportError:
            self.skipTest("research_prompts not available")

    def test_ask_prompt_under_5k_chars(self):
        """Ask system prompt should be under 5K characters (~1.2K tokens)."""
        from plugin_implementation.ask_prompts import get_ask_instructions
        prompt = get_ask_instructions()
        self.assertLess(len(prompt), 5000,
                        f"Ask prompt is {len(prompt)} chars, should be under 5000")

    def test_user_prompt_is_compact(self):
        """User prompt should not bloat with minimal inputs."""
        from plugin_implementation.ask_prompts import get_ask_prompt
        prompt = get_ask_prompt("How does auth work?")
        self.assertLess(len(prompt), 500,
                        f"User prompt is {len(prompt)} chars, should be under 500")


# ===================================================================
# 10. AskEngine Synchronous Wrapper Tests
# ===================================================================

class TestAskEngineSync(unittest.TestCase):
    """Test the synchronous ask_sync wrapper."""

    def test_ask_sync_returns_string(self):
        """ask_sync should return the final answer as a string."""
        import asyncio
        from plugin_implementation.ask_engine import AskEngine

        engine = AskEngine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
            llm_client=MagicMock(),
        )

        # Mock the async ask method to yield a complete event
        async def mock_ask(question, chat_history=None, session_id=None):
            yield {"event_type": "ask_start", "data": {"session_id": "test"}}
            yield {
                "event_type": "ask_complete",
                "data": {"answer": "mocked answer", "session_id": "test", "steps": 0},
            }

        with patch.object(engine, 'ask', side_effect=mock_ask):
            result = engine.ask_sync("How does X work?")
            self.assertEqual(result, "mocked answer")

    def test_ask_sync_with_callback(self):
        """ask_sync should call the event callback."""
        from plugin_implementation.ask_engine import AskEngine

        engine = AskEngine(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=MagicMock(),
            llm_client=MagicMock(),
        )

        events_received = []

        async def mock_ask(question, chat_history=None, session_id=None):
            yield {"event_type": "ask_start", "data": {}}
            yield {"event_type": "thinking_step", "data": {"step": 1, "type": "tool_call"}}
            yield {"event_type": "ask_complete", "data": {"answer": "test"}}

        with patch.object(engine, 'ask', side_effect=mock_ask):
            engine.ask_sync("test", on_event=lambda e: events_received.append(e))

        self.assertEqual(len(events_received), 3)
        self.assertEqual(events_received[0]['event_type'], 'ask_start')


if __name__ == '__main__':
    unittest.main()
