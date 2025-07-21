"""
Optimized LangGraph Wiki Generation Agent

Following LangGraph best practices with clean state management and configuration separation.
Configuration comes from API wrapper, state only contains workflow execution data.
"""
import json
import logging
import re
import time
import traceback
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel

from plugin_implementation import GitHubIndexer, WikiRetrieverStack
from plugin_implementation.prompts.wiki_prompts_enhanced import ENHANCED_WIKI_STRUCTURE_PROMPT, TARGET_AUDIENCES, \
    ENHANCED_CONTENT_GENERATION_PROMPT, ENHANCED_RETRY_CONTENT_PROMPT, ENHANCED_REPO_ANALYSIS_PROMPT, \
    QUALITY_ASSESSMENT_PROMPT
from plugin_implementation.state.wiki_state import WikiStyle, TargetAudience, WikiState, WikiStructureSpec, \
    PageGenerationState, WikiPage, QualityAssessmentState, QualityAssessment, EnhancementState, PageSpec, \
    RepositoryAnalysis

logger = logging.getLogger(__name__)


class OptimizedWikiGenerationAgent:
    """Optimized wiki generation agent with proper configuration management"""
    
    def __init__(self,
                 indexer: GitHubIndexer,
                 retriever_stack: WikiRetrieverStack,
                 llm: BaseLanguageModel,
                 repository_url: str,
                 branch: str = "main",
                 alita_client: Optional[Any] = None,
                 bucket_name: str = "wiki_artifacts",
                 # Basic parameters (formerly WikiConfiguration)
                 wiki_style: WikiStyle = WikiStyle.COMPREHENSIVE,
                 target_audience: TargetAudience = TargetAudience.MIXED,
                 # Quality parameters
                 quality_threshold: float = 0.7,
                 require_diagrams: bool = True,
                 require_code_examples: bool = True,
                 min_content_length: int = 300,
                 # Processing parameters
                 max_concurrent_pages: int = 4,
                 max_retries: int = 3,
                 enable_progress_tracking: bool = True,
                 # Advanced features
                 enable_quality_enhancement: bool = True,
                 enable_diagram_optimization: bool = True,
                 enable_structure_analysis: bool = True):
        
        # Validate all required components
        if not indexer:
            raise ValueError("GitHubIndexer is required")
        if not retriever_stack:
            raise ValueError("WikiRetrieverStack is required")
        if not llm:
            raise ValueError("BaseLanguageModel is required")
        if not repository_url:
            raise ValueError("Repository URL is required")
        
        # Store configuration as instance variables (not in state)
        self.indexer = indexer
        self.retriever_stack = retriever_stack
        self.llm = llm
        self.repository_url = repository_url
        self.branch = branch
        self.alita_client = alita_client
        self.bucket_name = bucket_name
        
        # Store wiki configuration parameters directly
        self.wiki_style = wiki_style
        self.target_audience = target_audience
        self.quality_threshold = quality_threshold
        self.require_diagrams = require_diagrams
        self.require_code_examples = require_code_examples
        self.min_content_length = min_content_length
        self.max_concurrent_pages = max_concurrent_pages
        self.max_retries = max_retries
        self.enable_progress_tracking = enable_progress_tracking
        self.enable_quality_enhancement = enable_quality_enhancement
        self.enable_diagram_optimization = enable_diagram_optimization
        self.enable_structure_analysis = enable_structure_analysis
        
        # Log component initialization for debugging
        logger.info(f"OptimizedWikiGenerationAgent initialized with:")
        logger.info(f"  - Repository: {repository_url} (branch: {branch})")
        logger.info(f"  - Indexer: {type(indexer).__name__}")
        logger.info(f"  - Retriever: {type(retriever_stack).__name__}")
        logger.info(f"  - LLM: {type(llm).__name__}")
        logger.info(f"  - Alita Client: {type(alita_client).__name__ if alita_client else 'None'}")
        
        # Build code_graph
        self.graph = self._build_graph()
        
        # Progress tracking callback (if available)
        try:
            from langchain_core.callbacks import dispatch_custom_event
            self.dispatch_event = dispatch_custom_event
        except ImportError:
            def fallback_dispatch(event_type, data):
                logger.info(f"Progress: {event_type} - {data.get('message', '')}")
            self.dispatch_event = fallback_dispatch

    # Node implementations (using self for configuration access)
    
    def analyze_repository(self, state: WikiState, config: RunnableConfig) -> Dict[str, Any]:
        """Analyze repository using EXISTING indexer data (no double indexing)"""
        
        # Configuration comes from self, not state
        repo_url = self.repository_url
        branch = self.branch
        
        # Validate components before starting
        if not self.indexer:
            raise ValueError("WikiIndexer is required but not provided")
        if not self.llm:
            raise ValueError("BaseLanguageModel (LLM) is required but not provided")
        
        logger.info(f"Starting repository analysis for {repo_url} (branch: {branch})")
        logger.info(f"Using LLM: {type(self.llm).__name__}")
        logger.info(f"Using Indexer: {type(self.indexer).__name__}")
        
        self._dispatch_progress("wiki_analysis_started", {
            "progress_percentage": 0.0,
            "message": f"🔍 Starting repository analysis: {repo_url}",
            "phase": "repository_analysis"
        })

        try:
            # Get documents from indexer (already indexed)
            all_documents = self.indexer.get_all_documents()
            if not all_documents:
                raise ValueError("No documents found in indexer - ensure repository is indexed first")
            
            # Get index statistics
            index_stats = getattr(self.indexer, 'last_index_stats', {}) or {}
            logger.info(f"Retrieved {len(all_documents)} documents from indexer")

            # Extract unique sources and create repository tree
            unique_sources = list(set(doc.metadata.get('source', '') for doc in all_documents))
            repository_tree = self._create_repository_tree(unique_sources)

            # Extract full README content
            readme_content = self._extract_full_readme_content(all_documents)

            # Create basic repository analysis
            repo_analysis_obj = self._create_repository_analysis(all_documents, index_stats)

            # Use LLM to create comprehensive analysis text (sync version)
            repo_analysis_text = self._llm_analyze_repository(
                repository_tree, readme_content, all_documents, unique_sources, config
            )

            # Store rich context in state (not configuration parameters)
            return {
                "repository_analysis": repo_analysis_obj,
                "repository_context": repo_analysis_text,
                "repository_tree": repository_tree,
                "readme_content": readme_content,
                "current_phase": "repository_analysis_complete",
                # "messages": [f"Repository analysis complete: {len(unique_sources)} files analyzed with LLM insights"]
            }

        except Exception as e:
            logger.error(f"Repository analysis failed: {e}", exc_info=True)
            self._dispatch_progress("wiki_error", {
                "message": f"❌ Repository analysis failed: {str(e)}",
                "phase": "repository_analysis"
            })
            return {
                "errors": [f"Repository analysis failed: {e}"],
                "current_phase": "repository_analysis_failed"
            }

    def generate_wiki_structure(self, state: WikiState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate wiki structure using LLM analysis with complete repository context"""

        # Configuration comes from self
        repo_analysis = state.get("repository_analysis")
        repository_tree = state.get("repository_tree", "")
        readme_content = state.get("readme_content", "")
        repo_context = state.get("repository_context", "")

        if not repo_analysis and not repo_context:
            error_msg = "Repository analysis missing from state"
            logger.error(error_msg)
            return {"errors": [error_msg]}

        try:
            # Validate LLM is available
            if not self.llm:
                raise ValueError("LLM is required for structure generation")

            logger.info(f"Starting structure generation with LLM: {type(self.llm).__name__}")

            # Create prompt with repository analysis
            structure_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert technical documentation architect specializing in comprehensive repository analysis. Your task is to create complete documentation structures that capture every significant component without omission. You must ensure comprehensive coverage of all repository elements, creating structures that scale appropriately based on actual codebase complexity and component count."),
                ("human", ENHANCED_WIKI_STRUCTURE_PROMPT)
            ])

            # Get target audience description
            target_audience = TARGET_AUDIENCES.get(
                self.target_audience.value,
                "Mixed audience with varied technical backgrounds"
            )

            # Use the LLM analysis directly (not formatting needed)
            repo_analysis_str = repo_context or str(repo_analysis)

            # Get wiki type - default to comprehensive
            wiki_type = self.wiki_style.value  # Could be made configurable if needed

            logger.info("Invoking LLM for structure generation...")

            struct_llm = self.llm.with_structured_output(WikiStructureSpec)

            response = struct_llm.invoke(
                structure_prompt.format_messages(
                    repository_tree=repository_tree,
                    readme_content=readme_content,
                    repo_analysis=repo_analysis_str,
                    target_audience=target_audience,
                    wiki_type=wiki_type
                ),
                config=config
            )

            logger.info(f"WikiStructureSpec created successfully")
            logger.info(f"Wiki structure sections: {len(response.sections)}")

            return {
                "wiki_structure_spec": response,
                "structure_planning_complete": True,
                "current_phase": "structure_complete",
            }

        except Exception as e:
            logger.error(f"Structure generation failed: {e}", exc_info=True)
            self._dispatch_progress("wiki_error", {
                "message": f"❌ Structure generation failed: {str(e)}",
                "phase": "structure_planning"
            })
            return {
                "errors": [f"Structure generation failed: {e}"],
                "current_phase": "structure_failed"
            }

    def dispatch_page_generation(self, state: WikiState, config: RunnableConfig) -> List[Send]:
        """Dispatch parallel page generation using Send"""

        logger.info("📄 Dispatching page generation")

        if not state.get("wiki_structure_spec"):
            logger.error("No wiki_structure_spec in state")
            return []

        wiki_structure = state["wiki_structure_spec"]

        # Generate all pages at once (simplified - no batching)
        sends = []
        for section_idx, section in enumerate(wiki_structure.sections):
            for page_idx, page in enumerate(section.pages):
                page_id = f"{section_idx}#{page_idx}"
                page_spec_dict = page.dict() if hasattr(page, 'dict') else page.__dict__

                sends.append(Send("generate_page_content", {
                    "page_id": page_id,
                    "page_spec": page_spec_dict,
                    "repository_context": state["repository_context"]
                }))

        logger.info(f"📄 Dispatching {len(sends)} page generation tasks")
        return sends

    def generate_page_content(self, state: PageGenerationState, config: RunnableConfig) -> Dict[str, Any]:
        """Generate content for a single page using LLM"""

        # Configuration comes from self
        page_id = state["page_id"]
        page_spec_dict = state["page_spec"]
        repo_context = state["repository_context"]

        # Recreate PageSpec object from dict
        from ..state.wiki_state import PageSpec
        page_spec = PageSpec(**page_spec_dict) if isinstance(page_spec_dict, dict) else page_spec_dict

        # Create meaningful display ID for progress tracking
        display_page_id = f"{page_spec.page_name}"

        try:
            # Get relevant content for this page (enhanced with location mapping)
            relevant_content = self._get_relevant_content_for_page(
                page_spec, self.repository_url, repo_context
            )

            # Generate content using LLM with enhanced location-aware prompt
            content_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert technical writer creating comprehensive, enterprise-grade documentation. Your task is to produce detailed technical content that covers ALL aspects of the assigned components without omission. You must create substantial documentation that provides complete coverage of functionality, implementation details, and usage patterns based exclusively on the provided repository context."),
                ("human", ENHANCED_CONTENT_GENERATION_PROMPT)
            ])

            target_audience = TARGET_AUDIENCES.get(
                self.target_audience.value,
                "Mixed audience"
            )

            # Use sync invoke instead of async ainvoke
            response = self.llm.invoke(
                content_prompt.format_messages(
                    section_name=page_spec.page_name.split('/')[0] if '/' in page_spec.page_name else "Main",
                    page_name=page_spec.page_name,
                    page_description=page_spec.description,
                    content_focus=page_spec.content_focus,
                    repository_url=self.repository_url,
                    wiki_style=self.wiki_style.value,
                    repository_context=repo_context,
                    relevant_content=relevant_content["content"],
                    related_files="\\n".join(relevant_content["files"]),
                    target_audience=target_audience
                ),
                config=config
            )

            generated_content = response.content

            return {
                "wiki_pages": [WikiPage(
                    page_id=page_id,
                    title=page_spec.page_name,
                    content=generated_content,
                    status="completed",
                    retry_count=0
                )],
                # "messages": [f"Generated content for {display_page_id}"]
            }

        except Exception as e:
            logger.error(f"Page generation failed for {display_page_id}: {e}")

            return {
                "wiki_pages": [WikiPage(
                    page_id=page_id,
                    title=page_spec.page_name,
                    content="",
                    status="failed",
                    retry_count=0
                )],
                "errors": [f"Page generation failed for {display_page_id}: {e}"]
            }

    def dispatch_quality_assessment(self, state: WikiState) -> List[Send]:
        """Dispatch parallel quality assessment tasks using Send pattern"""

        wiki_pages = state.get("wiki_pages", [])

        # Get only completed pages for quality assessment
        completed_pages = [page for page in wiki_pages
                          if page.status == "completed" and not hasattr(page, 'quality_assessment')]

        if not completed_pages:
            logger.info("📊 No pages to assess, proceeding to enhancement dispatch")
            return self.dispatch_enhancement(state)

        # Create assessment tasks using Send pattern
        sends = []
        for page in completed_pages:
            sends.append(Send("assess_page_quality", {
                "page_id": page.page_id,
                "page_content": page.content,
                "page_title": page.title
            }))

        logger.info(f"📊 Dispatching {len(sends)} quality assessment tasks")
        return sends

    def dispatch_enhancement(self, state: WikiState) -> List[Send]:
        """Dispatch parallel enhancement tasks for pages that failed quality gates"""

        quality_assessments = state.get("quality_assessments", [])

        # Find pages that failed quality assessment and need enhancement
        failed_assessments = []
        for qa in quality_assessments:
            assessment = qa.get("assessment")
            if assessment and not assessment.passes_quality_gate:
                # Check retry limits
                retry_counts = state.get("retry_counts", {})
                if retry_counts.get(qa["page_id"], 0) < self.max_retries:
                    failed_assessments.append(qa)

        if not failed_assessments:
            logger.info("🎯 No pages need enhancement, proceeding to finalization")
            return [Send("finalize_wiki", state)]

        # Create enhancement tasks using Send pattern
        sends = []
        for qa in failed_assessments:
            # Find the corresponding page content
            wiki_pages = state.get("wiki_pages", [])
            page = next((p for p in wiki_pages if p.page_id == qa["page_id"]), None)

            if page:
                sends.append(Send("enhance_page_content", {
                    "page_id": qa["page_id"],
                    "page_content": page.content,
                    "page_title": page.title,
                    "quality_assessment": qa["assessment"],
                    "repository_context": state["repository_context"]
                }))

        logger.info(f"🔧 Dispatching {len(sends)} enhancement tasks")
        return sends

    async def assess_page_quality(self, state: QualityAssessmentState, config: RunnableConfig) -> Dict[str, Any]:
        """Assess quality of a single page"""

        page_id = state["page_id"]
        content = state["page_content"]
        page_title = state["page_title"]

        logger.info(f"📋 Assessing quality of page: {page_title}")

        try:
            assessment = await self._assess_page_quality(page_id, content)

            return {
                "quality_assessments": [{
                    "page_id": page_id,
                    "assessment": assessment
                }]
            }

        except Exception as e:
            logger.error(f"Quality assessment failed for {page_id}: {e}")

            # Return default assessment
            default_assessment = QualityAssessment(
                scores={"overall": 0.5},
                overall_score=0.5,
                strengths=[],
                weaknesses=[f"Assessment failed: {e}"],
                improvement_suggestions=[],
                passes_quality_gate=False,
                confidence_level=0.0
            )

            return {
                "quality_assessments": [{
                    "page_id": page_id,
                    "assessment": default_assessment
                }]
            }

    async def enhance_page_content(self, state: EnhancementState, config: RunnableConfig) -> Dict[str, Any]:
        """Enhance content for a single page that failed quality gates"""

        page_id = state["page_id"]
        content = state["page_content"]
        page_title = state["page_title"]
        quality_assessment = state["quality_assessment"]
        repo_context = state["repository_context"]

        logger.info(f"🔧 Enhancing content for page: {page_title}")

        try:
            # Create enhancement prompt using existing page content and assessment
            enhancement_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a technical writer improving content based on quality feedback."),
                ("human", ENHANCED_RETRY_CONTENT_PROMPT)
            ])

            response = await self.llm.ainvoke(
                enhancement_prompt.format_messages(
                    original_content=content,
                    repo_name=self.repository_url,
                    section_title=page_title,
                    audience=self.target_audience.value if hasattr(self.target_audience, 'value') else str(self.target_audience),
                    enhancement_focus="Quality improvements based on assessment feedback",
                    validation_feedback=json.dumps(quality_assessment.dict(), indent=2),
                    quality_issues="\\n".join(quality_assessment.weaknesses),
                    improvement_requirements="\\n".join(quality_assessment.improvement_suggestions)
                ),
                config=config
            )

            enhanced_content = response.content

            return {
                "enhanced_pages": [WikiPage(
                    page_id=page_id,
                    title=page_title,
                    content=enhanced_content,
                    status="enhanced",
                    retry_count=0,
                    generated_at=datetime.now()
                )],
                "messages": [f"Enhanced content for {page_title}"]
            }

        except Exception as e:
            logger.error(f"Content enhancement failed for {page_id}: {e}")

            return {
                "enhanced_pages": [WikiPage(
                    page_id=page_id,
                    title=page_title,
                    content=content,  # Keep original content
                    status="enhancement_failed",
                    retry_count=0,
                    generated_at=datetime.now()
                )],
                "errors": [f"Enhancement failed for {page_title}: {e}"]
            }

    def finalize_wiki(self, state: WikiState, config: RunnableConfig) -> Dict[str, Any]:
        """Finalize wiki generation and prepare for export"""

        logger.info("🏁 Finalizing wiki generation")

        try:
            # Consolidate all pages from different phases
            wiki_pages = state.get("wiki_pages", [])
            enhanced_pages = state.get("enhanced_pages", [])
            quality_assessments = state.get("quality_assessments", [])

            # Create lookup for enhanced pages
            enhanced_lookup = {page.page_id: page for page in enhanced_pages}

            # Create lookup for quality assessments
            assessment_lookup = {qa["page_id"]: qa["assessment"] for qa in quality_assessments}

            # Merge pages with enhancements and assessments
            final_pages = []
            for page in wiki_pages:
                # Use enhanced version if available
                if page.page_id in enhanced_lookup:
                    final_page = enhanced_lookup[page.page_id]
                else:
                    final_page = page

                # Add quality assessment if available
                if page.page_id in assessment_lookup:
                    final_page.quality_assessment = assessment_lookup[page.page_id]

                final_pages.append(final_page)

            # Calculate final statistics
            wiki_structure = state["wiki_structure_spec"]
            total_pages = len(final_pages)
            total_diagrams = sum(page.content.count("```mermaid") for page in final_pages)

            # Calculate average quality from pages with assessments
            pages_with_quality = [page for page in final_pages if hasattr(page, 'quality_assessment') and page.quality_assessment]
            avg_quality = (
                sum(page.quality_assessment.overall_score for page in pages_with_quality) / len(pages_with_quality)
                if pages_with_quality else 0.0
            )

            # Count enhanced pages
            enhanced_count = len([page for page in final_pages if page.status == "enhanced"])

            # Create generation summary
            generation_summary = {
                "wiki_title": wiki_structure.wiki_title,
                "total_sections": len(wiki_structure.sections),
                "total_pages": total_pages,
                "total_diagrams": total_diagrams,
                "enhanced_pages": enhanced_count,
                "average_quality_score": avg_quality,
                "generation_timestamp": datetime.now().isoformat(),
                "execution_time": time.time() - state.get("start_time", time.time())
            }

            return {
                "wiki_pages": final_pages,
                "generation_summary": generation_summary,
                "current_phase": "finalization_complete",
                "messages": [f"Wiki finalized: {total_pages} pages, {enhanced_count} enhanced, {total_diagrams} diagrams"]
            }

        except Exception as e:
            logger.error(f"Wiki finalization failed: {e}")
            return {
                "errors": [f"Finalization failed: {e}"],
                "current_phase": "finalization_failed"
            }

    def export_wiki(self, state: WikiState, config: RunnableConfig) -> Dict[str, Any]:
        """Export wiki to artifacts"""

        self._dispatch_progress("wiki_export", {
            "progress_percentage": 95.0,
            "message": "📦 Exporting wiki artifacts",
            "phase": "export"
        })

        try:
            if self.alita_client:
                from ..artifact_export import ArtifactExporter

                exporter = ArtifactExporter(self.alita_client, self.bucket_name)

                # Convert WikiPage objects to format expected by exporter
                wiki_pages = state.get("wiki_pages", [])
                generated_pages = {page.page_id: page.content for page in wiki_pages}

                # Export wiki with all generated content
                artifacts = exporter.export_wiki_artifacts(
                    wiki_structure=state["wiki_structure_spec"],
                    wiki_pages=wiki_pages,
                    repo_metadata={
                        "repository_url": self.repository_url,
                        "branch": self.branch,
                        "generation_summary": state.get("generation_summary", {})
                    },
                    export_formats=["json", "markdown"]
                )

                self._dispatch_progress("wiki_export_complete", {
                    "progress_percentage": 100.0,
                    "message": f"🎉 Wiki export complete: {len(artifacts)} artifacts created",
                    "phase": "complete",
                    "details": {"artifacts": list(artifacts.keys())}
                })

                return {
                    "export_artifacts": list(artifacts.values()),
                    "export_complete": True,
                    "messages": [f"Export complete: {len(artifacts)} artifacts created"]
                }
            else:
                return {
                    "export_complete": True,
                    "messages": ["No client available for artifact export"]
                }

        except Exception as e:
            logger.error(f"Wiki export failed: {e}")
            return {
                "errors": [f"Export failed: {e}"],
                "export_complete": False
            }

    def _build_graph(self) -> CompiledStateGraph:
        """Build optimized wiki generation code_graph with clean map-reduce chain"""

        # Create code_graph (no config schema needed since config comes from self)
        builder = StateGraph(WikiState)

        # Add nodes
        builder.add_node("analyze_repository", self.analyze_repository)
        builder.add_node("generate_wiki_structure", self.generate_wiki_structure)
        builder.add_node("generate_page_content", self.generate_page_content)
        # builder.add_node("assess_page_quality", self.assess_page_quality)
        # builder.add_node("enhance_page_content", self.enhance_page_content)
        builder.add_node("finalize_wiki", self.finalize_wiki)
        builder.add_node("export_wiki", self.export_wiki)

        # Define clean map-reduce chain workflow
        builder.add_edge(START, "analyze_repository")
        builder.add_edge("analyze_repository", "generate_wiki_structure")

        # Phase 1: Page Generation (map)
        builder.add_conditional_edges(
            "generate_wiki_structure",
            self.dispatch_page_generation,
            ["generate_page_content"]
        )

        # Phase 2: Quality Assessment (map)
        # builder.add_conditional_edges(
        #     "generate_page_content",
        #     self.dispatch_quality_assessment,
        #     ["assess_page_quality", "enhance_page_content", "finalize_wiki"]
        # )
        #
        # # Phase 3: Enhancement (map)
        # builder.add_conditional_edges(
        #     "assess_page_quality",
        #     self.dispatch_enhancement,
        #     ["enhance_page_content", "finalize_wiki"]
        # )

        # Final steps (Reduce)
        # builder.add_edge("enhance_page_content", "finalize_wiki")
        builder.add_edge("generate_page_content", "finalize_wiki")
        builder.add_edge("finalize_wiki", "export_wiki")
        builder.add_edge("export_wiki", END)

        # Compile with checkpointing
        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    def generate_wiki(self, user_message: str) -> Dict[str, Any]:
        """Generate wiki using optimized LangGraph workflow with user message"""

        # Initialize state with clean phase-based structure (only workflow data, no config)
        initial_state: WikiState = {
            "start_time": time.time(),
            "current_phase": "initialization",

            # Repository analysis results (state data)
            "repository_analysis": None,
            "repository_context": None,
            "repository_tree": None,
            "readme_content": None,

            # Wiki structure results (state data)
            "wiki_structure_spec": None,
            "structure_planning_complete": False,

            # Phase 1: Content Generation (operator.add accumulation)
            "wiki_pages": [],

            # Phase 2: Quality Assessment (operator.add accumulation)
            "quality_assessments": [],

            # Phase 3: Enhancement (operator.add accumulation)
            "enhanced_pages": [],

            # Retry tracking (state data)
            "retry_counts": {},

            # Final results (state data)
            "generation_summary": None,
            "export_artifacts": [],
            "export_complete": False,

            # Monitoring (operator.add accumulation)
            "messages": [],
            "errors": []
        }

        # Execute workflow with user message (standard LangGraph pattern)
        try:
            runnable_config = {
                'max_concurrency': 8,
                'configurable': {
                    'thread_id': uuid.uuid4(),
                }
            }

            # Use sync invoke instead of async ainvoke
            final_state = self.graph.invoke({
                'messages': [{'role': 'user', 'content': user_message}]
            }, runnable_config)

            # Convert WikiPage objects to expected format for response
            wiki_pages = final_state.get("wiki_pages", [])
            generated_pages = {page.page_id: page.content for page in wiki_pages}

            # Convert quality assessments to dict format
            quality_assessments = {}
            for page in wiki_pages:
                if hasattr(page, 'quality_assessment') and page.quality_assessment:
                    # Convert QualityAssessment to dict if it's a Pydantic model
                    if hasattr(page.quality_assessment, 'dict'):
                        quality_assessments[page.page_id] = page.quality_assessment.dict()
                    else:
                        quality_assessments[page.page_id] = page.quality_assessment

            # Convert WikiStructureSpec to dict format for JSON serialization
            wiki_structure = final_state.get("wiki_structure_spec")
            wiki_structure_dict = wiki_structure.dict() if wiki_structure and hasattr(wiki_structure, 'dict') else wiki_structure

            return {
                "success": True,
                "wiki_structure": wiki_structure_dict,
                "generated_pages": generated_pages,
                "generation_summary": final_state.get("generation_summary", {}),
                "quality_assessments": quality_assessments,
                "export_artifacts": final_state.get("export_artifacts", []),
                "execution_time": time.time() - initial_state["start_time"],
                "messages": final_state.get("messages", []),
                "errors": final_state.get("errors", [])
            }

        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error(f"Wiki generation failed: Error parsing payload params: {stacktrace}")
            logger.error(f"Wiki generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - initial_state["start_time"]
            }

    # Helper Methods

    def _create_repository_analysis(self, documents: List[Any], index_stats: Dict[str, Any]) -> RepositoryAnalysis:
        """Create structured repository analysis from documents and stats"""

        # Extract file information
        all_sources = [doc.metadata.get('source', '') for doc in documents]
        unique_files = list(set(all_sources))

        # Detect file types
        file_types = {}
        for source in unique_files:
            if '.' in source:
                ext = source.split('.')[-1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1

        # Detect programming languages
        code_extensions = {
            'py': 'Python', 'js': 'JavaScript', 'ts': 'TypeScript',
            'java': 'Java', 'cpp': 'C++', 'c': 'C', 'go': 'Go',
            'rs': 'Rust', 'rb': 'Ruby', 'php': 'PHP', 'cs': 'C#'
        }

        languages = []
        for ext, count in file_types.items():
            if ext in code_extensions and count > 0:
                languages.append(code_extensions[ext])

        # Detect README
        readme_files = [f for f in unique_files if 'readme' in f.lower()]
        readme_content = ""
        if readme_files:
            readme_docs = [doc for doc in documents if doc.metadata.get('source', '').lower() in [f.lower() for f in readme_files]]
            if readme_docs:
                readme_content = readme_docs[0].page_content[:1000]  # First 1000 chars

        # Determine project complexity
        complexity = "simple"
        if len(unique_files) > 50 or len(languages) > 2:
            complexity = "moderate"
        if len(unique_files) > 200 or len(languages) > 3:
            complexity = "complex"

        return RepositoryAnalysis(
            total_files=len(unique_files),
            programming_languages=languages[:5],  # Top 5 languages
            primary_language=languages[0] if languages else "Unknown",
            has_readme=len(readme_files) > 0,
            readme_content=readme_content,
            total_symbols=index_stats.get("code_analysis", {}).get("total_symbols", 0),
            file_types=file_types,
            project_complexity=complexity,
            key_directories=self._extract_key_directories(unique_files),
            main_modules=self._extract_main_modules(unique_files),
            documentation_files=[f for f in unique_files if any(doc_ext in f.lower() for doc_ext in ['md', 'rst', 'txt', 'doc'])],
            configuration_files=[f for f in unique_files if any(cfg_ext in f.lower() for cfg_ext in ['json', 'yaml', 'yml', 'toml', 'ini', 'cfg'])],
            test_files=[f for f in unique_files if 'test' in f.lower() or 'spec' in f.lower()]
        )

    def _extract_key_directories(self, file_paths: List[str]) -> List[str]:
        """Extract key directory names from file paths"""
        directories = set()
        for path in file_paths:
            parts = path.split('/')
            if len(parts) > 1:
                # Add top-level directories
                directories.add(parts[0])
                # Add second-level directories for better context
                if len(parts) > 2:
                    directories.add(f"{parts[0]}/{parts[1]}")

        return sorted(list(directories))[:20]  # Top 20 directories

    def _extract_main_modules(self, file_paths: List[str]) -> List[str]:
        """Extract main module names from file paths"""
        modules = set()
        for path in file_paths:
            if path.endswith('.py'):
                # Python modules
                module_name = path.replace('/', '.').replace('.py', '')
                if not module_name.startswith('.') and 'test' not in module_name:
                    modules.add(module_name)
            elif any(path.endswith(ext) for ext in ['.js', '.ts']):
                # JavaScript/TypeScript modules
                if '/' in path:
                    module_name = path.split('/')[-1].split('.')[0]
                    modules.add(module_name)

        return sorted(list(modules))[:50]  # Top 50 modules

    def _parse_llm_json_response(self, response_content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, handling potential formatting issues"""
        try:
            # Try direct JSON parsing first
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\\n(.*?)\\n```', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try to extract any JSON-like content
            json_match = re.search(r'\\{.*\\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            # Fallback: create basic structure that matches WikiStructureSpec
            logger.warning("Could not parse LLM JSON response, using fallback structure")
            return {
                "wiki_title": "Generated Wiki",
                "overview": "Documentation for the repository",
                "sections": [{
                    "section_name": "Documentation",
                    "section_order": 1,
                    "description": "Main documentation",
                    "rationale": "Fallback structure due to parsing error",
                    "pages": [{
                        "page_name": "Overview",
                        "page_order": 1,
                        "description": "Repository overview",
                        "content_focus": "General information",
                        "rationale": "Basic overview page for fallback structure",
                        "user_questions": ["What is this project?", "How do I get started?"],
                        "search_keywords": ["overview", "introduction", "getting started"],
                        "estimated_complexity": "simple",
                        "target_folders": [],
                        "key_files": [],
                        "code_references": ""
                    }]
                }],
                "total_pages": 1,
                "estimated_effort": "low"
            }

    def _get_relevant_content_for_page(self, page_spec: PageSpec, repo_url: str, repo_context: str) -> Dict[str, Any]:
        """Get relevant content for page generation with proper context structure and imports"""

        # Build enhanced query using page specification and location fields
        query_parts = [
            page_spec.page_name,
            page_spec.description,
            page_spec.content_focus
        ]

        # Add location-specific terms to improve retrieval
        if hasattr(page_spec, 'target_folders') and page_spec.target_folders:
            folder_terms = [folder.split('/')[-1] for folder in page_spec.target_folders if folder]
            query_parts.extend(folder_terms)

        if hasattr(page_spec, 'key_files') and page_spec.key_files:
            file_terms = [file.split('/')[-1].replace('.py', '').replace('.js', '') for file in page_spec.key_files if file]
            query_parts.extend(file_terms)

        # Add search keywords if available
        if page_spec.search_keywords:
            query_parts.extend(page_spec.search_keywords[:5])

        query = " ".join(query_parts)

        try:
            # Use balanced k value for comprehensive coverage without token bloat
            relevant_docs = self.retriever_stack.search_repository(query, k=45)

            # Group documents by file path for proper structuring
            docs_by_file = {}
            documentation_docs = []
            
            for doc in relevant_docs:
                source = doc.metadata.get('source', 'unknown')
                
                # Separate documentation from code files
                if self._is_documentation_file(source):
                    documentation_docs.append(doc)
                else:
                    if source not in docs_by_file:
                        docs_by_file[source] = []
                    docs_by_file[source].append(doc)

            # Prioritize files based on page specification
            prioritized_files = []
            other_files = []
            
            for file_path in docs_by_file.keys():
                is_priority = False
                
                # Check if file matches target folders or key files
                if hasattr(page_spec, 'target_folders') and page_spec.target_folders:
                    for folder in page_spec.target_folders:
                        if folder and folder in file_path:
                            is_priority = True
                            break
                
                if hasattr(page_spec, 'key_files') and page_spec.key_files:
                    for key_file in page_spec.key_files:
                        if key_file and key_file in file_path:
                            is_priority = True
                            break
                
                if is_priority:
                    prioritized_files.append(file_path)
                else:
                    other_files.append(file_path)

            # Build structured context
            context_parts = []
            
            # Documentation context section
            if documentation_docs:
                context_parts.append("Documentation Context:")
                for doc in documentation_docs[:8]:  # Increased from 5 to 8 for better coverage
                    source = doc.metadata.get('source', 'unknown')
                    content = doc.page_content[:1000]  # Reasonable content length
                    context_parts.append(f"<documentation_source: {source}>")
                    context_parts.append(content)
                    context_parts.append("</documentation_source>")
                context_parts.append("")

            # Code context section
            if docs_by_file:
                context_parts.append("Code Context:")
                
                # Process prioritized files first, then others
                selected_files = prioritized_files[:20] + other_files[:20]  # Increased from 15 to 20 for better coverage
                
                for file_path in selected_files:
                    file_docs = docs_by_file[file_path]
                    
                    # Combine all content from this file
                    combined_content = "\\n".join(doc.page_content for doc in file_docs)
                    
                    # Extract imports using the existing context expander
                    imports = self._extract_imports_for_file(file_path, combined_content)
                    
                    # Limit content to reasonable size
                    # if len(combined_content) > 2000:
                    #     combined_content = combined_content[:2000] + "\\n\\n... (content truncated)"
                    
                    context_parts.append(f"<code_source: {file_path}>")
                    if imports:
                        context_parts.append("<imports>")
                        context_parts.append(imports)
                        context_parts.append("</imports>")
                    context_parts.append("<implementation>")
                    context_parts.append(combined_content)
                    context_parts.append("</implementation>")
                    context_parts.append("</code_source>")
                    context_parts.append("")

            # Build file list for tracking
            all_files = list(prioritized_files) + list(other_files)
            file_list = all_files[:30]  # Increased from 20 to 30 for better relationship building
            
            # Build location matches for feedback
            location_matches = []
            if hasattr(page_spec, 'target_folders') and page_spec.target_folders:
                for folder in page_spec.target_folders:
                    if any(folder in fp for fp in prioritized_files):
                        location_matches.append(f"📁 {folder}")
            
            if hasattr(page_spec, 'key_files') and page_spec.key_files:
                for key_file in page_spec.key_files:
                    if any(key_file in fp for fp in prioritized_files):
                        location_matches.append(f"📄 {key_file}")

            return {
                "content": "\\n".join(context_parts),
                "files": file_list,
                "location_matches": list(set(location_matches)),
                "prioritized_sources": len(prioritized_files),
                "documentation_files": len(documentation_docs),
                "code_files": len(docs_by_file)
            }

        except Exception as e:
            logger.warning(f"Could not retrieve relevant content: {e}")
            return {
                "content": repo_context,  # Fallback to repo context
                "files": [],
                "location_matches": [],
                "prioritized_sources": 0,
                "documentation_files": 0,
                "code_files": 0
            }

        except Exception as e:
            logger.warning(f"Could not retrieve relevant content: {e}")
            return {
                "content": repo_context,  # Fallback to repo context
                "files": [],
                "location_matches": [],
                "prioritized_sources": 0
            }

    async def _assess_page_quality(self, page_id: str, content: str) -> QualityAssessment:
        """Assess quality of a generated page using LLM"""

        try:
            quality_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a technical documentation quality assessor."),
                ("human", QUALITY_ASSESSMENT_PROMPT)
            ])

            section_name, page_name = page_id.split('/', 1) if '/' in page_id else ("Main", page_id)
            target_audience = TARGET_AUDIENCES.get(self.target_audience.value, "Mixed audience")

            response = await self.llm.ainvoke(
                quality_prompt.format_messages(
                    content=content,
                    page_name=page_name,
                    section_name=section_name,
                    target_audience=target_audience
                )
            )

            # Parse assessment response
            assessment_data = self._parse_llm_json_response(response.content)
            return QualityAssessment(**assessment_data)

        except Exception as e:
            logger.error(f"Quality assessment failed for {page_id}: {e}")
            # Return default assessment
            return QualityAssessment(
                scores={"overall": 0.6},
                overall_score=0.6,
                strengths=[],
                weaknesses=[f"Assessment error: {e}"],
                improvement_suggestions=[],
                passes_quality_gate=False,
                confidence_level=0.0
            )

    def _dispatch_progress(self, event_type: str, data: Dict[str, Any]):
        """Dispatch progress event if tracking is enabled"""
        if self.enable_progress_tracking:
            self.dispatch_event(event_type, data)

    def _extract_full_readme_content(self, documents: List[Any]) -> str:
        """Extract full README content by combining all README chunks"""
        readme_docs = [doc for doc in documents if 'readme' in doc.metadata.get('source', '').lower()]

        if not readme_docs:
            return "No README file found"

        # Group by source file and sort by chunk order if available
        readme_by_file = {}
        for doc in readme_docs:
            source = doc.metadata.get('source', '')
            if source not in readme_by_file:
                readme_by_file[source] = []
            readme_by_file[source].append(doc)

        # Use the first README file found (usually README.md)
        first_readme_source = list(readme_by_file.keys())[0]
        readme_chunks = readme_by_file[first_readme_source]

        # Sort by section_id if available (for proper reconstruction)
        readme_chunks.sort(key=lambda x: x.metadata.get('section_id', 0))

        # Combine all chunks
        full_content = "\\n\\n".join([chunk.page_content for chunk in readme_chunks])
        return full_content  # Limit to first 5000 chars for LLM context

    def _create_repository_tree(self, unique_sources: List[str]) -> str:
        """Create a simplified repository tree structure"""
        if not unique_sources:
            return "No files found"

        # Sort files for better tree structure
        sorted_sources = sorted(unique_sources)

        # Create a simple tree representation
        tree_lines = []
        current_dir = ""

        for source in sorted_sources:  # Limit to first 50 files
            if '/' in source:
                dir_parts = source.split('/')[:-1]
                file_name = source.split('/')[-1]
                dir_path = '/'.join(dir_parts)

                if dir_path != current_dir:
                    tree_lines.append(f"📁 {dir_path}/")
                    current_dir = dir_path

                tree_lines.append(f"  📄 {file_name}")
            else:
                tree_lines.append(f"📄 {source}")

        return "\\n".join(tree_lines)

    def _llm_analyze_repository(self, repository_tree: str, readme_content: str,
                               all_documents: List[Any], unique_sources: List[str], config: RunnableConfig) -> str:
        """Use LLM to analyze repository architecture, patterns, and technologies"""

        # Prepare code samples for analysis
        code_samples = self._extract_representative_code_samples(all_documents)

        # Prepare file statistics
        file_stats = self._prepare_basic_file_stats(unique_sources)

        # Create repository analysis prompt using enhanced version
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert software architect and comprehensive repository analysis specialist. Your mission is to perform exhaustive analysis of codebases, documenting every significant component without omission. You must analyze ALL architectural patterns, technologies, and relationships present in the repository to enable complete documentation coverage."),
            ("human", ENHANCED_REPO_ANALYSIS_PROMPT)
        ])

        logger.info("Invoking LLM for comprehensive repository analysis...")

        try:
            response = self.llm.invoke(
                analysis_prompt.format_messages(
                    repository_tree=repository_tree,
                    readme_content=readme_content if readme_content else "No README found",
                    code_samples=code_samples,
                    file_stats=file_stats
                ),
                config=config
            )

            analysis_text = response.content
            logger.info(f"LLM repository analysis completed: {len(analysis_text)} characters")
            return analysis_text

        except Exception as e:
            logger.error(f"LLM repository analysis failed: {e}")
            raise

    def _extract_representative_code_samples(self, all_documents: List[Any]) -> str:
        """Extract representative code samples for LLM analysis"""

        code_samples = []

        # Get samples from different file types
        python_files = [doc for doc in all_documents if doc.metadata.get('source', '').endswith('.py')]
        config_files = [doc for doc in all_documents if any(ext in doc.metadata.get('source', '').lower()
                                                            for ext in ['yaml', 'json', 'toml', 'ini'])]
        doc_files = [doc for doc in all_documents if doc.metadata.get('source', '').endswith('.md')]

        # Sample Python code (max 3 files)
        for doc in python_files[:3]:
            source = doc.metadata.get('source', 'unknown')
            content = doc.page_content[:800]  # First 800 chars
            code_samples.append(f"=== {source} ===\\n{content}\\n")

        # Sample configuration files (max 2)
        for doc in config_files[:2]:
            source = doc.metadata.get('source', 'unknown')
            content = doc.page_content[:400]  # First 400 chars
            code_samples.append(f"=== {source} ===\\n{content}\\n")

        # Sample documentation files (max 2, excluding README)
        for doc in doc_files[:2]:
            source = doc.metadata.get('source', 'unknown')
            if 'readme' not in source.lower():
                content = doc.page_content[:600]  # First 600 chars
                code_samples.append(f"=== {source} ===\\n{content}\\n")

        combined_samples = "\\n".join(code_samples)

        # Limit total length to avoid context overflow
        if len(combined_samples) > 4000:
            combined_samples = combined_samples[:4000] + "\\n... [truncated for context length]"

        return combined_samples if combined_samples else "No representative code samples available"

    def _prepare_basic_file_stats(self, unique_sources: List[str]) -> str:
        """Prepare basic file statistics for LLM analysis"""

        # Count file types
        file_types = {}
        for source in unique_sources:
            if '.' in source:
                ext = source.split('.')[-1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1

        # Sort by count
        sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)

        stats_lines = [
            f"Total Files: {len(unique_sources)}",
            "File Type Distribution:"
        ]

        for ext, count in sorted_types[:50]:  # Top 10 file types
            stats_lines.append(f"  - .{ext}: {count} files")
        
        return "\\n".join(stats_lines)

    def _is_documentation_file(self, file_path: str) -> bool:
        """Check if file is documentation"""
        doc_extensions = {'.md', '.rst', '.txt', '.doc', '.docx'}
        doc_names = {'readme', 'changelog', 'license', 'contributing', 'docs'}
        
        path_lower = file_path.lower()
        
        # Check extension
        if any(path_lower.endswith(ext) for ext in doc_extensions):
            return True
        
        # Check filename
        from pathlib import Path
        filename = Path(file_path).stem.lower()
        if filename in doc_names:
            return True
        
        # Check if in docs directory
        if '/docs/' in path_lower or '/documentation/' in path_lower:
            return True
        
        return False
    
    def _extract_imports_for_file(self, file_path: str, content: str) -> str:
        """Extract import statements from code content"""
        import re
        from pathlib import Path
        
        # Determine language from file extension
        extension = Path(file_path).suffix.lower()
        
        # Define import patterns for different languages
        import_patterns = {
            '.py': [
                r'^from\s+([^\s]+)\s+import\s+(.+)$',
                r'^import\s+([^\s,]+)(?:\s+as\s+[^\s,]+)?(?:\s*,\s*[^\s,]+(?:\s+as\s+[^\s,]+)?)*$'
            ],
            '.js': [
                r'^import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*\s+=\s+require\([\'"]([^\'"]+)[\'"]\)',
                r'^import\s+[\'"]([^\'"]+)[\'"]'
            ],
            '.jsx': [
                r'^import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*\s+=\s+require\([\'"]([^\'"]+)[\'"]\)',
                r'^import\s+[\'"]([^\'"]+)[\'"]'
            ],
            '.ts': [
                r'^import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*\s+=\s+require\([\'"]([^\'"]+)[\'"]\)',
                r'^import\s+[\'"]([^\'"]+)[\'"]'
            ],
            '.tsx': [
                r'^import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
                r'^const\s+.*\s+=\s+require\([\'"]([^\'"]+)[\'"]\)',
                r'^import\s+[\'"]([^\'"]+)[\'"]'
            ],
            '.java': [
                r'^import\s+([^\s;]+);?$',
                r'^import\s+static\s+([^\s;]+);?$'
            ]
        }
        
        patterns = import_patterns.get(extension, [])
        if not patterns:
            return "Import patterns not supported for this file type."
        
        imports = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check against import patterns for the language
            for pattern in patterns:
                if re.match(pattern, line):
                    imports.append(line)
                    break
        
        if not imports:
            return "No imports found in this file."
        
        return '\n'.join(imports)
