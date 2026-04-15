"""
Optimized State Definition for Wiki Generation

Following LangGraph best practices with clean state management.
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic import AliasChoices, ConfigDict, field_validator
from enum import Enum


class WikiStyle(str, Enum):
    """Wiki generation styles"""
    COMPREHENSIVE = "comprehensive"
    TECHNICAL = "technical"  
    MINIMAL = "minimal"
    TUTORIAL = "tutorial"


class TargetAudience(str, Enum):
    """Target audiences for wiki content"""
    DEVELOPERS = "developers"
    BEGINNERS = "beginners" 
    MIXED = "mixed"
    TECHNICAL = "technical"
    BUSINESS = "business"


# Remove WikiConfiguration class entirely as it's now handled by direct parameters


class PageSpec(BaseModel):
    """Specification for a wiki page with graph-based retrieval support"""
    page_name: str
    page_order: int
    description: str
    content_focus: str
    rationale: str  # Why this page is needed and what problems it solves
    target_symbols: List[str] = Field(default_factory=list)  # Exact class/function names for graph-based retrieval
    target_docs: List[str] = Field(default_factory=list)  # Documentation files for page context (e.g., README.md, docs/auth.md)
    target_folders: List[str] = Field(default_factory=list)  # Folders in repo this page references
    key_files: List[str] = Field(default_factory=list)  # Specific files this page will reference
    retrieval_query: str = Field(default="")  # Fallback query for vector store retrieval (documentation files)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Extra metadata (e.g. cluster_node_ids)


class SectionSpec(BaseModel):
    """Specification for a wiki section"""
    section_name: str
    section_order: int  
    description: str
    rationale: str
    pages: List[PageSpec]


class WikiStructureSpec(BaseModel):
    """Complete wiki structure specification"""
    wiki_title: str
    overview: str
    sections: List[SectionSpec]
    total_pages: int


class QualityAssessment(BaseModel):
    """Content quality assessment results"""
    scores: Dict[str, float]
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    passes_quality_gate: bool
    confidence_level: float


class ValidationResult(BaseModel):
    """Content validation results"""
    validation_results: Dict[str, bool]
    diagram_analysis: Dict[str, Any]
    issues_found: List[str]
    severity_level: str
    passes_validation: bool
    recommendations: List[str]


class WikiPage(BaseModel):
    """Generated wiki page data"""
    page_id: str
    title: str
    content: str
    status: str  # generating, completed, failed, enhanced
    quality_assessment: Optional[QualityAssessment] = None
    validation_result: Optional[ValidationResult] = None
    retry_count: int = 0
    generated_at: datetime = Field(default_factory=datetime.now)


class WikiPageBatch(BaseModel):
    """Batch of wiki pages for parallel processing"""
    batch_id: str
    page_ids: List[str]
    batch_status: str  # pending, processing, completed, failed
    completed_at: Optional[datetime] = None


# Main State Types

class WikiState(TypedDict):
    """Main state for wiki generation workflow - Following LangGraph best practices"""

    # Repository analysis (from indexer)
    repository_analysis: Optional[str]  # LLM analysis text
    repository_context: Optional[str]   # Formatted context for LLM
    repository_tree: Optional[str]      # Full repository tree structure
    readme_content: Optional[str]       # Complete README content
    repository_tree_stats: Optional[Dict[str, int]]  # Parsed stats footer (files, code_files, doc_files, languages)
    conceptual_summary_raw: Optional[str]            # Raw LLM JSON/text prior to parsing
    conceptual_summary: Optional[ConceptualRepositorySummary]  # Parsed structured conceptual model
    page_budget: Optional[int]                       # Adaptive page budget (min 5, max 55)
    page_budget_justification: Optional[str]         # Reasoning from LLM for budget
    budget_compression_recommendation: Optional[str] # If compression / consolidation recommended

    # Wiki structure planning (LLM-driven)
    wiki_structure_spec: Optional[WikiStructureSpec]
    structure_planning_complete: bool

    # Phase 1: Content Generation - Natural accumulation with operator.add
    wiki_pages: Annotated[List[WikiPage], operator.add]

    # Phase 2: Quality Assessment - Natural accumulation with operator.add
    quality_assessments: Annotated[List[Dict[str, Any]], operator.add]  # {page_id, assessment}

    # Phase 3: Enhancement (only for failed pages) - Natural accumulation with operator.add
    enhanced_pages: Annotated[List[WikiPage], operator.add]

    # Retry tracking
    retry_counts: Dict[str, int]

    # Progress tracking
    start_time: float
    current_phase: str

    # Final results
    export_complete: bool
    artifacts: List[dict]
    generation_summary: Optional[Dict[str, Any]]

    # Monitoring and debugging - Lists that accumulate
    messages: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]


# Parallel Processing State Types - Following LangGraph Send patterns

class PageGenerationState(TypedDict):
    """State for individual page generation tasks"""
    page_id: str
    page_spec: PageSpec
    batch_id: str
    repository_context: str


class QualityAssessmentState(TypedDict):
    """State for quality assessment tasks"""
    page_id: str
    page_content: str
    page_title: str


class EnhancementState(TypedDict):
    """State for content enhancement tasks"""
    page_id: str
    page_content: str
    page_title: str
    quality_assessment: QualityAssessment
    repository_context: str


class PageGenerationState(TypedDict):
    """State for individual page generation tasks"""
    page_id: str
    page_spec: Dict[str, Any]  # PageSpec as dict for JSON serialization
    repository_context: str


# Helper Types for Complex Operations

class RepositoryAnalysis(BaseModel):
    """Structured repository analysis results"""
    total_files: int
    programming_languages: List[str]
    primary_language: str
    has_readme: bool
    readme_content: str
    total_symbols: int
    file_types: Dict[str, int]
    project_complexity: str  # simple, moderate, complex
    key_directories: List[str]
    main_modules: List[str]
    documentation_files: List[str]
    configuration_files: List[str]
    test_files: List[str]


class ContentContext(BaseModel):
    """Context for content generation"""
    repository_summary: str
    relevant_code_snippets: List[Dict[str, str]]
    related_documentation: List[Dict[str, str]]
    file_references: List[str]
    technical_concepts: List[str]


class GenerationMetrics(BaseModel):
    """Metrics for generation performance"""
    total_pages_generated: int
    total_diagrams_created: int
    average_quality_score: float
    validation_pass_rate: float
    total_generation_time: float
    pages_per_minute: float
    retry_rate: float
    enhancement_applied: int


class ExportSummary(BaseModel):
    """Summary of exported wiki"""
    wiki_title: str
    total_sections: int
    total_pages: int
    total_diagrams: int
    average_quality_score: float
    export_formats: List[str]
    artifact_files: List[str]
    generation_timestamp: str
    quality_metrics: Dict[str, float]


# ============================================================================
# SECTION-BASED PLANNING MODELS (for agentic generator)
# ============================================================================

class SymbolSummary(BaseModel):
    """Lightweight symbol representation for planning phase.
    
    Contains just enough information for the LLM to decide which symbols
    to include in each section, without loading full code.
    Built from document metadata - no regex parsing needed.
    """
    name: str = Field(description="Symbol name, e.g., 'UserService'")
    qualified_name: str = Field(description="Full path, e.g., 'src.services.user::UserService'")
    symbol_type: str = Field(description="Type: 'class', 'function', 'interface', 'method'")
    file_path: str = Field(description="Source file path, e.g., 'src/services/user.py'")
    brief: str = Field(default="", description="First line of docstring (from metadata)")
    relationships: List[str] = Field(
        default_factory=list,
        description="Relationship strings from expansion: 'extends BaseService', 'uses UserRepository'"
    )
    estimated_tokens: int = Field(default=0, description="Estimated token count of full code")


class OverviewSection(BaseModel):
    """A single overview section for high-level content.
    
    LLM decides title and content during planning phase when full context is available.
    Examples: "Architecture Overview", "Key Concepts", "Design Patterns", "Data Flow"
    """
    title: str = Field(description="Section title - LLM decides based on topic needs")
    content: str = Field(description="Full markdown content with optional <code_context path> citations")


class DetailSectionSpec(BaseModel):
    """Specification for a detail section to be generated with focused context.
    
    These sections are generated AFTER planning, with only the relevant symbols loaded.
    """
    section_id: str = Field(description="Unique identifier for this section, e.g., 'auth_implementation'")
    title: str = Field(description="Section title, e.g., 'Authentication Implementation'")
    description: str = Field(description="What this section should cover in detail")

    # Optional intent for section-specific retrieval/expansion policy
    intent: Optional[str] = Field(
        default=None,
        description="Optional intent tag (e.g., 'graph_construction', 'api', 'config', 'runtime')"
    )
    
    # Symbols needed for this section
    primary_symbols: List[str] = Field(
        default_factory=list,
        description="Symbol names whose FULL code is needed (5-10 per section)"
    )
    supporting_symbols: List[str] = Field(
        default_factory=list,
        description="Symbol names where signatures/docstrings are sufficient"
    )
    retrieval_queries: List[str] = Field(
        default_factory=list,
        description="Search queries used to retrieve symbols for this section"
    )
    symbol_paths: List[str] = Field(
        default_factory=list,
        description="File paths for the symbols (for code citations)"
    )

    model_config = ConfigDict(extra="ignore")

    @field_validator("primary_symbols", "supporting_symbols", mode="before")
    @classmethod
    def _coerce_symbol_lists(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict):
            # Some models return {symbol: path} or {idx: symbol}; pick keys as names.
            return [str(k) for k in v.keys()]
        return v

    @field_validator("symbol_paths", mode="before")
    @classmethod
    def _coerce_symbol_paths(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict):
            # Common drift: model returns {SymbolName: "path/to/file.py"}
            return [str(x) for x in v.values() if x is not None]
        return v

    @field_validator("retrieval_queries", mode="before")
    @classmethod
    def _coerce_retrieval_queries(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict):
            return [str(x) for x in v.values() if x is not None]
        return v
    
    # Content guidance
    suggested_elements: List[str] = Field(
        default_factory=list,
        description="Suggested content elements: 'workflow diagram', 'code example', 'sequence diagram'"
    )


class PageStructure(BaseModel):
    """Complete page structure from planning - HYBRID approach.
    
    Generated during planning phase with full context:
    - Introduction (required): Multi-audience, value proposition
    - Overview sections (free-form): LLM decides count and titles
    - Detail section specs (structured): Specs for focused generation
    - Conclusion (required): Takeaways and next steps
    """
    title: str = Field(
        description="Page title",
        validation_alias=AliasChoices("title", "page_title")
    )
    
    # REQUIRED: Introduction - written during planning with complete context
    introduction: str = Field(
        description="Comprehensive introduction with WHY/WHAT/WHO, multi-audience, can include <code_context> citations"
    )
    
    # FREE-FORM: Overview sections - LLM decides how many and what titles
    overview_sections: List[OverviewSection] = Field(
        default_factory=list,
        description="Overview sections (0 or more) - LLM decides titles like 'Architecture', 'Key Concepts', etc."
    )
    
    # STRUCTURED: Detail section specifications for focused generation
    detail_sections: List[DetailSectionSpec] = Field(
        default_factory=list,
        description="Specifications for sections generated with focused context"
    )
    
    # REQUIRED: Conclusion - written during planning with complete context
    conclusion: str = Field(
        description="Comprehensive conclusion with takeaways and next steps, can include <code_context> citations"
    )

    model_config = ConfigDict(extra="ignore")


# -------------------- Phase 2 Conceptual Summary Models --------------------

class Component(BaseModel):
    name: str
    purpose: str
    key_files: List[str] = Field(default_factory=list)
    key_directories: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    maturity: str = Field(default="unknown")  # experimental|stable|legacy|unknown

class DataFlow(BaseModel):
    name: str
    description: str
    sources: List[str] = Field(default_factory=list)
    sinks: List[str] = Field(default_factory=list)
    transformation_summary: str
    critical_path: bool = False
    failure_modes: List[str] = Field(default_factory=list)

class ExtensionPoint(BaseModel):
    name: str
    mechanism: str  # plugin/hooks/config/override
    locations: List[str] = Field(default_factory=list)
    stability: str = Field(default="internal")  # stable|volatile|internal
    typical_use_cases: List[str] = Field(default_factory=list)

class GlossaryItem(BaseModel):
    term: str
    definition: str
    related_terms: List[str] = Field(default_factory=list)

class RecommendedSection(BaseModel):
    title: str
    rationale: str
    priority: int
    suggested_pages: List[str] = Field(default_factory=list)

class ConceptualRepositorySummary(BaseModel):
    high_level_architecture: str
    major_components: List[Component]
    data_flows: List[DataFlow]
    extension_points: List[ExtensionPoint]
    glossary: List[GlossaryItem]
    recommended_documentation_sections: List[RecommendedSection]
    risks: List[str]
    notes: str

