"""
Optimized State Definition for Wiki Generation

Following LangGraph best practices with clean state management.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from datetime import datetime
from pydantic import BaseModel, Field
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
    """Specification for a wiki page with vector store optimization"""
    page_name: str
    page_order: int
    description: str
    content_focus: str
    rationale: str  # Why this page is needed and what problems it solves
    user_questions: List[str] = Field(default_factory=list)  # Questions this page answers
    search_keywords: List[str] = Field(default_factory=list)  # Keywords for search optimization
    estimated_complexity: str  # simple, moderate, complex
    target_folders: List[str] = Field(default_factory=list)  # Folders in repo this page references
    key_files: List[str] = Field(default_factory=list)  # Specific files this page will reference
    code_references: str = Field(default="")
    diagram_recommendations: List[str] = Field(default_factory=list) # diagram recommendations


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
    estimated_effort: str  # low, medium, high


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
    export_artifacts: List[str]
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
