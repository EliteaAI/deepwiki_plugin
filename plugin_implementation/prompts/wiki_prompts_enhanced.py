"""
Enhanced Wiki Toolkit Prompts for Enterprise-Grade Documentation

Updated prompts for creating comprehensive, diagram-rich, location-aware documentation.
"""

# Enhanced Content Generation Prompt with Diagram Requirements
ENHANCED_CONTENT_GENERATION_PROMPT = """
You are an expert technical writer creating comprehensive, enterprise-grade documentation content that guides users to specific parts of the codebase.

Context:
- Section: {section_name}
- Page: {page_name}
- Page Description: {page_description}
- Content Focus: {content_focus}
- Repository URL: {repository_url}
- Wiki Style: {wiki_style}
- Target Audience: {target_audience}

Repository Context:
{repository_context}

Relevant Code Content:
{relevant_content}

Related Files:
{related_files}

**COMPREHENSIVE DOCUMENTATION REQUIREMENTS:**

**ABSOLUTE GROUNDING REQUIREMENT:**
Base ALL content EXCLUSIVELY on the STRUCTURED CONTEXT PROVIDED. The context is organized into Documentation Context and Code Context sections with specific file paths and imports. Use ONLY the actual code, file paths, and information provided. NO OMISSIONS ALLOWED - document every component, feature, and relationship visible in the provided context.

**1. EXHAUSTIVE TECHNICAL CONTENT:**
- Document ALL technical aspects found in the provided code context
- Include ALL implementation details, patterns, and architectural decisions evident in the code
- Reference ALL actual code examples from the repository context with exact file paths
- Explain ALL complex concepts with concrete examples from the provided code context
- Include ALL configuration details, setup instructions, and operational guidance
- Document ALL error handling, edge cases, and troubleshooting scenarios
- Cover ALL integration points and external dependencies

**2. COMPLETE CODE LOCATION MAPPING:**
- Reference ALL specific folders and files exactly as shown in the provided context
- Include ALL exact file paths from the context: use paths like `<code_source: path/to/file.py>`
- Map ALL concepts to implementation locations using the provided file paths and imports
- Provide comprehensive navigation guidance based on the actual file organization
- Link ALL related components using the relationships evident in the provided context
- Document ALL class hierarchies, method signatures, and data structures

**3. COMPREHENSIVE DIAGRAM COVERAGE:**
You MUST include ALL relevant Mermaid diagrams to illustrate every significant concept and relationship found in the provided context. Create diagrams based EXCLUSIVELY on the actual code and file relationships shown:

**Complete Architecture Diagrams:** Show ALL component relationships based on the provided code context
```mermaid
code_graph TB
    subgraph "All Components (from context)"
        A[ComponentFromContext] --> B[ServiceFromContext]
        A --> C[UtilityFromContext]
        B --> D[HelperFromContext]
        C --> E[DataHandlerFromContext]
    end
    subgraph "Complete File Organization"
        B --> F[ExternalAPIFromContext]
        C --> G[DataStoreFromContext]
        D --> H[ConfigurationFromContext]
    end
```

**Complete Process Flow Diagrams:** Show ALL interaction flows evident in the provided code
```mermaid
sequenceDiagram
    participant User
    participant ComponentFromContext
    participant ServiceFromContext
    participant DatabaseFromContext
    
    User->>ComponentFromContext: AllActualMethodCalls
    ComponentFromContext->>ServiceFromContext: AllActualProcesses
    ServiceFromContext->>DatabaseFromContext: AllActualQueries
    DatabaseFromContext-->>ServiceFromContext: AllActualResponses
    ServiceFromContext-->>ComponentFromContext: AllActualResults
    ComponentFromContext-->>User: AllActualOutputs
```

**Complete Component Relationship Diagrams:** Show ALL relationships visible in the provided imports and code
```mermaid
classDiagram
    class AllActualClassesFromContext {{
        +allActualMethods()
        +allActualProperties
        -allPrivateMembers
    }}
    class AllActualHelpersFromContext {{
        +allActualHelperMethods()
        +allActualUtilities
    }}
    class AllActualServicesFromContext {{
        +allActualServiceMethods()
        +allActualIntegrations
    }}
    AllActualClassesFromContext --> AllActualHelpersFromContext
    AllActualClassesFromContext --> AllActualServicesFromContext
    AllActualHelpersFromContext --> AllActualServicesFromContext
```

**4. COMPLETE CODE EXAMPLE COVERAGE:**
Include ALL actual code snippets from the provided context with proper file path attribution:
```python
# From: <actual_file_path_from_context>
# Complete imports: <all_actual_imports_from_context>
class AllActualClassesFromContext:
    def all_actual_methods(self, all_parameters):
        # All actual implementation details from context
        # All actual error handling
        # All actual business logic
        pass
    
    def all_additional_methods(self, additional_parameters):
        # All additional implementation details
        pass
```

**5. COMPREHENSIVE DOCUMENTATION STRUCTURE:**
Use this complete template structure and adapt based on actual content:

## {page_name}

### Complete Overview
[Comprehensive technical overview covering ALL implementation details found in context]

### Complete Architecture
[Include ALL architecture diagrams covering every component and relationship]
```mermaid
[Complete architecture diagram covering all components]
```

### All Implementation Details
[Exhaustive dive into ALL implementation aspects with complete file references]

### All Key Components
[Complete list with ALL specific file paths and detailed explanations]

### All Usage Examples
[ALL code examples with complete file references and usage patterns]

### All Integration Points
[Complete coverage of ALL connections with comprehensive sequence diagrams]
```mermaid
[Complete sequence diagram covering all integration flows]
```

### All Configuration Options
[Complete setup details, ALL config files, ALL environment variables]

### All Related Components
[Complete cross-references to ALL related functionality and dependencies]

### Complete Troubleshooting Guide
[ALL common issues, ALL edge cases, ALL solutions and workarounds]

### All API References
[Complete documentation of ALL methods, parameters, return values]

### All Performance Considerations
[ALL optimization strategies, ALL bottlenecks, ALL scaling considerations]

**ABSOLUTE QUALITY REQUIREMENTS:**
- Comprehensive technical content covering ALL aspects found in the context
- Complete diagram coverage with ALL relevant Mermaid diagrams for every concept
- ALL specific file path references throughout the documentation
- ALL concrete code examples with complete attribution
- Complete navigation guidance for developers to ALL relevant code locations
- Enterprise-grade documentation depth with NO omissions

**CRITICAL SUCCESS CRITERIA:**
- NO component, method, or feature mentioned in the context should be omitted
- ALL relationships between components must be documented
- ALL code examples must be complete and functional
- ALL file paths must be accurate and complete
- ALL diagrams must reflect actual code structure and relationships

Generate comprehensive, complete, diagram-rich, location-aware documentation content that covers EVERY aspect of the assigned components without any omissions:
"""

# Enhanced Repository Analysis Prompt
ENHANCED_REPO_ANALYSIS_PROMPT = """
You are an expert software architect and comprehensive repository analysis specialist. Your mission is to perform an exhaustive, detailed analysis of a repository without any omissions, documenting every significant component, pattern, and relationship.

**CRITICAL COMPLETENESS DIRECTIVE:** 
NO OMISSIONS ALLOWED. Base your analysis EXCLUSIVELY on the provided repository context. Document EVERY component, service, feature, and integration point found in the provided content. This analysis must be comprehensive and serve as the foundation for complete documentation coverage.

ANALYSIS INPUTS:
- Repository Structure: {repository_tree}
- README Content: {readme_content}
- Sample Code Content: {code_samples}
- File Statistics: {file_stats}

**COMPREHENSIVE ARCHITECTURAL ANALYSIS:**

**1. EXHAUSTIVE ARCHITECTURAL EXAMINATION:**
- Document ALL architectural patterns visible in the folder structure (no exceptions)
- Analyze EVERY layer of the codebase organization and component relationships
- Identify ALL design patterns, frameworks, and architectural decisions present
- Map ALL cross-cutting concerns (logging, config, error handling, security, monitoring)
- Document ALL extension points, plugin systems, and customization mechanisms
- Analyze ALL data flow patterns and communication mechanisms
- Identify ALL performance considerations and optimization strategies

**2. COMPLETE TECHNOLOGY STACK INVENTORY:**
- Document ALL frameworks, libraries, and dependencies found in the codebase
- Identify ALL external integrations, APIs, and service connections
- Map ALL development tools, testing frameworks, and CI/CD processes
- Document ALL runtime environments, deployment configurations, and infrastructure
- Identify ALL data storage, caching, and persistence mechanisms
- Analyze ALL monitoring, logging, and observability implementations
- Document ALL security measures, authentication, and authorization systems

**3. EXHAUSTIVE FUNCTIONAL ANALYSIS:**
- Document the complete domain model and business logic implementation
- Identify ALL capabilities, features, and user workflows
- Map ALL user interaction patterns and interface implementations
- Document ALL data processing pipelines and transformation logic
- Identify ALL integration patterns and external system interactions
- Analyze ALL error handling, recovery, and resilience mechanisms
- Document ALL configuration management and customization options

**4. COMPREHENSIVE COMPONENT-TO-LOCATION MAPPING:**
Create detailed mappings for EVERY component found in the repository:

**Complete Architecture Mapping:**
- Application/library structure: Map ALL entry points, plugin_implementation modules, and architectural layers
- Authentication/authorization: Document ALL security-related files and implementations
- API layers: Map ALL endpoints, controllers, middleware, and routing logic
- Configuration management: Document ALL config files, environment handling, and settings
- Error handling: Map ALL exception handling, error management, and recovery logic
- Data access: Document ALL database access, ORM, and data layer implementations
- Business logic: Map ALL service classes, domain models, and business rules
- Infrastructure: Document ALL deployment, containerization, and infrastructure code

**Complete Feature Implementation Mapping:**
- External integrations: Map ALL service connectors, API clients, and third-party integrations
- Processing systems: Document ALL data processing, transformation, and computation logic
- Workflow management: Map ALL workflow definitions, state machines, and process orchestration
- User interfaces: Document ALL UI components, forms, and user interaction logic
- Reporting/analytics: Map ALL reporting, metrics, and analytics implementations
- Notifications: Document ALL notification systems, messaging, and communication logic

**Complete Infrastructure and Operations Mapping:**
- Deployment: Map ALL containerization, orchestration, and deployment configurations
- Testing: Document ALL test suites, fixtures, and testing infrastructure
- Development tools: Map ALL build scripts, development utilities, and tooling
- Documentation: Document ALL existing documentation, guides, and reference materials
- CI/CD: Map ALL pipeline definitions, automation, and deployment processes
- Monitoring: Document ALL logging, metrics, alerting, and observability implementations

**5. COMPREHENSIVE COMPLEXITY AND PRIORITY ANALYSIS:**
- Identify ALL components requiring detailed documentation (no arbitrary limits)
- Analyze ALL frequently modified areas and their documentation needs
- Document ALL critical integration points and their complexity levels
- Map ALL common user pain points and documentation requirements
- Identify ALL performance-critical components and their optimization needs
- Document ALL security-sensitive areas and their compliance requirements

**6. COMPLETE DOCUMENTATION STRATEGY FRAMEWORK:**
Based on the exhaustive analysis, provide:
- Comprehensive mapping of ALL components deserving dedicated documentation sections
- Complete inventory of ALL architectural diagrams needed for full understanding
- Detailed list of ALL integration patterns requiring step-by-step documentation
- Exhaustive collection of ALL code examples needed for comprehensive illustration
- Complete catalog of ALL troubleshooting scenarios and their solutions
- Full mapping of ALL API endpoints, interfaces, and integration points

**MANDATORY DELIVERABLES:**
Provide a comprehensive analysis that includes:
- Specific file/folder locations for EVERY major component and feature
- Complete dependency mapping and relationship documentation
- Exhaustive feature inventory with implementation details
- Full architectural pattern analysis with concrete examples
- Complete technology stack documentation with versions and purposes
- Comprehensive integration point mapping with protocols and data formats

**QUALITY ASSURANCE REQUIREMENTS:**
- Base ALL conclusions on actual repository structure and provided content
- Ensure NO component, service, or feature is omitted from analysis
- Provide complete traceability from every conclusion to specific code locations
- Document ALL relationships between components with concrete evidence
- Ensure comprehensive coverage that enables complete documentation generation
"""

# Enhanced Wiki Structure Analysis Prompt
ENHANCED_WIKI_STRUCTURE_PROMPT = """
You are an expert technical documentation architect analyzing a repository to create a comprehensive wiki structure that rivals industry-leading documentation.

Repository Information:
- Repository Tree: {repository_tree}
- README Content: {readme_content}  
- Analysis: {repo_analysis}
- Target Audience: {target_audience}
- Wiki Type: {wiki_type}

**CREATE REPOSITORY-SPECIFIC DOCUMENTATION STRUCTURE**

Based on the repository analysis, analyze the unique characteristics of this codebase and create a comprehensive wiki structure that naturally reflects its architecture, purpose, and complexity.

**ANALYSIS-DRIVEN APPROACH:**
- Study the repository analysis to understand the unique architecture and components
- Identify the most important concepts, workflows, and integration points
- Create sections that naturally emerge from the repository's actual structure
- Organize documentation around the real user needs and codebase complexity
- Design a structure that matches this specific repository's logical organization
- Let the repository's complexity and content naturally determine the appropriate scope

**COVERAGE REQUIREMENTS:**
- Comprehensive coverage of all major components and systems
- In-depth guides for complex areas requiring detailed explanation
- User-focused organization based on actual usage patterns
- Troubleshooting and practical guidance for real-world scenarios
- Clear navigation that matches the repository's conceptual model

**COMPREHENSIVE COVERAGE APPROACH:**
- **NO OMISSIONS ALLOWED**: Document every significant component, service, and feature found in the repository
- **Exhaustive Analysis**: Cover ALL aspects of the project without exception
- **Complete Component Coverage**: Every module, class, service, and integration point must be documented
- **Scalable Structure**: Let repository complexity naturally determine the appropriate documentation scope
- **Quality Over Arbitrary Limits**: Focus on comprehensive coverage rather than hitting specific page counts
- **Hierarchical Completeness**: Ensure every level of the system architecture is properly documented

**QUALITY STANDARDS:**
- Appropriate documentation depth based on actual repository complexity and component count
- Hierarchical organization with clear navigation paths to every component
- Each page should provide substantial coverage of its assigned components
- Include specific diagram recommendations based on actual content type and relationships
- Map ALL content to specific repository locations from the analysis
- Ensure complete traceability from documentation to actual code

**SECTION INSPIRATION (NOT MANDATORY):**
Different repositories may naturally organize around different concepts. Here are some examples of how various types of repositories might be structured - use these as inspiration, not requirements:

*For Framework/Library Projects:*
- Architecture and Core Concepts
- API Reference and Usage
- Integration Patterns
- Extension and Customization

*For Application Projects:*
- System Architecture
- Feature Implementation
- Configuration and Deployment
- User Workflows and Processes

*For Tool/SDK Projects:*
- Getting Started and Setup
- Core Functionality
- Integration Examples
- Advanced Usage Patterns

*For Data/ML Projects:*
- Data Pipeline Architecture
- Model Implementation
- Processing Workflows
- Performance and Optimization

**IMPORTANT: These are examples only. Create sections that naturally fit THIS repository's unique characteristics, not generic templates.**

**FLEXIBLE STRUCTURE CREATION:**
Analyze the repository and create sections that:
- Reflect the actual codebase organization and purpose
- Address the real complexity and user needs
- Follow natural information hierarchies
- Support the identified workflows and use cases
- Provide comprehensive coverage without forced categorization

Generate a comprehensive JSON structure with sections that emerge organically from the repository analysis, ensuring each page has:
- Clear purpose based on actual repository characteristics
- Specific repository file/folder mappings from the analysis
- Diagram recommendations appropriate for the content type
- Estimated complexity reflecting actual implementation complexity
- Cross-references that match real component relationships

Create documentation that matches the depth and quality of the best enterprise software documentation while being uniquely suited to this specific repository.

**REQUIRED JSON FORMAT:**

Return a comprehensive JSON structure with this exact format matching the Pydantic models:

{{
    "wiki_title": "Repository-specific title based on actual analysis",
    "overview": "Comprehensive overview that references specific repository folders and components from the analysis",
    "sections": [
        {{
            "section_name": "Section name that naturally emerges from repository analysis",
            "section_order": 1,
            "description": "Description based on actual repository characteristics",
            "rationale": "Why this section is essential based on the specific repository structure and complexity",
            "pages": [
                {{
                    "page_name": "Page name that reflects actual repository concepts",
                    "page_order": 1,
                    "description": "Description based on actual repository needs",
                    "content_focus": "Focus areas derived from actual repository analysis",
                    "rationale": "Why this page is needed based on specific repository complexity and structure",
                    "user_questions": ["Actual questions users would have about this repository"],
                    "search_keywords": ["Keywords relevant to this specific repository"],
                    "estimated_complexity": "complex|moderate|simple",
                    "target_folders": ["Actual folders from repository analysis"],
                    "key_files": ["Actual files from repository analysis"],
                    "code_references": "References to actual repository code and structures",
                    "diagram_recommendations": ["Diagram types appropriate for this content"]
                }}
            ]
        }}
    ],
    "total_pages": "Actual count based on repository complexity",
    "estimated_effort": "high|medium|low"
}}

**COMPREHENSIVE STRUCTURE REQUIREMENTS:**
- Create complete documentation structure that covers ALL repository components without omission
- Each page should provide comprehensive coverage of its assigned components
- Include ALL necessary Mermaid diagrams covering every concept and relationship
- Include exhaustive lists of target_folders and key_files from the repository analysis
- Add comprehensive diagram_recommendations covering all relevant diagram types
- Ensure ALL major components identified in the repository analysis have dedicated pages
- Create hierarchical structure that matches the repository's complete logical organization
- Use exact Pydantic model field names: page_order, section_order, rationale, user_questions, search_keywords, estimated_complexity, target_folders, key_files, code_references
- **MOST IMPORTANT: Let the structure emerge naturally from the repository analysis - create as many sections and pages as needed to cover everything comprehensively**
"""

# Enhanced Content Enhancement Prompt for Retry
ENHANCED_RETRY_CONTENT_PROMPT = """
You are a technical editor reviewing and enhancing documentation content based on quality feedback.

Original Content:
{original_content}

Context:
- Repository: {repo_name}
- Section: {section_title}
- Target Audience: {audience}
- Enhancement Focus: {enhancement_focus}

Quality Assessment Feedback:
{validation_feedback}

Key Issues to Address:
{quality_issues}

Improvement Requirements:
{improvement_requirements}

**ENHANCEMENT REQUIREMENTS:**

1. **Address All Quality Issues**: Fix every issue mentioned in the quality feedback
2. **Add Missing Diagrams**: If diagrams are missing, add relevant Mermaid diagrams
3. **Improve Technical Depth**: Add more technical details and implementation specifics
4. **Enhance Code Examples**: Include better code examples with file paths
5. **Strengthen Location Guidance**: Add more specific file/folder references

**MANDATORY DIAGRAM INTEGRATION:**
If the content lacks diagrams, add relevant ones:

- **Architecture diagrams** for system overviews
- **Sequence diagrams** for process flows
- **Class diagrams** for component relationships
- **Flowcharts** for decision processes

**CONTENT QUALITY STANDARDS:**
- Minimum 1000 words of technical content
- At least 3-5 Mermaid diagrams covering main concepts and architecture aspects
- Specific file path references throughout
- Concrete code examples with attribution
- Clear troubleshooting guidance

Enhanced Content:
"""

# Additional constants needed for compatibility

# Target Audiences
TARGET_AUDIENCES = {
    "developers": "Software developers and engineers",
    "devops": "DevOps engineers and system administrators", 
    "architects": "Solution architects and technical leads",
    "mixed": "Mixed audience of developers, architects, and operators"
}

# Quality Standards
QUALITY_STANDARDS = {
    "technical_accuracy": 0.9,
    "clarity": 0.8,
    "completeness": 0.8,
    "diagram_relevance": 0.8
}

# Export Summary Prompt
EXPORT_SUMMARY_PROMPT = """
Provide a comprehensive summary of the wiki generation results:

Wiki Generation Summary:
- Title: {wiki_title}
- Total Pages Generated: {total_pages}
- Total Diagrams Created: {total_diagrams}
- Average Quality Score: {average_quality}
- Generation Time: {execution_time}

Key Achievements:
- Comprehensive documentation structure created
- Enterprise-grade content with technical depth
- Diagram-rich visual explanations
- Location-aware navigation guidance

Provide detailed summary of what was accomplished.
"""

# Quality Assessment Prompt
QUALITY_ASSESSMENT_PROMPT = """
Assess the quality of this documentation content against enterprise standards:

Content to Evaluate:
{content}

Assessment Criteria:
- Target Audience: {target_audience}
- Quality Standards: {quality_standards}
- Page Requirements: Technical depth, diagrams, code examples, file references

Evaluation Framework:
1. **Technical Accuracy** (0-1): Is the technical information correct and current?
2. **Clarity and Readability** (0-1): Is the content clear and well-structured?
3. **Completeness** (0-1): Does it cover all necessary aspects comprehensively?
4. **Diagram Integration** (0-1): Are relevant diagrams included and helpful?
5. **Code Examples** (0-1): Are concrete, accurate code examples provided?
6. **Location Guidance** (0-1): Does it guide users to specific files/folders?

Provide scores, detailed feedback, strengths, weaknesses, and improvement suggestions.
"""

# Content Validation Prompt
CONTENT_VALIDATION_PROMPT = """
Validate this documentation content for enterprise publication standards:

Content: {content}
Publication Requirements: {requirements}

Validation Checklist:
1. **Technical Correctness**: Verify all technical details are accurate
2. **Formatting Standards**: Check markdown, code blocks, diagrams are properly formatted
3. **Completeness**: Ensure all required sections are covered
4. **Quality Compliance**: Meets enterprise documentation standards
5. **Accessibility**: Content is accessible to target audience
6. **Navigation**: Proper cross-references and file path guidance

Provide validation results with pass/fail status and specific issues to address.
"""

# Diagram Enhancement Prompt  
DIAGRAM_ENHANCEMENT_PROMPT = """
Create or enhance Mermaid diagrams for this documentation content:

Content Context: {content}
Diagram Requirements:
- Type: {diagram_type}
- Purpose: {diagram_purpose}
- Integration Point: Where this fits in the documentation

Create appropriate Mermaid diagrams that:
1. Enhance technical understanding
2. Illustrate complex relationships
3. Provide visual clarity
4. Follow Mermaid best practices
5. Are properly formatted for documentation

Return only the Mermaid diagram code with proper markdown formatting.
"""
