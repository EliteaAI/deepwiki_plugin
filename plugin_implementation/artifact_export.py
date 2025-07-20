"""
Artifact Export Utilities

This module provides utilities for exporting and storing wiki artifacts
via the Alita client system.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import zipfile

from plugin_implementation.state.wiki_state import WikiStructureSpec, WikiPage

logger = logging.getLogger(__name__)


class ArtifactExporter:
    """Handles export and storage of wiki artifacts."""
    
    def __init__(self, client, bucket_name: str = "wiki_artifacts"):
        """
        Initialize artifact exporter.
        
        Args:
            client: Alita client for storing artifacts
            bucket_name: Name of the bucket to store artifacts in
        """
        self.client = client
        self.bucket_name = bucket_name
        self.export_formats = ["json", "markdown", "html", "pdf"]
    
    def export_wiki_artifacts(self, 
                             wiki_structure: WikiStructureSpec,
                             wiki_pages: List[WikiPage],
                             repo_metadata: Dict[str, Any],
                             export_formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Export wiki artifacts in multiple formats.
        
        Args:
            wiki_structure: The wiki structure
            wiki_pages: List of wiki pages
            repo_metadata: Repository metadata
            export_formats: List of formats to export (default: ["json", "markdown"])
            
        Returns:
            Dictionary mapping format names to main artifact URLs. 
            For markdown format, individual files are also stored as separate artifacts for UI rendering.
        """
        if export_formats is None:
            export_formats = ["json", "markdown"]
        
        artifacts = {}
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Export in each requested format
                for format_name in export_formats:
                    try:
                        if format_name == "json":
                            artifact_url = self._export_json_format(
                                wiki_structure, wiki_pages, repo_metadata, temp_path
                            )
                        elif format_name == "markdown":
                            artifact_url = self._export_markdown_format(
                                wiki_structure, wiki_pages, repo_metadata, temp_path
                            )
                        elif format_name == "html":
                            artifact_url = self._export_html_format(
                                wiki_structure, wiki_pages, repo_metadata, temp_path
                            )
                        elif format_name == "pdf":
                            artifact_url = self._export_pdf_format(
                                wiki_structure, wiki_pages, repo_metadata, temp_path
                            )
                        else:
                            logger.warning(f"Unsupported export format: {format_name}")
                            continue
                        
                        artifacts[format_name] = artifact_url
                        logger.info(f"Exported wiki in {format_name} format: {artifact_url}")
                        
                    except Exception as e:
                        logger.error(f"Failed to export wiki in {format_name} format: {str(e)}")
                
                # Create comprehensive export summary
                summary_url = self._create_export_summary(
                    wiki_structure, wiki_pages, repo_metadata, artifacts, temp_path
                )
                artifacts["summary"] = summary_url
                
                return artifacts
                
        except Exception as e:
            logger.error(f"Error exporting wiki artifacts: {str(e)}")
            return {}
    
    def _export_json_format(self, 
                           wiki_structure: WikiStructureSpec,
                           wiki_pages: List[WikiPage],
                           repo_metadata: Dict[str, Any],
                           temp_path: Path) -> str:
        """Export wiki in simple JSON format optimized for Git repository structure."""
        
        # Create a mapping of page IDs to pages for easier lookup
        pages_by_id = {page.page_id: page for page in wiki_pages}
        
        # Use the WikiStructureSpec sections if available, otherwise fallback grouping
        if wiki_structure.sections and len(wiki_structure.sections) > 0:
            # Use the actual sections from wiki_structure
            sections_data = []
            for section_idx, section in enumerate(wiki_structure.sections):
                section_pages = []
                for page_idx, page_spec in enumerate(section.pages):
                    # Create expected page_id based on structure indices
                    expected_page_id = f"{section_idx}#{page_idx}"
                    
                    # Find matching page by the actual page_id format
                    matching_page = pages_by_id.get(expected_page_id)
                    
                    # Fallback: try to find by page_name if direct ID match fails
                    if not matching_page:
                        matching_page = next(
                            (p for p in wiki_pages if p.title == page_spec.page_name), 
                            None
                        )
                    
                    if matching_page:
                        section_pages.append({
                            'page_name': f"{self._create_safe_filename(matching_page.title)}.md",
                            'page_content': matching_page.content
                        })
                
                if section_pages:  # Only add sections that have pages
                    sections_data.append({
                        'section_name': section.section_name,
                        'pages': section_pages
                    })
        else:
            # Fallback: Group pages by section index from page_id format (e.g., "0#1" -> section 0)
            pages_by_section = {}
            for page in wiki_pages:
                section_name = "General"
                
                # Extract section from page_id format like "0#1" -> section 0
                if '#' in page.page_id:
                    try:
                        section_idx = int(page.page_id.split('#')[0])
                        # Try to get section name from wiki_structure if available
                        if (wiki_structure.sections and 
                            section_idx < len(wiki_structure.sections)):
                            section_name = wiki_structure.sections[section_idx].section_name
                        else:
                            section_name = f"Section {section_idx}"
                    except (ValueError, IndexError):
                        section_name = "General"
                
                if section_name not in pages_by_section:
                    pages_by_section[section_name] = []
                
                pages_by_section[section_name].append({
                    'page_name': f"{self._create_safe_filename(page.title)}.md",
                    'page_content': page.content
                })
            
            sections_data = [
                {
                    'section_name': section_name,
                    'pages': pages
                }
                for section_name, pages in pages_by_section.items()
            ]
        
        # Create simple, clean JSON structure
        wiki_data = {
            'wiki_title': wiki_structure.wiki_title,
            'sections': sections_data
        }
        
        # Write to temporary file
        json_path = temp_path / "wiki_structure.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(wiki_data, f, indent=2, ensure_ascii=False)
        
        # Read file data and store as artifact
        with open(json_path, 'rb') as f:
            artifact_data = f.read()
        
        artifact_name = f"wiki_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.client.create_artifact(
            bucket_name=self.bucket_name,
            artifact_name=artifact_name,
            artifact_data=artifact_data
        )
        
        return artifact_name
    
    def _export_markdown_format(self, 
                               wiki_structure: WikiStructureSpec,
                               wiki_pages: List[WikiPage],
                               repo_metadata: Dict[str, Any],
                               temp_path: Path) -> str:
        """Export wiki in markdown format with proper directory structure."""
        
        markdown_dir = temp_path / "wiki_markdown"
        markdown_dir.mkdir(exist_ok=True)
        
        # Create directory structure based on sections
        if wiki_structure.sections and len(wiki_structure.sections) > 0:
            # Create a mapping of page IDs to pages for easier lookup
            pages_by_id = {page.page_id: page for page in wiki_pages}
            
            # Use the actual sections from wiki_structure
            for section_idx, section in enumerate(wiki_structure.sections):
                section_dir = markdown_dir / self._create_safe_filename(section.section_name)
                section_dir.mkdir(exist_ok=True)
                
                for page_idx, page_spec in enumerate(section.pages):
                    # Create expected page_id based on structure indices
                    expected_page_id = f"{section_idx}#{page_idx}"
                    
                    # Find matching page by the actual page_id format
                    matching_page = pages_by_id.get(expected_page_id)
                    
                    # Fallback: try to find by page_name if direct ID match fails
                    if not matching_page:
                        matching_page = next(
                            (p for p in wiki_pages if p.title == page_spec.page_name), 
                            None
                        )
                    
                    if matching_page:
                        self._create_markdown_page_in_section(matching_page, section_dir)
        else:
            # Fallback: Group pages by section index from page_id format (e.g., "0#1" -> section 0)
            pages_by_section = {}
            for page in wiki_pages:
                section_name = "general"
                
                # Extract section from page_id format like "0#1" -> section 0
                if '#' in page.page_id:
                    try:
                        section_idx = int(page.page_id.split('#')[0])
                        # Try to get section name from wiki_structure if available
                        if (wiki_structure.sections and 
                            section_idx < len(wiki_structure.sections)):
                            section_name = wiki_structure.sections[section_idx].section_name
                        else:
                            section_name = f"section_{section_idx}"
                    except (ValueError, IndexError):
                        section_name = "general"
                
                if section_name not in pages_by_section:
                    pages_by_section[section_name] = []
                pages_by_section[section_name].append(page)
            
            # Create directories and files
            for section_name, pages in pages_by_section.items():
                section_dir = markdown_dir / self._create_safe_filename(section_name)
                section_dir.mkdir(exist_ok=True)
                
                for page in pages:
                    self._create_markdown_page_in_section(page, section_dir)
        
        # Create main README
        self._create_markdown_index(wiki_structure, wiki_pages, markdown_dir)
        
        # Store individual markdown files as artifacts for UI debugging/rendering
        individual_artifacts = []
        for file_path in markdown_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix == '.md':
                # Read markdown file
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                # Create artifact name preserving directory structure
                relative_path = file_path.relative_to(markdown_dir)
                safe_name = str(relative_path).replace('/', '_').replace('\\', '_')
                # Remove .md extension from safe_name and add timestamp before re-adding .md
                base_name = safe_name.replace('.md', '')
                artifact_name = f"markdown_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                
                # Store individual file as artifact
                self.client.create_artifact(
                    bucket_name=self.bucket_name,
                    artifact_name=artifact_name,
                    artifact_data=file_data
                )
                individual_artifacts.append(artifact_name)
                logger.info(f"Stored individual markdown file: {artifact_name}")
        
        # Create archive with all files
        archive_path = temp_path / "wiki_markdown.zip"
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in markdown_dir.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(markdown_dir))
        
        # Read archive data and store as artifact
        with open(archive_path, 'rb') as f:
            artifact_data = f.read()
        
        zip_artifact_name = f"wiki_markdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        self.client.create_artifact(
            bucket_name=self.bucket_name,
            artifact_name=zip_artifact_name,
            artifact_data=artifact_data
        )
        
        logger.info(f"Stored markdown archive: {zip_artifact_name}")
        logger.info(f"Stored {len(individual_artifacts)} individual markdown files for UI rendering")
        
        return zip_artifact_name
    
    def _create_export_summary(self, 
                              wiki_structure: WikiStructureSpec,
                              wiki_pages: List[WikiPage],
                              repo_metadata: Dict[str, Any],
                              artifacts: Dict[str, str],
                              temp_path: Path) -> str:
        """Create export summary artifact."""
        
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif hasattr(obj, 'model_dump'):
                    return obj.model_dump()
                elif hasattr(obj, '__dict__'):
                    return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
                return super(DateTimeEncoder, self).default(obj)
        
        summary_data = {
            "export_summary": {
                "generated_at": datetime.now().isoformat(),
                "repository": repo_metadata.get("name", "unknown"),
                "wiki_title": wiki_structure.wiki_title,
                "wiki_description": getattr(wiki_structure, 'overview', 'Generated wiki documentation'),
                "total_pages": len(wiki_pages),
                "exported_formats": list(artifacts.keys()),
                "artifacts": artifacts
            },
            "content_statistics": {
                "total_content_length": sum(len(page.content) for page in wiki_pages),
                "total_diagrams": 0,  # Diagrams not tracked in current WikiPage structure
                "total_citations": 0,  # Citations not tracked in current WikiPage structure
                "pages_by_category": self._get_pages_by_category(wiki_pages)
            },
            "pages_overview": [
                {
                    "title": page.title,
                    "content_length": len(page.content),
                    "diagrams": 0,  # Diagrams not tracked in current WikiPage structure
                    "citations": 0,  # Citations not tracked in current WikiPage structure
                    "tags": getattr(page, 'tags', [])  # Tags not in current WikiPage structure
                }
                for page in wiki_pages
            ]
        }
        
        # Write summary
        summary_path = temp_path / "export_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
        
        # Read summary data and store as artifact
        with open(summary_path, 'rb') as f:
            artifact_data = f.read()
        
        artifact_name = f"export_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.client.create_artifact(
            bucket_name=self.bucket_name,
            artifact_name=artifact_name,
            artifact_data=artifact_data
        )
        
        return artifact_name
    
    def _create_markdown_index(self, 
                              wiki_structure: WikiStructureSpec,
                              wiki_pages: List[WikiPage],
                              output_dir: Path):
        """Create markdown index page showing directory structure."""
        
        content = f"# {wiki_structure.wiki_title}\n\n"
        content += f"{getattr(wiki_structure, 'overview', 'Generated wiki documentation')}\n\n"
        
        content += "## Wiki Structure\n\n"
        
        # Show directory structure based on sections
        if wiki_structure.sections and len(wiki_structure.sections) > 0:
            # Create a mapping of page IDs to pages for easier lookup
            pages_by_id = {page.page_id: page for page in wiki_pages}
            
            for section_idx, section in enumerate(wiki_structure.sections):
                section_name = self._create_safe_filename(section.section_name)
                content += f"### {section.section_name}/\n\n"
                
                for page_idx, page_spec in enumerate(section.pages):
                    # Create expected page_id based on structure indices
                    expected_page_id = f"{section_idx}#{page_idx}"
                    
                    # Find matching page by the actual page_id format
                    matching_page = pages_by_id.get(expected_page_id)
                    
                    # Fallback: try to find by page_name if direct ID match fails
                    if not matching_page:
                        matching_page = next(
                            (p for p in wiki_pages if p.title == page_spec.page_name), 
                            None
                        )
                    
                    if matching_page:
                        filename = self._create_safe_filename(matching_page.title)
                        content += f"- [{matching_page.title}]({section_name}/{filename}.md)\n"
                content += "\n"
        else:
            # Fallback: Group pages by section index from page_id format (e.g., "0#1" -> section 0)
            pages_by_section = {}
            for page in wiki_pages:
                section_name = "general"
                
                # Extract section from page_id format like "0#1" -> section 0
                if '#' in page.page_id:
                    try:
                        section_idx = int(page.page_id.split('#')[0])
                        # Try to get section name from wiki_structure if available
                        if (wiki_structure.sections and 
                            section_idx < len(wiki_structure.sections)):
                            section_name = wiki_structure.sections[section_idx].section_name
                        else:
                            section_name = f"section_{section_idx}"
                    except (ValueError, IndexError):
                        section_name = "general"
                
                if section_name not in pages_by_section:
                    pages_by_section[section_name] = []
                pages_by_section[section_name].append(page)
            
            for section_name, pages in pages_by_section.items():
                safe_section = self._create_safe_filename(section_name)
                content += f"### {section_name.title()}/\n\n"
                for page in pages:
                    filename = self._create_safe_filename(page.title)
                    content += f"- [{page.title}]({safe_section}/{filename}.md)\n"
                content += "\n"
        
        content += f"\n---\n\n*Generated by Alita Wiki Toolkit*\n"
        
        with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(content)

    def _create_markdown_page_in_section(self, page: WikiPage, section_dir: Path):
        """Create individual markdown page in a specific section directory."""
        
        filename = self._create_safe_filename(page.title)
        content = page.content
        
        with open(section_dir / f"{filename}.md", 'w', encoding='utf-8') as f:
            f.write(content)

    def _get_pages_by_category(self, wiki_pages) -> Dict[str, List]:
        """Group pages by category."""
        
        pages_by_category = {}
        
        for page in wiki_pages:
            category = "General"
            # Tags not supported in current WikiPage structure
            # if page.tags:
            #     category = page.tags[0].title()
            
            if category not in pages_by_category:
                pages_by_category[category] = []
            pages_by_category[category].append(page)
        
        return pages_by_category
    
    def _create_safe_filename(self, title: str) -> str:
        """Create safe filename from title."""
        import re
        
        safe_name = re.sub(r'[^\w\s-]', '', title)
        safe_name = re.sub(r'[-\s]+', '-', safe_name)
        return safe_name.strip('-').lower()
