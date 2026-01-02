"""Output organization and file management."""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from agent import PaperSummary
from arxiv_fetcher import PaperMetadata


class OutputManager:
    """Manages output file organization."""
    
    def __init__(self, output_dir: Path, create_subdirs: bool = True):
        """
        Initialize OutputManager.
        
        Args:
            output_dir: Base directory for output
            create_subdirs: Whether to create category subdirectories
        """
        self.output_dir = Path(output_dir)
        self.create_subdirs = create_subdirs
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, summary: PaperSummary) -> Path:
        """
        Determine output path for a summary.
        
        Args:
            summary: PaperSummary object
            
        Returns:
            Path where summary should be saved
        """
        if self.create_subdirs and summary.category:
            # Sanitize category name for filesystem
            safe_category = "".join(c for c in summary.category if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_category = safe_category.replace(' ', '_').lower()
            category_dir = self.output_dir / safe_category
            category_dir.mkdir(parents=True, exist_ok=True)
            return category_dir / f"{summary.arxiv_id}.md"
        else:
            return self.output_dir / f"{summary.arxiv_id}.md"
    
    def save_summary(self, summary: PaperSummary, paper_metadata: PaperMetadata) -> Path:
        """
        Save paper summary to file.
        
        Args:
            summary: PaperSummary object
            paper_metadata: Original PaperMetadata for additional context
            
        Returns:
            Path to saved file
        """
        output_path = self.get_output_path(summary)
        
        # Create markdown content with full metadata
        content = summary.to_markdown()
        
        # Add metadata section at the end
        metadata_section = f"""
---

## Metadata

**Authors:** {', '.join(paper_metadata.authors)}

**Published:** {paper_metadata.published.strftime('%Y-%m-%d')}

**Categories:** {', '.join(paper_metadata.categories)}

**PDF URL:** {paper_metadata.pdf_url}

**ArXiv Entry:** {paper_metadata.entry_id}

**Summary Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        content += metadata_section
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Also save JSON metadata
        json_path = output_path.with_suffix('.json')
        metadata_dict = {
            'paper': paper_metadata.to_dict(),
            'summary': {
                'arxiv_id': summary.arxiv_id,
                'title': summary.title,
                'key_findings': summary.key_findings,
                'methodology': summary.methodology,
                'results': summary.results,
                'relevance': summary.relevance,
                'applications': summary.applications,
                'limitations': summary.limitations,
                'category': summary.category
            },
            'generated_at': datetime.now().isoformat()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def save_index(self, summaries: list[PaperSummary], paper_metadatas: list[PaperMetadata]):
        """
        Create an index file listing all papers.
        
        Args:
            summaries: List of PaperSummary objects
            paper_metadatas: List of corresponding PaperMetadata objects
        """
        index_path = self.output_dir / "index.md"
        
        content = "# Paper Index\n\n"
        content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        content += f"Total Papers: {len(summaries)}\n\n"
        content += "---\n\n"
        
        # Group by category if available
        if any(s.category for s in summaries):
            from collections import defaultdict
            by_category = defaultdict(list)
            for summary, metadata in zip(summaries, paper_metadatas):
                category = summary.category or "uncategorized"
                by_category[category].append((summary, metadata))
            
            for category in sorted(by_category.keys()):
                content += f"## {category.title()}\n\n"
                for summary, metadata in sorted(by_category[category], key=lambda x: x[1].published, reverse=True):
                    output_path = self.get_output_path(summary)
                    relative_path = output_path.relative_to(self.output_dir)
                    content += f"- [{summary.title}]({relative_path}) - {metadata.published.strftime('%Y-%m-%d')} - {summary.arxiv_id}\n"
                content += "\n"
        else:
            # List all papers chronologically
            for summary, metadata in sorted(zip(summaries, paper_metadatas), key=lambda x: x[1].published, reverse=True):
                output_path = self.get_output_path(summary)
                relative_path = output_path.relative_to(self.output_dir)
                content += f"- [{summary.title}]({relative_path}) - {metadata.published.strftime('%Y-%m-%d')} - {summary.arxiv_id}\n"
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)

