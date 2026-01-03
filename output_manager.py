"""Output organization and file management."""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
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
        self.cache_file = self.output_dir / "doi_cache.json"
        self._cache: Dict[str, Dict[str, str]] = {}
        self._load_cache()
    
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
        doi_line = f"**DOI:** {paper_metadata.doi}\n\n" if paper_metadata.doi else ""
        metadata_section = f"""
---

## Metadata

**Authors:** {', '.join(paper_metadata.authors)}

**Published:** {paper_metadata.published.strftime('%Y-%m-%d')}

**Categories:** {', '.join(paper_metadata.categories)}

{doi_line}**PDF URL:** {paper_metadata.pdf_url}

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
    
    def _load_cache(self):
        """Load DOI cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load DOI cache: {e}")
                self._cache = {}
        else:
            self._cache = {}
    
    def _save_cache(self):
        """Save DOI cache to file with spacing between entries for readability."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                f.write('{\n')
                entries = list(self._cache.items())
                for i, (doi, data) in enumerate(entries):
                    # Write the DOI key and opening brace
                    f.write(f'  {json.dumps(doi, ensure_ascii=False)}: {{\n')
                    
                    # Write each field in the entry
                    fields = list(data.items())
                    for j, (key, value) in enumerate(fields):
                        comma = ',' if j < len(fields) - 1 else ''
                        value_str = json.dumps(value, ensure_ascii=False)
                        f.write(f'    {json.dumps(key, ensure_ascii=False)}: {value_str}{comma}\n')
                    
                    # Close the entry
                    f.write('  }')
                    if i < len(entries) - 1:
                        f.write(',')
                    f.write('\n')
                    
                    # Add blank line between entries (except after the last one)
                    if i < len(entries) - 1:
                        f.write('\n')
                
                f.write('}\n')
        except IOError as e:
            print(f"Warning: Could not save DOI cache: {e}")
    
    def is_cached(self, doi: Optional[str]) -> Optional[bool]:
        """
        Check if a DOI has been assessed previously.
        
        Args:
            doi: DOI string (can be None)
            
        Returns:
            True if accepted, False if rejected, None if not cached or no DOI
        """
        if not doi:
            return None
        
        if doi in self._cache:
            status = self._cache[doi].get("status", "").lower()
            if status == "accept":
                return True
            elif status == "reject":
                return False
        
        return None
    
    def get_cached_data(self, doi: Optional[str]) -> Optional[Dict[str, str]]:
        """
        Get full cached data for a DOI.
        
        Args:
            doi: DOI string (can be None)
            
        Returns:
            Dictionary with 'status', 'reasoning', 'relevance', 'confidence', 'impact', and 'timestamp' keys, 
            or None if not cached. May also contain legacy 'estimated_impact' key for backward compatibility.
        """
        if not doi or doi not in self._cache:
            return None
        
        return self._cache[doi].copy()
    
    def save_decision(self, doi: Optional[str], relevance: float, reasoning: str = "", 
                     confidence: float = 0.0, impact: float = 0.0, title: str = "", 
                     relevance_threshold: float = 0.5, test_label: Optional[str] = None):
        """
        Save a DOI assessment decision to cache.
        
        Args:
            doi: DOI string (can be None)
            relevance: Relevance score (0.0-1.0) indicating alignment with user interests
            reasoning: Justification for the decision
            confidence: Confidence score for the relevance assessment (0.0-1.0)
            impact: Estimated impact score for the field at large (0.0-1.0)
            title: Paper title
            relevance_threshold: Threshold for determining accept/reject (default 0.5)
            test_label: Optional test label ("positive" or "negative") for testing mode
        """
        if not doi:
            return
        
        # Determine accept/reject status based on relevance threshold
        is_accepted = relevance >= relevance_threshold
        
        cache_entry = {
            "status": "accept" if is_accepted else "reject",
            "title": title,
            "reasoning": reasoning,
            "relevance": relevance,
            "confidence": confidence,
            "impact": impact,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add test label if provided
        if test_label:
            cache_entry["test_label"] = test_label
        
        self._cache[doi] = cache_entry
        self._save_cache()

