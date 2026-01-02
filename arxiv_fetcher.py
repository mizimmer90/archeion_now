"""Arxiv paper fetching functionality."""
import arxiv
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import requests


@dataclass
class PaperMetadata:
    """Metadata for an arxiv paper."""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime
    pdf_url: str
    entry_id: str
    doi: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'arxiv_id': self.arxiv_id,
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'categories': self.categories,
            'published': self.published.isoformat(),
            'pdf_url': self.pdf_url,
            'entry_id': self.entry_id,
            'doi': self.doi
        }

    @staticmethod
    def _fallback_arxiv_doi(arxiv_id: str) -> str:
        """
        ArXiv mints DOIs for all e-prints via DataCite using the
        pattern 10.48550/arXiv.<id_without_version>. Use this as a
        deterministic fallback when the API omits the DOI field.
        """
        base_id = arxiv_id.split('v')[0]
        return f"10.48550/arXiv.{base_id}"

    @classmethod
    def from_arxiv_entry(cls, entry: arxiv.Result) -> "PaperMetadata":
        """Create PaperMetadata from arxiv.Result."""
        raw_id = entry.entry_id.split('/')[-1]
        doi = getattr(entry, "doi", None) or cls._fallback_arxiv_doi(raw_id)
        return cls(
            arxiv_id=raw_id,
            title=entry.title,
            authors=[str(author) for author in entry.authors],
            abstract=entry.summary,
            categories=entry.categories,
            published=entry.published,
            pdf_url=entry.pdf_url,
            entry_id=entry.entry_id,
            doi=doi
        )
    
    def __repr__(self) -> str:
        """String representation showing title, authors, and DOI."""
        authors_str = ', '.join(self.authors) if self.authors else 'Unknown authors'
        doi_str = self.doi if self.doi else 'No DOI'
        return f"PaperMetadata(title='{self.title}', authors='{authors_str}', doi='{doi_str}')"


class ArxivFetcher:
    """Fetcher for arxiv papers."""
    
    def __init__(self, categories: List[str], days_back: int = 7, max_papers: Optional[int] = None):
        """
        Initialize the ArxivFetcher.
        
        Args:
            categories: List of arxiv categories (e.g., ['cs.AI', 'cs.LG'])
            days_back: Number of days to look back
            max_papers: Maximum number of papers to fetch (None for no limit)
        """
        self.categories = categories
        self.days_back = days_back
        self.max_papers = max_papers
    
    def fetch_recent_papers(self) -> List[PaperMetadata]:
        """
        Fetch recent papers from arxiv.
        
        Returns:
            List of PaperMetadata objects
        """
        cutoff_date = datetime.now() - timedelta(days=self.days_back)
        now_date = datetime.now()
        
        # Build search query: filter by date and categories
        query_parts = []
        
        # Date filter
        # Arxiv date ranges require 12-digit timestamps without wildcards (YYYYMMDDHHMM)
        date_str = cutoff_date.strftime("%Y%m%d%H%M")
        now_str = now_date.strftime("%Y%m%d%H%M")
        query_parts.append(f"submittedDate:[{date_str} TO {now_str}]")
        
        # Category filter
        if self.categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in self.categories])
            query_parts.append(f"({cat_query})")
        
        query = " AND ".join(query_parts)
        
        print(f"Searching arxiv with query: {query}")
        
        search = arxiv.Search(
            query=query,
            max_results=self.max_papers if self.max_papers else 10000,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        try:
            for result in search.results():
                # Double-check date filter (arxiv query might be approximate)
                if result.published >= cutoff_date.replace(tzinfo=result.published.tzinfo):
                    papers.append(PaperMetadata.from_arxiv_entry(result))
                    if self.max_papers and len(papers) >= self.max_papers:
                        break
        except Exception as e:
            print(f"Error fetching papers: {e}")
        
        print(f"Fetched {len(papers)} papers")
        return papers
    
    def download_pdf(self, paper: PaperMetadata, output_dir: Path) -> Optional[Path]:
        """
        Download PDF for a paper.
        
        Args:
            paper: PaperMetadata object
            output_dir: Directory to save PDF
            
        Returns:
            Path to downloaded PDF or None if download failed
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{paper.arxiv_id}.pdf"
        
        if pdf_path.exists():
            return pdf_path
        
        try:
            response = requests.get(paper.pdf_url, timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            return pdf_path
        except Exception as e:
            print(f"Error downloading PDF for {paper.arxiv_id}: {e}")
            return None
