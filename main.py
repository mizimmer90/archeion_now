"""Main orchestration script for archeion_now."""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from dotenv import load_dotenv
from tqdm import tqdm

from config import load_config, Config
from arxiv_fetcher import ArxivFetcher, PaperMetadata
from agent import LLMAgent, RelevanceDecision, PaperSummary
from pdf_reader import extract_text_from_pdf
from output_manager import OutputManager


def get_package_dir() -> Path:
    """Get the directory where the package files are installed."""
    # Get the directory of this module
    # When installed, this will point to the installed package location
    package_dir = Path(__file__).parent.absolute()
    return package_dir


def get_default_config_path() -> Path:
    """Get the default config.yaml path from the package."""
    package_dir = get_package_dir()
    return package_dir / "config.yaml"


def get_default_interests_path() -> Path:
    """Get the default interests.txt path from the package."""
    package_dir = get_package_dir()
    return package_dir / "interests.txt"


def load_interests(interests_file: Path) -> str:
    """Load user interests from file."""
    if not interests_file.exists():
        raise FileNotFoundError(f"Interests file not found: {interests_file}")
    
    with open(interests_file, 'r', encoding='utf-8') as f:
        return f.read()


@dataclass
class ProcessingResult:
    """Result of processing a single paper."""
    paper: PaperMetadata
    initial_decision: RelevanceDecision
    final_decision: RelevanceDecision
    summary: Optional[PaperSummary] = None
    is_cached: bool = False
    test_label: Optional[str] = None  # "positive" or "negative" for test mode


def load_json_config() -> Optional[Config]:
    """Load configuration from unified JSON config file."""
    import json
    json_config_path = Path.home() / '.archeion_now_config.json'
    
    if not json_config_path.exists():
        return None
    
    try:
        with open(json_config_path, 'r') as f:
            data = json.load(f)
        
        # Convert JSON config to Config object format
        # Map papers_dir to output.output_dir for compatibility
        config_data = {
            'interests_file': Path(data.get('interests_file', './interests.txt')),
            'relevance_threshold': data.get('relevance_threshold', 0.5),
            'arxiv': data.get('arxiv', {}),
            'llm': data.get('llm', {}),
            'output': {
                'output_dir': Path(data.get('papers_dir', './papers')),  # Use papers_dir
                'create_subdirs': data.get('output', {}).get('create_subdirs', True),
                'include_pdf': data.get('output', {}).get('include_pdf', False)
            }
        }
        
        return Config(**config_data)
    except Exception as e:
        print(f"Warning: Could not load JSON config: {e}")
        return None


def initialize_components(config_path: Optional[Path] = None, 
                         interests_file: Optional[Path] = None) -> Tuple[Config, str, LLMAgent, OutputManager, ArxivFetcher]:
    """
    Shared initialization logic for loading config, interests, and creating components.
    
    Returns:
        Tuple of (config, interests, agent, output_manager, fetcher)
    """
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    # Priority: JSON config (if config_path is None) > YAML config > defaults
    config = None
    
    if config_path is None:
        # Try to load from JSON config first
        config = load_json_config()
        
        if config is None:
            # Fall back to YAML config
            config_path = get_default_config_path()
            if config_path.exists():
                config = load_config(config_path)
            else:
                # Use defaults
                config = Config.default()
    else:
        # Use provided YAML config path
        config = load_config(config_path)
    
    # If interests_file in config is relative, resolve it
    if not config.interests_file.is_absolute():
        if config_path:
            config_dir = config_path.parent
            resolved_interests = (config_dir / config.interests_file).resolve()
            if resolved_interests.exists():
                config.interests_file = resolved_interests
        else:
            # For JSON config, resolve relative to home or current dir
            resolved_interests = Path(config.interests_file).expanduser().resolve()
            if resolved_interests.exists():
                config.interests_file = resolved_interests
    
    # Set interests_file: CLI argument > config file setting > package default
    if interests_file is None:
        # Check if config interests_file exists, otherwise use package default
        if not config.interests_file.exists():
            package_interests = get_default_interests_path()
            if package_interests.exists():
                config.interests_file = package_interests
    else:
        # Override with CLI-provided path
        config.interests_file = interests_file
    
    # Load interests
    try:
        interests = load_interests(config.interests_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Interests file not found: {e}")
    
    # Initialize components
    fetcher = ArxivFetcher(
        categories=config.arxiv.categories,
        days_back=config.arxiv.days_back,
        max_papers=config.arxiv.max_papers
    )
    
    agent = LLMAgent(config.llm, interests)
    
    output_manager = OutputManager(
        output_dir=config.output.output_dir,
        create_subdirs=config.output.create_subdirs
    )
    
    return config, interests, agent, output_manager, fetcher


def process_single_paper(
    paper: PaperMetadata,
    agent: LLMAgent,
    output_manager: OutputManager,
    fetcher: ArxivFetcher,
    config: Config,
    relevance_threshold: float,
    test_label: Optional[str] = None,
    save_summary: bool = True,
    verbose: bool = True,
    skip_cache: bool = False
) -> ProcessingResult:
    """
    Process a single paper through the full pipeline.
    
    Args:
        paper: Paper to process
        agent: LLM agent
        output_manager: Output manager
        fetcher: Arxiv fetcher
        config: Configuration
        relevance_threshold: Threshold for relevance
        test_label: Optional test label ("positive" or "negative")
        save_summary: Whether to save summary to disk (only if paper is relevant)
        verbose: Whether to print progress
        skip_cache: If True, skip cache check and reprocess even if cached
    
    Returns:
        ProcessingResult with all decisions and summary
    """
    # Check cache first (unless skip_cache is True)
    cached_decision = None
    is_cached = False
    if not skip_cache:
        cached_decision = output_manager.is_cached(paper.doi)
        is_cached = cached_decision is not None
    
    if is_cached:
        # Get cached data
        cached_data = output_manager.get_cached_data(paper.doi)
        cached_reasoning = cached_data.get("reasoning", "Previously assessed (cached)") if cached_data else "Previously assessed (cached)"
        cached_confidence = cached_data.get("confidence", 1.0) if cached_data else 1.0
        cached_impact = cached_data.get("impact", cached_data.get("estimated_impact", 0.0)) if cached_data else 0.0
        if cached_data and "relevance" in cached_data:
            cached_relevance = cached_data.get("relevance", 0.0)
        else:
            cached_relevance = 1.0 if cached_decision else 0.0
        
        decision = RelevanceDecision(
            relevance=cached_relevance,
            confidence=cached_confidence,
            reasoning=cached_reasoning,
            impact=cached_impact
        )
        final_decision = decision
        summary = None
    else:
        # Check relevance with LLM
        if verbose:
            print(f"  Checking relevance...")
        decision = agent.check_relevance(paper)
        
        # Only process full paper if initial relevance is above threshold
        if decision.relevance >= relevance_threshold:
            # Download and extract PDF if needed
            full_text = None
            if config.output.include_pdf:
                if verbose:
                    print(f"  Downloading PDF...")
                pdf_path = fetcher.download_pdf(paper, config.output.output_dir / "pdfs")
                if pdf_path:
                    full_text = extract_text_from_pdf(pdf_path)
            
            # Summarize paper
            if verbose:
                print(f"  Summarizing...")
            summary = agent.summarize_paper(paper, full_text)
            
            # Re-evaluate relevance after reading the full paper
            if verbose:
                print(f"  Re-evaluating relevance after reading full paper...")
            final_decision = agent.recheck_relevance_after_summary(paper, summary, full_text)
        else:
            # Not relevant, no need to process further
            summary = None
            final_decision = decision
        
        # Save decision to cache
        output_manager.save_decision(
            paper.doi,
            final_decision.relevance,
            final_decision.reasoning,
            final_decision.confidence,
            final_decision.impact,
            paper.title,
            relevance_threshold=relevance_threshold,
            test_label=test_label
        )
    
    return ProcessingResult(
        paper=paper,
        initial_decision=decision,
        final_decision=final_decision,
        summary=summary,
        is_cached=is_cached,
        test_label=test_label
    )


def load_dois_from_input(input_value: Optional[str]) -> list[str]:
    """
    Load DOIs from either a file path or a comma/space-separated string.
    
    Args:
        input_value: Either a file path (if file exists) or a string of DOIs
        
    Returns:
        List of DOI strings
    """
    if not input_value:
        return []
    
    # Check if it's a file path
    file_path = Path(input_value)
    if file_path.exists() and file_path.is_file():
        # Load from file (one DOI per line)
        dois = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    dois.append(line)
        return dois
    else:
        # Treat as space/comma-separated string
        # Split by comma or whitespace
        import re
        dois = re.split(r'[,\s]+', input_value.strip())
        return [d for d in dois if d]  # Remove empty strings


def main(config_path: Optional[Path] = None, interests_file: Optional[Path] = None, max_papers: Optional[int] = None, skip_cache: bool = False) -> int:
    """
    Main execution function.
    
    Args:
        config_path: Path to config.yaml file (defaults to package's config.yaml)
        interests_file: Path to interests.txt file (defaults to package's interests.txt if not in config)
        max_papers: Maximum number of papers to fetch (overrides config if provided)
        skip_cache: If True, skip cache check and reprocess all papers even if cached
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        config, interests, agent, output_manager, fetcher = initialize_components(config_path, interests_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create an interests.txt file with your research interests.")
        return 1
    
    # Use CLI-provided max_papers if given, otherwise use config value
    max_papers_to_use = max_papers if max_papers is not None else config.arxiv.max_papers
    relevance_threshold = config.relevance_threshold
    
    # When max_papers is set, we want to process that many NEW papers (not cached)
    # So we fetch with a higher limit to ensure we have enough papers to process
    # Use a multiplier (5x) to account for cached papers and non-relevant papers
    if max_papers_to_use and not skip_cache:
        fetch_limit = max_papers_to_use * 5  # Fetch 5x to account for cache hits and non-relevant
        fetcher.max_papers = fetch_limit
    else:
        fetcher.max_papers = max_papers_to_use
    
    print("=" * 60)
    print("Archeion Now - Arxiv Paper Processing")
    print("=" * 60)
    if config_path:
        print(f"Configuration loaded from: {config_path}")
    else:
        json_config = Path.home() / '.archeion_now_config.json'
        if json_config.exists():
            print(f"Configuration loaded from: {json_config}")
        else:
            print(f"Configuration loaded from: {get_default_config_path()}")
    print(f"Interests file: {config.interests_file}")
    print(f"Output directory: {config.output.output_dir}")
    print(f"Arxiv categories: {', '.join(config.arxiv.categories)}")
    print(f"Days back: {config.arxiv.days_back}")
    if max_papers_to_use:
        if skip_cache:
            print(f"Max papers: {max_papers_to_use} (re-run mode, will process all)")
        else:
            print(f"Max papers: {max_papers_to_use} (will process {max_papers_to_use} NEW papers, cached papers don't count)")
    else:
        print(f"Max papers: unlimited")
    print(f"Relevance threshold: {relevance_threshold}")
    print(f"LLM Provider: {config.llm.provider} ({config.llm.model})")
    print("=" * 60)
    print()
    print(f"Loaded interests from {config.interests_file}")
    print()
    
    # Fetch papers
    print("Fetching papers from arxiv...")
    papers = fetcher.fetch_recent_papers()
    
    if not papers:
        print("No papers found. Exiting.")
        return 0
    
    print(f"Found {len(papers)} papers to process")
    print()
    
    # Process papers
    relevant_papers = []
    summaries = []
    paper_metadatas = []
    processed_papers = []  # Track all papers we've processed
    new_papers_processed = 0  # Count only newly processed papers (not cached)
    
    print("Processing papers...")
    print("-" * 60)
    
    # Load cache and show stats
    if not skip_cache:
        cached_count = sum(1 for p in papers if output_manager.is_cached(p.doi) is not None)
        if cached_count > 0:
            print(f"Found {cached_count} papers with cached assessments (these won't count toward max_papers)")
    else:
        print("Re-run mode: Skipping cache, will reprocess all papers")
    print()
    
    # Track which papers we've already seen (to avoid duplicates if we fetch more)
    seen_dois = set()
    
    # Process papers until we've processed max_papers new papers (if limit is set)
    paper_index = 0
    while paper_index < len(papers):
        paper = papers[paper_index]
        
        # Skip if we've already processed this paper
        if paper.doi and paper.doi in seen_dois:
            paper_index += 1
            continue
        
        # Check if we've reached the limit for new papers
        if max_papers_to_use and not skip_cache and new_papers_processed >= max_papers_to_use:
            print(f"\nReached limit of {max_papers_to_use} newly processed papers. Stopping.")
            break
        
        result = process_single_paper(
            paper=paper,
            agent=agent,
            output_manager=output_manager,
            fetcher=fetcher,
            config=config,
            relevance_threshold=relevance_threshold,
            save_summary=True,
            verbose=True,
            skip_cache=skip_cache
        )
        
        processed_papers.append(paper)
        if paper.doi:
            seen_dois.add(paper.doi)
        
        # Count this as a new paper if it wasn't cached (or if we're in skip_cache mode)
        if not result.is_cached or skip_cache:
            new_papers_processed += 1
        
        # Check if paper is relevant based on threshold
        if result.final_decision.relevance >= relevance_threshold:
            cache_prefix = "[Cache] " if result.is_cached else ""
            progress_info = ""
            if max_papers_to_use and not skip_cache:
                progress_info = f" [New: {new_papers_processed}/{max_papers_to_use}]"
            print(f"\n{cache_prefix}✓ Relevant: {paper.title[:80]}...{progress_info}")
            print(f"  Relevance: {result.initial_decision.relevance:.2f}")
            if not result.is_cached:
                print(f"  Confidence: {result.initial_decision.confidence:.2f}")
            else:
                print(f"  Confidence: {result.initial_decision.confidence:.2f} (cached)")
            print(f"  Impact: {result.initial_decision.impact:.2f}")
            if result.initial_decision.reasoning:
                print(f"  Reasoning: {result.initial_decision.reasoning[:100]}...")
            
            # Only process PDF and summary if not cached
            if not result.is_cached:
                # Display re-evaluation results
                print(f"  Final Relevance: {result.final_decision.relevance:.2f}")
                print(f"  Final Confidence: {result.final_decision.confidence:.2f}")
                print(f"  Final Impact: {result.final_decision.impact:.2f}")
                if result.final_decision.reasoning:
                    print(f"  Final Reasoning: {result.final_decision.reasoning[:100]}...")
                
                # Paper is still relevant after full review (we're already in the >= threshold branch)
                print(f"  ✓ Paper confirmed as relevant after full review")
                # Save summary
                if result.summary:
                    output_path = output_manager.save_summary(result.summary, paper)
                    print(f"  Saved to: {output_path}")
                    relevant_papers.append(paper)
                    summaries.append(result.summary)
                    paper_metadatas.append(paper)
            else:
                # For cached papers, skip PDF download and summary generation
                print(f"  Using cached assessment (skipping PDF download and summary)")
                relevant_papers.append(paper)
        else:
            cache_prefix = "[Cache] " if result.is_cached else ""
            progress_info = ""
            if max_papers_to_use and not skip_cache:
                progress_info = f" [New: {new_papers_processed}/{max_papers_to_use}]"
            print(f"\n{cache_prefix}✗ Not relevant: {paper.title[:80]}...{progress_info}")
            print(f"  Relevance: {result.final_decision.relevance:.2f} (threshold: {relevance_threshold})")
            if result.final_decision.reasoning:
                print(f"  Reason: {result.final_decision.reasoning[:100]}...")
        
        paper_index += 1
    
    print()
    print("-" * 60)
    print(f"Processing complete!")
    print(f"Total papers examined: {len(processed_papers)}")
    if max_papers_to_use and not skip_cache:
        print(f"New papers processed: {new_papers_processed} / {max_papers_to_use}")
    print(f"Relevant papers: {len(relevant_papers)} / {len(processed_papers)}")
    print()
    
    # Create index
    if summaries:
        print("Creating index...")
        output_manager.save_index(summaries, paper_metadatas)
        index_path = config.output.output_dir / "index.md"
        print(f"Index saved to: {index_path}")
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    
    return 0


def test_papers(config_path: Optional[Path] = None, interests_file: Optional[Path] = None,
                positive_input: Optional[str] = None, negative_input: Optional[str] = None) -> int:
    """
    Test paper classification by processing positive and negative examples.
    
    Args:
        config_path: Path to config.yaml file (defaults to package's config.yaml)
        interests_file: Path to interests.txt file (defaults to package's interests.txt if not in config)
        positive_input: File path or comma/space-separated string of DOI identifiers for papers that should match interests
        negative_input: File path or comma/space-separated string of DOI identifiers for papers that should NOT match interests
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        config, interests, agent, output_manager, fetcher = initialize_components(config_path, interests_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create an interests.txt file with your research interests.")
        return 1
    
    relevance_threshold = config.relevance_threshold
    
    print("=" * 60)
    print("Archeion Now - Paper Classification Test")
    print("=" * 60)
    print(f"Configuration loaded from: {config_path or get_default_config_path()}")
    print(f"Interests file: {config.interests_file}")
    print(f"Relevance threshold: {relevance_threshold}")
    print(f"LLM Provider: {config.llm.provider} ({config.llm.model})")
    print("=" * 60)
    print()
    print(f"Loaded interests from {config.interests_file}")
    print()
    
    # Load DOIs from input (file or string)
    positive_dois = load_dois_from_input(positive_input)
    negative_dois = load_dois_from_input(negative_input)
    
    if not positive_dois and not negative_dois:
        print("Error: At least one positive or negative DOI must be provided.")
        return 1
    
    # Fetch test papers
    print("Fetching test papers from arxiv...")
    all_dois = (positive_dois or []) + (negative_dois or [])
    papers = fetcher.fetch_papers_by_doi(all_dois)
    
    if not papers:
        print("No papers found. Exiting.")
        return 1
    
    print(f"Found {len(papers)} papers to process")
    print()
    
    # Create mapping from input DOI/ID to label
    input_to_label = {}
    for doi in positive_dois:
        input_to_label[doi] = "positive"
        # Also normalize common formats
        if "10.48550/arXiv." in doi:
            arxiv_id = doi.replace("10.48550/arXiv.", "")
            input_to_label[arxiv_id] = "positive"
            input_to_label[arxiv_id.split('v')[0]] = "positive"  # Without version
        elif doi.startswith("arXiv:"):
            arxiv_id = doi.replace("arXiv:", "")
            input_to_label[arxiv_id] = "positive"
            input_to_label[arxiv_id.split('v')[0]] = "positive"
    for doi in negative_dois:
        input_to_label[doi] = "negative"
        # Also normalize common formats
        if "10.48550/arXiv." in doi:
            arxiv_id = doi.replace("10.48550/arXiv.", "")
            input_to_label[arxiv_id] = "negative"
            input_to_label[arxiv_id.split('v')[0]] = "negative"  # Without version
        elif doi.startswith("arXiv:"):
            arxiv_id = doi.replace("arXiv:", "")
            input_to_label[arxiv_id] = "negative"
            input_to_label[arxiv_id.split('v')[0]] = "negative"
    
    # Create mapping from paper DOI/ID to label
    paper_to_label = {}
    for paper in papers:
        # Try to match by various identifiers
        label = None
        # Try direct DOI match
        if paper.doi and paper.doi in input_to_label:
            label = input_to_label[paper.doi]
        # Try arxiv_id match
        elif paper.arxiv_id in input_to_label:
            label = input_to_label[paper.arxiv_id]
        # Try arxiv_id without version
        elif paper.arxiv_id.split('v')[0] in input_to_label:
            label = input_to_label[paper.arxiv_id.split('v')[0]]
        # Try matching DOI patterns
        elif paper.doi:
            for input_doi, lbl in input_to_label.items():
                if input_doi in paper.doi or paper.doi in input_doi:
                    label = lbl
                    break
        
        if label:
            # Store label by both DOI and arxiv_id for easy lookup
            if paper.doi:
                paper_to_label[paper.doi] = label
            paper_to_label[paper.arxiv_id] = label
            paper_to_label[paper.arxiv_id.split('v')[0]] = label
    
    # Process papers
    results = []  # List of ProcessingResult
    
    print("Processing papers...")
    print("-" * 60)
    
    for paper in tqdm(papers, desc="Processing papers"):
        # Determine label for this paper
        label = (paper_to_label.get(paper.doi) or 
                paper_to_label.get(paper.arxiv_id) or
                paper_to_label.get(paper.arxiv_id.split('v')[0]))
        
        if not label:
            print(f"Warning: Could not determine label for paper {paper.doi or paper.arxiv_id} (title: {paper.title[:50]}...)")
            continue
        
        # Process paper using shared function
        result = process_single_paper(
            paper=paper,
            agent=agent,
            output_manager=output_manager,
            fetcher=fetcher,
            config=config,
            relevance_threshold=relevance_threshold,
            test_label=label,
            save_summary=False,  # Don't save summaries in test mode
            verbose=True
        )
        
        results.append(result)
        
        # Display result
        cache_prefix = "[Cache] " if result.is_cached else ""
        print(f"\n{cache_prefix}Paper: {paper.title[:80]}...")
        print(f"  Label: {label}")
        print(f"  Initial Relevance: {result.initial_decision.relevance:.2f}")
        print(f"  Final Relevance: {result.final_decision.relevance:.2f}")
        print(f"  Final Confidence: {result.final_decision.confidence:.2f}")
    
    print()
    print("-" * 60)
    print("Calculating metrics...")
    print("-" * 60)
    
    # Calculate metrics
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for result in results:
        predicted_relevant = result.final_decision.relevance >= relevance_threshold
        actual_relevant = (result.test_label == "positive")
        
        if actual_relevant and predicted_relevant:
            true_positives += 1
        elif not actual_relevant and not predicted_relevant:
            true_negatives += 1
        elif not actual_relevant and predicted_relevant:
            false_positives += 1
        elif actual_relevant and not predicted_relevant:
            false_negatives += 1
    
    total = len(results)
    
    # Calculate additional metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    
    # Display results
    print()
    print("=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Total papers tested: {total}")
    print(f"  Positive examples: {sum(1 for r in results if r.test_label == 'positive')}")
    print(f"  Negative examples: {sum(1 for r in results if r.test_label == 'negative')}")
    print()
    print("Classification Results:")
    print(f"  True Positives (TP):  {true_positives:3d} - Correctly identified as relevant")
    print(f"  True Negatives (TN):  {true_negatives:3d} - Correctly identified as not relevant")
    print(f"  False Positives (FP): {false_positives:3d} - Incorrectly identified as relevant")
    print(f"  False Negatives (FN): {false_negatives:3d} - Incorrectly identified as not relevant")
    print()
    print("Metrics:")
    print(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"  Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"  F1 Score:  {f1_score:.3f}")
    print()
    
    # Show detailed breakdown
    print("Detailed Breakdown:")
    print("-" * 60)
    
    # False Positives
    if false_positives > 0:
        print("\nFalse Positives (incorrectly marked as relevant):")
        for result in results:
            predicted_relevant = result.final_decision.relevance >= relevance_threshold
            actual_relevant = (result.test_label == "positive")
            if not actual_relevant and predicted_relevant:
                print(f"  - {result.paper.title[:70]}...")
                print(f"    Relevance: {result.final_decision.relevance:.2f}, Label: {result.test_label}")
    
    # False Negatives
    if false_negatives > 0:
        print("\nFalse Negatives (incorrectly marked as not relevant):")
        for result in results:
            predicted_relevant = result.final_decision.relevance >= relevance_threshold
            actual_relevant = (result.test_label == "positive")
            if actual_relevant and not predicted_relevant:
                print(f"  - {result.paper.title[:70]}...")
                print(f"    Relevance: {result.final_decision.relevance:.2f}, Label: {result.test_label}")
    
    print()
    print("=" * 60)
    print("Done! Results saved to cache with test labels.")
    print("=" * 60)
    
    return 0


def create_cli() -> argparse.ArgumentParser:
    """Create and configure CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Archeion Now - Automated ArXiv paper processing and summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config.yaml and interests.txt
  archeion_now

  # Use custom config file
  archeion_now --config custom_config.yaml

  # Use custom config and interests files
  archeion_now --config custom_config.yaml --interests custom_interests.txt

  # Use custom interests file with default config
  archeion_now --interests my_interests.txt

  # Limit to 10 papers
  archeion_now --max-papers 10

  # Test paper classification with positive and negative examples (as arguments)
  archeion_now test_papers --positive "10.48550/arXiv.2301.12345,10.48550/arXiv.2302.23456" --negative "10.48550/arXiv.2303.34567"

  # Test with DOIs from files (one DOI per line)
  archeion_now test_papers --positive positive_examples.txt --negative negative_examples.txt

  # Test with custom config and interests files
  archeion_now test_papers --config custom_config.yaml --interests custom_interests.txt --positive positive_examples.txt --negative negative_examples.txt

  # Test with custom config only
  archeion_now test_papers --config custom_config.yaml --positive positive_examples.txt --negative negative_examples.txt

  # Test with custom interests file only
  archeion_now test_papers --interests custom_interests.txt --positive positive_examples.txt --negative negative_examples.txt
        """
    )
    
    # Common arguments (available to all commands)
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config.yaml file (default: ./config.yaml)"
    )
    
    parser.add_argument(
        "--interests", "-i",
        type=str,
        default=None,
        help="Path to interests.txt file (overrides config file setting if provided)"
    )
    
    # Add max-papers argument for default/main command
    parser.add_argument(
        "--max-papers", "-m",
        type=int,
        default=None,
        help="Maximum number of papers to fetch (overrides config file setting if provided) - only for main command"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run', metavar='COMMAND')
    
    # Main command (default) - no need for separate parser, handled by default
    # subparsers.add_parser('run', help='Run normal paper processing (default)')
    
    # Test papers command
    test_parser = subparsers.add_parser(
        'test_papers', 
        help='Test paper classification with positive and negative examples',
        description='Test paper classification by processing positive and negative examples. '
                    'Uses the same processing pipeline as the main command but calculates '
                    'classification metrics instead of saving summaries.',
        parents=[],  # We'll add config/interests explicitly
        add_help=False  # We'll add help manually to show all options
    )
    # Add help option
    test_parser.add_argument(
        "-h", "--help",
        action="help",
        help="show this help message and exit"
    )
    # Add config and interests explicitly so they show in help
    test_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config.yaml file (default: ./config.yaml or package default)"
    )
    test_parser.add_argument(
        "--interests", "-i",
        type=str,
        default=None,
        help="Path to interests.txt file (overrides config file setting if provided)"
    )
    test_parser.add_argument(
        "--positive", "-p",
        type=str,
        default=None,
        help="File path (one DOI per line, # for comments) or quoted comma/space-separated string of DOI identifiers for papers that should match interests (positive examples). Example: --positive 'DOI1,DOI2' or --positive positive_dois.txt"
    )
    test_parser.add_argument(
        "--negative", "-n",
        type=str,
        default=None,
        help="File path (one DOI per line, # for comments) or quoted comma/space-separated string of DOI identifiers for papers that should NOT match interests (negative examples). Example: --negative 'DOI1,DOI2' or --negative negative_dois.txt"
    )
    
    return parser


def cli_main():
    """CLI entry point."""
    parser = create_cli()
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    interests_file = Path(args.interests) if args.interests else None
    
    # Handle subcommands
    if args.command == 'test_papers':
        positive_input = getattr(args, 'positive', None)
        negative_input = getattr(args, 'negative', None)
        return test_papers(
            config_path=config_path,
            interests_file=interests_file,
            positive_input=positive_input,
            negative_input=negative_input
        )
    elif args.command is None:
        # Default to main command (no subcommand specified)
        max_papers = getattr(args, 'max_papers', None)
        return main(config_path=config_path, interests_file=interests_file, max_papers=max_papers)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())

