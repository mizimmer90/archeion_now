"""Main orchestration script for archeion_now."""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm

from config import load_config
from arxiv_fetcher import ArxivFetcher
from agent import LLMAgent, RelevanceDecision
from pdf_reader import extract_text_from_pdf
from output_manager import OutputManager

# Relevance threshold for determining if a paper should be processed
RELEVANCE_THRESHOLD = 0.5


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


def main(config_path: Optional[Path] = None, interests_file: Optional[Path] = None, max_papers: Optional[int] = None) -> int:
    """
    Main execution function.
    
    Args:
        config_path: Path to config.yaml file (defaults to package's config.yaml)
        interests_file: Path to interests.txt file (defaults to package's interests.txt if not in config)
        max_papers: Maximum number of papers to fetch (overrides config if provided)
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Load environment variables
    load_dotenv()
    
    # Load configuration - use package default if not provided
    if config_path is None:
        config_path = get_default_config_path()
    
    config = load_config(config_path)
    
    # If interests_file in config is relative, resolve it relative to config file location
    if not config.interests_file.is_absolute():
        config_dir = config_path.parent
        resolved_interests = (config_dir / config.interests_file).resolve()
        if resolved_interests.exists():
            config.interests_file = resolved_interests
    
    # Set interests_file: CLI argument > resolved config file setting > package default
    if interests_file is None:
        # Check if config interests_file exists, otherwise use package default
        if not config.interests_file.exists():
            package_interests = get_default_interests_path()
            if package_interests.exists():
                config.interests_file = package_interests
    else:
        # Override with CLI-provided path
        config.interests_file = interests_file
    
    # Use CLI-provided max_papers if given, otherwise use config value
    max_papers_to_use = max_papers if max_papers is not None else config.arxiv.max_papers
    
    print("=" * 60)
    print("Archeion Now - Arxiv Paper Processing")
    print("=" * 60)
    print(f"Configuration loaded from: {config_path}")
    print(f"Interests file: {config.interests_file}")
    print(f"Output directory: {config.output.output_dir}")
    print(f"Arxiv categories: {', '.join(config.arxiv.categories)}")
    print(f"Days back: {config.arxiv.days_back}")
    print(f"Max papers: {max_papers_to_use if max_papers_to_use else 'unlimited'}")
    print(f"LLM Provider: {config.llm.provider} ({config.llm.model})")
    print("=" * 60)
    print()
    
    # Load interests
    try:
        interests = load_interests(config.interests_file)
        print(f"Loaded interests from {config.interests_file}")
        print()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create an interests.txt file with your research interests.")
        return 1
    
    # Initialize components
    print("Initializing components...")
    fetcher = ArxivFetcher(
        categories=config.arxiv.categories,
        days_back=config.arxiv.days_back,
        max_papers=max_papers_to_use
    )
    
    agent = LLMAgent(config.llm, interests)
    
    output_manager = OutputManager(
        output_dir=config.output.output_dir,
        create_subdirs=config.output.create_subdirs
    )
    
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
    
    print("Processing papers...")
    print("-" * 60)
    
    # Load cache and show stats
    cached_count = sum(1 for p in papers if output_manager.is_cached(p.doi) is not None)
    if cached_count > 0:
        print(f"Found {cached_count} papers with cached assessments")
    print()
    
    for paper in tqdm(papers, desc="Processing papers"):
        # Check cache first
        cached_decision = output_manager.is_cached(paper.doi)
        is_cached = cached_decision is not None
        
        if is_cached:
            # Get cached data including reasoning, confidence, impact, and relevance
            cached_data = output_manager.get_cached_data(paper.doi)
            cached_reasoning = cached_data.get("reasoning", "Previously assessed (cached)") if cached_data else "Previously assessed (cached)"
            cached_confidence = cached_data.get("confidence", 1.0) if cached_data else 1.0
            # Support both new 'impact' and legacy 'estimated_impact' keys
            cached_impact = cached_data.get("impact", cached_data.get("estimated_impact", 0.0)) if cached_data else 0.0
            # If relevance is not in cache, infer from status (backward compatibility)
            if cached_data and "relevance" in cached_data:
                cached_relevance = cached_data.get("relevance", 0.0)
            else:
                # Backward compatibility: infer relevance from cached decision (0.0 or 1.0)
                cached_relevance = 1.0 if cached_decision else 0.0
            
            # Use cached decision
            decision = RelevanceDecision(
                relevance=cached_relevance,
                confidence=cached_confidence,
                reasoning=cached_reasoning,
                impact=cached_impact
            )
        else:
            # Check relevance with LLM
            decision = agent.check_relevance(paper)
            # Save decision to cache with all metrics
            output_manager.save_decision(
                paper.doi, 
                decision.relevance, 
                decision.reasoning,
                decision.confidence,
                decision.impact,
                paper.title,
                relevance_threshold=RELEVANCE_THRESHOLD
            )
        
        # Check if paper is relevant based on threshold
        if decision.relevance >= RELEVANCE_THRESHOLD:
            cache_prefix = "[Cache] " if is_cached else ""
            print(f"\n{cache_prefix}✓ Relevant: {paper.title[:80]}...")
            print(f"  Relevance: {decision.relevance:.2f}")
            if not is_cached:
                print(f"  Confidence: {decision.confidence:.2f}")
            else:
                print(f"  Confidence: {decision.confidence:.2f} (cached)")
            print(f"  Impact: {decision.impact:.2f}")
            if decision.reasoning:
                print(f"  Reasoning: {decision.reasoning[:100]}...")
            
            # Only process PDF and summary if not cached
            if not is_cached:
                # Download and extract PDF if needed
                full_text = None
                if config.output.include_pdf:
                    pdf_path = fetcher.download_pdf(paper, config.output.output_dir / "pdfs")
                    if pdf_path:
                        full_text = extract_text_from_pdf(pdf_path)
                
                # Summarize paper
                print(f"  Summarizing...")
                summary = agent.summarize_paper(paper, full_text)
                
                # Re-evaluate relevance after reading the full paper
                print(f"  Re-evaluating relevance after reading full paper...")
                final_decision = agent.recheck_relevance_after_summary(paper, summary, full_text)
                
                # Display re-evaluation results
                print(f"  Final Relevance: {final_decision.relevance:.2f}")
                print(f"  Final Confidence: {final_decision.confidence:.2f}")
                print(f"  Final Impact: {final_decision.impact:.2f}")
                if final_decision.reasoning:
                    print(f"  Final Reasoning: {final_decision.reasoning[:100]}...")
                
                # Check if paper should still be considered relevant after full review
                if final_decision.relevance < RELEVANCE_THRESHOLD:
                    print(f"  ✗ Paper rejected after full review - relevance {final_decision.relevance:.2f} below threshold {RELEVANCE_THRESHOLD}")
                    # Update cache with the final decision (rejected)
                    output_manager.save_decision(
                        paper.doi, 
                        final_decision.relevance, 
                        final_decision.reasoning,
                        final_decision.confidence,
                        final_decision.impact,
                        paper.title,
                        relevance_threshold=RELEVANCE_THRESHOLD
                    )
                else:
                    # Paper is still relevant after full review
                    print(f"  ✓ Paper confirmed as relevant after full review")
                    # Save summary
                    output_path = output_manager.save_summary(summary, paper)
                    print(f"  Saved to: {output_path}")
                    
                    # Update cache with the final decision (approved)
                    output_manager.save_decision(
                        paper.doi, 
                        final_decision.relevance, 
                        final_decision.reasoning,
                        final_decision.confidence,
                        final_decision.impact,
                        paper.title,
                        relevance_threshold=RELEVANCE_THRESHOLD
                    )
                    
                    relevant_papers.append(paper)
                    summaries.append(summary)
                    paper_metadatas.append(paper)
            else:
                # For cached papers, skip PDF download and summary generation
                # (they were already processed in a previous run)
                print(f"  Using cached assessment (skipping PDF download and summary)")
                relevant_papers.append(paper)
                # Note: Don't add to summaries/paper_metadatas since we're skipping summary generation
                # If summaries were created previously, they'll still exist in the output directory
        else:
            cache_prefix = "[Cache] " if is_cached else ""
            print(f"\n{cache_prefix}✗ Not relevant: {paper.title[:80]}...")
            print(f"  Relevance: {decision.relevance:.2f} (threshold: {RELEVANCE_THRESHOLD})")
            if decision.reasoning:
                print(f"  Reason: {decision.reasoning[:100]}...")
    
    print()
    print("-" * 60)
    print(f"Processing complete!")
    print(f"Relevant papers: {len(relevant_papers)} / {len(papers)}")
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

  # Combine flags
  archeion_now --config custom_config.yaml --max-papers 20
        """
    )
    
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
    
    parser.add_argument(
        "--max-papers", "-m",
        type=int,
        default=None,
        help="Maximum number of papers to fetch (overrides config file setting if provided)"
    )
    
    return parser


def cli_main():
    """CLI entry point."""
    parser = create_cli()
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    interests_file = Path(args.interests) if args.interests else None
    max_papers = args.max_papers
    
    return main(config_path=config_path, interests_file=interests_file, max_papers=max_papers)


if __name__ == "__main__":
    sys.exit(cli_main())

