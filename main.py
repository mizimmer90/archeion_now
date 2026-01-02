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
from agent import LLMAgent
from pdf_reader import extract_text_from_pdf
from output_manager import OutputManager


def load_interests(interests_file: Path) -> str:
    """Load user interests from file."""
    if not interests_file.exists():
        raise FileNotFoundError(f"Interests file not found: {interests_file}")
    
    with open(interests_file, 'r', encoding='utf-8') as f:
        return f.read()


def main(config_path: Optional[Path] = None, interests_file: Optional[Path] = None) -> int:
    """
    Main execution function.
    
    Args:
        config_path: Path to config.yaml file (defaults to ./config.yaml)
        interests_file: Path to interests.txt file (overrides config if provided)
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    if config_path is None:
        config_path = Path("./config.yaml")
    
    config = load_config(config_path)
    
    # Override interests_file if provided via CLI
    if interests_file is not None:
        config.interests_file = interests_file
    
    print("=" * 60)
    print("Archeion Now - Arxiv Paper Processing")
    print("=" * 60)
    print(f"Configuration loaded from: {config_path}")
    print(f"Interests file: {config.interests_file}")
    print(f"Output directory: {config.output.output_dir}")
    print(f"Arxiv categories: {', '.join(config.arxiv.categories)}")
    print(f"Days back: {config.arxiv.days_back}")
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
        max_papers=config.arxiv.max_papers
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
    
    for paper in tqdm(papers, desc="Processing papers"):
        # Check relevance
        decision = agent.check_relevance(paper)
        
        if decision.is_relevant:
            print(f"\n✓ Relevant: {paper.title[:80]}...")
            print(f"  Confidence: {decision.confidence:.2f}")
            print(f"  Reasoning: {decision.reasoning[:100]}...")
            
            # Download and extract PDF if needed
            full_text = None
            if config.output.include_pdf:
                pdf_path = fetcher.download_pdf(paper, config.output.output_dir / "pdfs")
                if pdf_path:
                    full_text = extract_text_from_pdf(pdf_path)
            
            # Summarize paper
            print(f"  Summarizing...")
            summary = agent.summarize_paper(paper, full_text)
            
            # Save summary
            output_path = output_manager.save_summary(summary, paper)
            print(f"  Saved to: {output_path}")
            
            relevant_papers.append(paper)
            summaries.append(summary)
            paper_metadatas.append(paper)
        else:
            print(f"\n✗ Not relevant: {paper.title[:80]}...")
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
  python main.py

  # Use custom config file
  python main.py --config custom_config.yaml

  # Use custom config and interests files
  python main.py --config custom_config.yaml --interests custom_interests.txt

  # Use custom interests file with default config
  python main.py --interests my_interests.txt
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
    
    return parser


def cli_main():
    """CLI entry point."""
    parser = create_cli()
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    interests_file = Path(args.interests) if args.interests else None
    
    return main(config_path=config_path, interests_file=interests_file)


if __name__ == "__main__":
    sys.exit(cli_main())

