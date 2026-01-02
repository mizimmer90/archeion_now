"""Main orchestration script for archeion_now."""
import os
import sys
from pathlib import Path
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


def main():
    """Main execution function."""
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config_path = Path("./config.yaml")
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    
    config = load_config(config_path)
    
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


if __name__ == "__main__":
    sys.exit(main())

