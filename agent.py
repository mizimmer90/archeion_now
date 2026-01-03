"""Agentic workflow for paper filtering and summarization."""
import os
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from config import LLMConfig
from arxiv_fetcher import PaperMetadata


@dataclass
class RelevanceDecision:
    """Result of relevance checking."""
    relevance: float  # Estimated relevance to user interests (0.0-1.0)
    confidence: float  # Confidence in the relevance assessment (0.0-1.0)
    reasoning: str
    impact: float = 0.0  # Estimated impact score (0.0-1.0) for the field at large


@dataclass
class PaperSummary:
    """Structured summary of a paper."""
    arxiv_id: str
    title: str
    key_findings: str
    methodology: str
    results: str
    relevance: str
    applications: str
    limitations: str
    category: Optional[str] = None  # For file organization
    
    def to_markdown(self) -> str:
        """Convert summary to markdown format."""
        return f"""# {self.title}

**ArXiv ID:** {self.arxiv_id}

## Key Findings and Contributions
{self.key_findings}

## Methodology Overview
{self.methodology}

## Experimental Results
{self.results}

## Relevance to Interests
{self.relevance}

## Potential Applications or Follow-up Directions
{self.applications}

## Critical Limitations or Concerns
{self.limitations}
"""


class LLMAgent:
    """Agent using LLM for paper processing."""
    
    def __init__(self, config: LLMConfig, interests: str):
        """
        Initialize the LLM agent.
        
        Args:
            config: LLM configuration
            interests: User's interests and summary structure
        """
        self.config = config
        self.interests = interests
        self._setup_client()
    
    def _setup_client(self):
        """Set up the LLM client based on provider."""
        if self.config.provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
            self._call_llm = self._call_openai
        elif self.config.provider == "anthropic":
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = Anthropic(api_key=api_key)
            self._call_llm = self._call_anthropic
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.provider}")
    
    def _call_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Anthropic API."""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=4096,
            temperature=self.config.temperature,
            system=system_prompt or "",
            messages=messages
        )
        return response.content[0].text
    
    def check_relevance(self, paper: PaperMetadata) -> RelevanceDecision:
        """
        Check if a paper is relevant based on title and abstract.
        
        Args:
            paper: PaperMetadata object
            
        Returns:
            RelevanceDecision object
        """
        prompt = f"""You are an AI assistant helping to filter research papers based on user interests.

User Interests and Criteria:
{self.interests}

Paper Title: {paper.title}

Paper Abstract:
{paper.abstract}

Paper Categories: {', '.join(paper.categories)}

Based on the title and abstract, evaluate:
1. The paper's relevance to the user's interests (0.0-1.0)
2. Your confidence in this relevance assessment (0.0-1.0)
3. The paper's potential impact on the research field at large, regardless of personal relevance (0.0-1.0)

Respond in the following JSON format:
{{
    "relevance": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of the relevance assessment",
    "impact": 0.0-1.0
}}

The relevance score should reflect how well the paper aligns with the user's interests:
- 0.0-0.3: Low relevance, minimal alignment
- 0.3-0.6: Moderate relevance, some alignment
- 0.6-0.8: High relevance, strong alignment
- 0.8-1.0: Very high relevance, excellent alignment

The confidence score reflects how certain you are about the relevance assessment:
- High confidence: Clear alignment (or lack thereof) with user interests
- Low confidence: Unclear or ambiguous alignment

The impact score should reflect:
- 0.0-0.3: Incremental work, minor contributions
- 0.3-0.6: Solid contributions with moderate impact potential
- 0.6-0.8: Significant contributions likely to influence the field
- 0.8-1.0: Breakthrough work with transformative potential"""
        
        try:
            response = self._call_llm(prompt, system_prompt="You are a helpful research assistant.")
            # Parse JSON response
            import json
            # Try to extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return RelevanceDecision(
                    relevance=result.get("relevance", 0.0),
                    confidence=result.get("confidence", 0.0),
                    reasoning=result.get("reasoning", ""),
                    impact=result.get("impact", 0.0)
                )
            else:
                # Fallback: assume not relevant if we can't parse
                return RelevanceDecision(
                    relevance=0.0,
                    confidence=0.0,
                    reasoning="Could not parse relevance decision",
                    impact=0.0
                )
        except Exception as e:
            print(f"Error checking relevance: {e}")
            return RelevanceDecision(
                relevance=0.0,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                impact=0.0
            )
    
    def recheck_relevance_after_summary(self, paper: PaperMetadata, summary: PaperSummary, full_text: Optional[str] = None) -> RelevanceDecision:
        """
        Re-check relevance after reading the full paper and generating summary.
        This provides a more accurate assessment based on the complete paper content.
        
        Args:
            paper: PaperMetadata object
            summary: PaperSummary object with the generated summary
            full_text: Optional full text of the paper
            
        Returns:
            RelevanceDecision object with updated metrics
        """
        # Build content for evaluation - use summary fields and full text if available
        content_to_evaluate = f"""Title: {paper.title}

Abstract:
{paper.abstract}

Key Findings: {summary.key_findings}

Methodology: {summary.methodology}

Results: {summary.results}

Relevance: {summary.relevance}

Limitations: {summary.limitations}"""
        
        if full_text:
            # Include a portion of full text for context
            content_to_evaluate += f"\n\nFull Paper Excerpt (first 10000 chars):\n{full_text[:10000]}"
        
        prompt = f"""You are an AI assistant helping to filter research papers based on user interests.

User Interests and Criteria:
{self.interests}

Paper Categories: {', '.join(paper.categories)}

You have now read the ENTIRE paper and generated a summary. Based on the complete paper content (not just the abstract), re-evaluate:
1. The paper's relevance to the user's interests (0.0-1.0) - this should be more accurate now that you've read the full paper
2. Your confidence in this relevance assessment (0.0-1.0) - this should be higher now that you've read the full paper
3. The paper's potential impact on the research field at large (0.0-1.0)

Complete Paper Content and Summary:
{content_to_evaluate}

Based on the FULL paper content, evaluate the relevance, your confidence in that assessment, and the paper's impact.

Respond in the following JSON format:
{{
    "relevance": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of the relevance assessment based on the full content",
    "impact": 0.0-1.0
}}

The relevance score should reflect how well the paper aligns with the user's interests:
- 0.0-0.3: Low relevance, minimal alignment
- 0.3-0.6: Moderate relevance, some alignment
- 0.6-0.8: High relevance, strong alignment
- 0.8-1.0: Very high relevance, excellent alignment

The confidence score reflects how certain you are about the relevance assessment:
- High confidence: Clear alignment (or lack thereof) with user interests based on full paper
- Low confidence: Unclear or ambiguous alignment even after reading full paper

The impact score should reflect:
- 0.0-0.3: Incremental work, minor contributions
- 0.3-0.6: Solid contributions with moderate impact potential
- 0.6-0.8: Significant contributions likely to influence the field
- 0.8-1.0: Breakthrough work with transformative potential"""
        
        try:
            response = self._call_llm(prompt, system_prompt="You are a helpful research assistant.")
            # Parse JSON response
            import json
            # Try to extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return RelevanceDecision(
                    relevance=result.get("relevance", 0.0),
                    confidence=result.get("confidence", 0.0),
                    reasoning=result.get("reasoning", ""),
                    impact=result.get("impact", 0.0)
                )
            else:
                # Fallback: assume not relevant if we can't parse
                return RelevanceDecision(
                    relevance=0.0,
                    confidence=0.0,
                    reasoning="Could not parse relevance decision after full paper review",
                    impact=0.0
                )
        except Exception as e:
            print(f"Error rechecking relevance after summary: {e}")
            return RelevanceDecision(
                relevance=0.0,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                impact=0.0
            )
    
    def summarize_paper(self, paper: PaperMetadata, full_text: Optional[str] = None) -> PaperSummary:
        """
        Summarize a paper based on user's summary structure.
        
        Args:
            paper: PaperMetadata object
            full_text: Optional full text of the paper
            
        Returns:
            PaperSummary object
        """
        content_to_analyze = paper.abstract
        if full_text:
            # Truncate full text if too long (keep first 15000 chars for context)
            content_to_analyze = full_text[:15000] + "\n\n[... paper continues ...]"
        
        prompt = f"""You are an AI assistant helping to summarize research papers based on user interests.

User Interests and Summary Structure:
{self.interests}

Paper Title: {paper.title}
Paper Authors: {', '.join(paper.authors)}
Paper Categories: {', '.join(paper.categories)}
Published: {paper.published}

Paper Content:
{content_to_analyze}

Please provide a comprehensive summary following the user's summary structure. Respond in the following JSON format:
{{
    "key_findings": "key findings and contributions",
    "methodology": "methodology overview",
    "results": "experimental results (if applicable)",
    "relevance": "relevance to user interests",
    "applications": "potential applications or follow-up directions",
    "limitations": "critical limitations or concerns",
    "category": "suggested category for organization (e.g., 'agentic_ai', 'llm_applications', 'computer_vision', etc.)"
}}"""
        
        try:
            response = self._call_llm(prompt, system_prompt="You are a helpful research assistant.")
            import json
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return PaperSummary(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    key_findings=result.get("key_findings", ""),
                    methodology=result.get("methodology", ""),
                    results=result.get("results", ""),
                    relevance=result.get("relevance", ""),
                    applications=result.get("applications", ""),
                    limitations=result.get("limitations", ""),
                    category=result.get("category")
                )
            else:
                # Fallback: create basic summary
                return PaperSummary(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    key_findings="Could not generate summary",
                    methodology="",
                    results="",
                    relevance="",
                    applications="",
                    limitations="",
                    category=None
                )
        except Exception as e:
            print(f"Error summarizing paper: {e}")
            return PaperSummary(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                key_findings=f"Error: {str(e)}",
                methodology="",
                results="",
                relevance="",
                applications="",
                limitations="",
                category=None
            )

