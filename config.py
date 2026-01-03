"""Configuration management for archeion_now."""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import yaml


class ArxivConfig(BaseModel):
    """Configuration for Arxiv paper fetching."""
    days_back: int = Field(default=7, description="Number of days to look back for papers")
    categories: list[str] = Field(
        default=["cs.AI", "cs.LG", "cs.CV", "stat.ML"],
        description="Arxiv categories to search (e.g., cs.AI, cs.LG, stat.ML)"
    )
    max_papers: Optional[int] = Field(default=None, description="Maximum papers to fetch (None for no limit)")


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    provider: str = Field(default="openai", description="LLM provider: 'openai' or 'anthropic'")
    model: str = Field(default="gpt-4o-mini", description="Model name to use")
    temperature: float = Field(default=0.3, description="Temperature for LLM generation")


class OutputConfig(BaseModel):
    """Configuration for output organization."""
    output_dir: Path = Field(default=Path("./papers"), description="Directory to save paper summaries")
    create_subdirs: bool = Field(default=True, description="Create subdirectories for organization")
    include_pdf: bool = Field(default=False, description="Save PDF files locally")


class Config(BaseModel):
    """Main configuration model."""
    interests_file: Path = Field(default=Path("./interests.txt"), description="Path to interests/summary structure file")
    relevance_threshold: float = Field(default=0.5, description="Relevance threshold for determining if a paper should be processed (0.0-1.0)")
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert string paths to Path objects
        if 'interests_file' in data:
            data['interests_file'] = Path(data['interests_file'])
        if 'output' in data and 'output_dir' in data['output']:
            data['output']['output_dir'] = Path(data['output']['output_dir'])
        
        return cls(**data)

    def save_yaml(self, config_path: Path):
        """Save configuration to YAML file."""
        data = self.model_dump()
        # Convert Path objects to strings for YAML
        if isinstance(data.get('interests_file'), Path):
            data['interests_file'] = str(data['interests_file'])
        if 'output' in data and isinstance(data['output'].get('output_dir'), Path):
            data['output']['output_dir'] = str(data['output']['output_dir'])
        
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file or use defaults."""
    if config_path is None:
        config_path = Path("./config.yaml")
    
    if config_path.exists():
        return Config.from_yaml(config_path)
    else:
        # Create default config file
        config = Config.default()
        config.save_yaml(config_path)
        print(f"Created default config file at {config_path}")
        return config

