# Archeion Now

An agentic workflow system for automatically discovering, filtering, and summarizing relevant ArXiv papers based on your research interests.

## Features

- 🔍 Automatically fetches recent papers from ArXiv based on categories and date range
- 🤖 Uses LLM agents to intelligently filter papers by relevance
- 📝 Generates structured summaries tailored to your interests
- 📁 Organizes output with category-based subdirectories
- 📊 Creates an index of all processed papers
- ⚙️ Fully configurable via YAML configuration file

## Installation

### Option 1: Install as a package (Recommended)

1. Clone this repository:
```bash
cd archeion_now
```

2. Install in editable mode:
```bash
pip install -e .
```

This will install all dependencies and make the `archeion_now` command available system-wide.

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Option 2: Install dependencies only

1. Clone this repository:
```bash
cd archeion_now
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

You'll need at least one of:
- `OPENAI_API_KEY` (for OpenAI models like GPT-4)
- `ANTHROPIC_API_KEY` (for Anthropic models like Claude)

## Configuration

### 1. Interests File

Create or edit `interests.txt` to describe your research interests and desired summary structure. This file should include:
- Your research interests
- The structure you want for summaries
- Any focus areas or specific criteria

Example:
```
## Interests
I am interested in:
- Agentic AI systems
- Large language model applications
- Computer vision

## Summary Structure
For each relevant paper, please provide:
1. Key findings
2. Methodology
3. Results
...
```

### 2. Config File

Edit `config.yaml` to configure:
- `interests_file`: Path to your interests file
- `arxiv`: ArXiv search settings (categories, days back, max papers)
- `llm`: LLM provider and model settings
- `output`: Output directory and organization settings

Example `config.yaml`:
```yaml
interests_file: "./interests.txt"
arxiv:
  days_back: 7
  categories:
    - cs.AI
    - cs.LG
    - cs.CV
  max_papers: null
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.3
output:
  output_dir: "./papers"
  create_subdirs: true
  include_pdf: false
```

## Usage

### If installed as a package (Option 1):

```bash
# Run with default config.yaml and interests.txt
archeion_now

# Use custom config file
archeion_now --config custom_config.yaml

# Use custom config and interests files
archeion_now --config custom_config.yaml --interests custom_interests.txt

# Use custom interests file with default config
archeion_now --interests my_interests.txt

# Show help
archeion_now --help
```

### If using dependencies only (Option 2):

```bash
# Run with default config.yaml and interests.txt
python main.py

# Use custom config file
python main.py --config custom_config.yaml

# Use custom config and interests files
python main.py --config custom_config.yaml --interests custom_interests.txt

# Use custom interests file with default config
python main.py --interests my_interests.txt
```

Note: Make sure to set up environment variables (see Installation section).

## Workflow

1. **Configuration Loading**: Loads your interests and configuration settings
2. **Paper Fetching**: Fetches recent papers from ArXiv based on your criteria
3. **Relevance Filtering**: For each paper, the agent checks if it's relevant based on title/abstract
4. **Paper Summarization**: For relevant papers, the agent reads the full paper (if PDFs are enabled) and generates a structured summary
5. **Output Organization**: Summaries are saved with metadata, organized by category, and an index is generated

## Output Structure

The output directory will contain:
```
papers/
├── index.md                    # Index of all papers
├── category1/
│   ├── 1234.5678.md           # Markdown summary
│   └── 1234.5678.json         # JSON metadata
├── category2/
│   └── ...
└── pdfs/                      # PDF files (if enabled)
    └── ...
```

Each summary includes:
- Key findings and contributions
- Methodology overview
- Experimental results
- Relevance to your interests
- Potential applications
- Limitations
- Full metadata (authors, dates, categories, links)

## ArXiv Categories

Common category codes:
- `cs.AI`: Artificial Intelligence
- `cs.LG`: Machine Learning
- `cs.CV`: Computer Vision
- `cs.CL`: Computation and Language
- `stat.ML`: Machine Learning (Statistics)
- `cs.NE`: Neural and Evolutionary Computing

See [ArXiv category taxonomy](https://arxiv.org/category_taxonomy) for full list.

## Tips

1. **Start with a narrow date range**: Begin with 1-3 days back to test the system
2. **Be specific in interests**: More specific interests lead to better filtering
3. **Adjust confidence**: The agent provides confidence scores - you can modify the code to filter by confidence if needed
4. **Category selection**: Choose relevant categories to reduce processing time
5. **PDF processing**: Enable PDF processing only if you need full paper analysis (slower, more expensive)

## Web UI

Archeion Now includes a web-based user interface for managing your interests and browsing processed papers.

### Starting the Web UI

If installed as a package:
```bash
archeion_now_ui
```

Or run directly:
```bash
python web_ui.py
```

The UI will start on `http://127.0.0.1:5000` by default. You can customize the host and port:
```bash
archeion_now_ui --host 0.0.0.0 --port 8080
```

### First-Time Setup

When you first access the web UI, you'll be prompted to configure:
1. **Papers Output Directory**: Where relevant papers and summaries are saved
2. **Interests File Location**: Path to your `interests.txt` file (can be anywhere on your system)

This configuration is saved to `~/.archeion_now_ui_config.json` and persists across sessions.

### Web UI Features

- **Dashboard**: Overview of your papers, run paper processing, and quick access to all sections
- **Interests Editor**: View and edit your `interests.txt` file directly in the browser
- **Config Editor**: View and edit your `config.yaml` file with YAML validation
- **Papers Browser**: Browse all processed papers, search by title/category, and view full summaries
- **Run Processing**: Start paper processing directly from the web UI with real-time status updates

The web UI provides a convenient way to:
- Update your research interests and configuration without editing files manually
- Run paper processing with optional limits (max papers)
- Monitor processing status in real-time
- Browse and search through your processed papers
- View detailed summaries in a clean, readable format

## Customization

The codebase is modular and easy to customize:
- `config.py`: Configuration management
- `arxiv_fetcher.py`: ArXiv paper fetching
- `agent.py`: LLM agent for filtering and summarization
- `pdf_reader.py`: PDF text extraction
- `output_manager.py`: Output organization
- `main.py`: Main orchestration
- `web_ui.py`: Web interface

## License

MIT License

