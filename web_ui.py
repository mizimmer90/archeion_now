"""Web UI for archeion_now using Flask."""
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from typing import Optional, Dict, List
from datetime import datetime

# Note: These imports are kept for potential future use (e.g., triggering paper processing from UI)
# from config import load_config, Config
# from main import initialize_components, process_single_paper
# from arxiv_fetcher import ArxivFetcher, PaperMetadata
# from output_manager import OutputManager
# from agent import LLMAgent

# Get the directory where this module is located
_MODULE_DIR = Path(__file__).parent.absolute()
_TEMPLATES_DIR = _MODULE_DIR / 'templates'

app = Flask(__name__, template_folder=str(_TEMPLATES_DIR))
app.config['SECRET_KEY'] = os.urandom(24)

# Configuration storage
UI_CONFIG_FILE = Path.home() / '.archeion_now_ui_config.json'


def load_ui_config() -> Dict:
    """Load UI configuration."""
    if UI_CONFIG_FILE.exists():
        try:
            with open(UI_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading UI config: {e}")
    return {}


def save_ui_config(config: Dict):
    """Save UI configuration."""
    with open(UI_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def is_configured() -> bool:
    """Check if UI is configured."""
    config = load_ui_config()
    return 'papers_dir' in config and 'interests_file' in config


@app.route('/')
def index():
    """Main page - redirect to setup if not configured, otherwise show dashboard."""
    if not is_configured():
        return redirect(url_for('setup'))
    return redirect(url_for('dashboard'))


@app.route('/setup', methods=['GET', 'POST'])
def setup():
    """Initial setup page."""
    if request.method == 'POST':
        papers_dir = request.form.get('papers_dir', '').strip()
        interests_file = request.form.get('interests_file', '').strip()
        
        if not papers_dir or not interests_file:
            return render_template('setup.html', error='Both papers directory and interests file are required.')
        
        papers_path = Path(papers_dir).expanduser().resolve()
        interests_path = Path(interests_file).expanduser().resolve()
        
        # Validate paths
        if not interests_path.exists():
            return render_template('setup.html', error=f'Interests file not found: {interests_path}')
        
        # Create papers directory if it doesn't exist
        papers_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        save_ui_config({
            'papers_dir': str(papers_path),
            'interests_file': str(interests_path),
            'configured_at': datetime.now().isoformat()
        })
        
        return redirect(url_for('dashboard'))
    
    return render_template('setup.html')


@app.route('/dashboard')
def dashboard():
    """Main dashboard."""
    if not is_configured():
        return redirect(url_for('setup'))
    
    config = load_ui_config()
    papers_dir = Path(config['papers_dir'])
    
    # Count papers
    paper_count = 0
    if papers_dir.exists():
        # Count markdown files (excluding index.md)
        paper_count = len([f for f in papers_dir.rglob('*.md') if f.name != 'index.md'])
    
    return render_template('dashboard.html', paper_count=paper_count)


@app.route('/interests')
def interests_page():
    """Interests editor page."""
    if not is_configured():
        return redirect(url_for('setup'))
    return render_template('interests.html')


@app.route('/papers')
def papers_page():
    """Papers browser page."""
    if not is_configured():
        return redirect(url_for('setup'))
    return render_template('papers.html')


@app.route('/api/interests', methods=['GET', 'POST'])
def interests():
    """Get or update interests file."""
    if not is_configured():
        return jsonify({'error': 'Not configured'}), 400
    
    config = load_ui_config()
    interests_path = Path(config['interests_file'])
    
    if request.method == 'GET':
        if not interests_path.exists():
            return jsonify({'error': 'Interests file not found'}), 404
        
        try:
            with open(interests_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({'content': content})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'POST':
        content = request.json.get('content', '')
        try:
            with open(interests_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/api/papers')
def list_papers():
    """List all papers with summaries."""
    if not is_configured():
        return jsonify({'error': 'Not configured'}), 400
    
    config = load_ui_config()
    papers_dir = Path(config['papers_dir'])
    
    if not papers_dir.exists():
        return jsonify({'papers': []})
    
    papers = []
    
    # Find all markdown files (excluding index.md)
    for md_file in papers_dir.rglob('*.md'):
        if md_file.name == 'index.md':
            continue
        
        try:
            # Read markdown file
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title (first line after #)
            title = "Unknown"
            arxiv_id = md_file.stem
            for line in content.split('\n'):
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            
            # Try to read corresponding JSON file for metadata
            json_file = md_file.with_suffix('.json')
            metadata = {}
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            # Extract summary preview (first few lines of key findings)
            preview = ""
            in_section = False
            for line in content.split('\n'):
                if '## Key Findings' in line or '## Key Findings and Contributions' in line:
                    in_section = True
                    continue
                if in_section and line.strip() and not line.startswith('#'):
                    preview = line.strip()[:200]
                    break
            
            papers.append({
                'arxiv_id': arxiv_id,
                'title': title,
                'preview': preview,
                'path': str(md_file.relative_to(papers_dir)),
                'metadata': metadata,
                'category': metadata.get('summary', {}).get('category', 'uncategorized')
            })
        except Exception as e:
            print(f"Error reading paper {md_file}: {e}")
            continue
    
    # Sort by date (most recent first)
    papers.sort(key=lambda p: p.get('metadata', {}).get('paper', {}).get('published', ''), reverse=True)
    
    return jsonify({'papers': papers})


@app.route('/api/papers/<arxiv_id>')
def get_paper(arxiv_id: str):
    """Get full paper summary."""
    if not is_configured():
        return jsonify({'error': 'Not configured'}), 400
    
    config = load_ui_config()
    papers_dir = Path(config['papers_dir'])
    
    # Find the paper file
    md_file = None
    for f in papers_dir.rglob(f'{arxiv_id}.md'):
        md_file = f
        break
    
    if not md_file or not md_file.exists():
        return jsonify({'error': 'Paper not found'}), 404
    
    try:
        # Read markdown content
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to read JSON metadata
        json_file = md_file.with_suffix('.json')
        metadata = {}
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                pass
        
        return jsonify({
            'content': content,
            'metadata': metadata
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/papers/<path:filename>')
def serve_paper(filename: str):
    """Serve paper markdown files."""
    if not is_configured():
        return jsonify({'error': 'Not configured'}), 400
    
    config = load_ui_config()
    papers_dir = Path(config['papers_dir'])
    
    # Security: ensure filename is within papers_dir
    file_path = (papers_dir / filename).resolve()
    if not str(file_path).startswith(str(papers_dir.resolve())):
        return jsonify({'error': 'Invalid path'}), 403
    
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    return send_from_directory(papers_dir, filename)


@app.route('/api/config')
def get_config():
    """Get current configuration."""
    if not is_configured():
        return jsonify({'error': 'Not configured'}), 400
    
    config = load_ui_config()
    return jsonify(config)


def main():
    """Run the web UI server."""
    import argparse
    parser = argparse.ArgumentParser(description='Archeion Now Web UI')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    print(f"Starting Archeion Now Web UI at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

