"""Web UI for archeion_now using Flask."""
import os
import json
import threading
import sys
import io
import queue
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from typing import Optional, Dict, List
from datetime import datetime

# Import main processing functions
from main import main as run_main

# Get the directory where this module is located
_MODULE_DIR = Path(__file__).parent.absolute()
_TEMPLATES_DIR = _MODULE_DIR / 'templates'

app = Flask(__name__, template_folder=str(_TEMPLATES_DIR))
app.config['SECRET_KEY'] = os.urandom(24)

# Configuration storage - unified config file
CONFIG_FILE = Path.home() / '.archeion_now_config.json'

# Job status storage (in-memory for now)
job_status = {
    'running': False,
    'status': 'idle',
    'message': '',
    'error': None,
    'progress': []
}
job_lock = threading.Lock()
job_thread = None
kill_flag = threading.Event()
progress_queue = queue.Queue()


def get_default_config() -> Dict:
    """Get default configuration values (matching config.yaml structure)."""
    return {
        'papers_dir': './papers',
        'interests_file': './interests.txt',
        'relevance_threshold': 0.5,
        'arxiv': {
            'days_back': 7,
            'categories': ['cs.AI', 'cs.LG', 'cs.CV', 'stat.ML', 'physics.bio-ph', 'physics.chem-ph'],
            'max_papers': None
        },
        'llm': {
            'provider': 'openai',
            'model': 'gpt-4o-mini',
            'temperature': 0.3
        },
        'output': {
            'create_subdirs': True,
            'include_pdf': False
        }
    }


def load_config() -> Dict:
    """Load unified configuration, merging with defaults."""
    defaults = get_default_config()
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                user_config = json.load(f)
            # Merge with defaults (user config takes precedence)
            config = {**defaults, **user_config}
            # Deep merge for nested dicts
            if 'arxiv' in user_config:
                config['arxiv'] = {**defaults['arxiv'], **user_config['arxiv']}
            if 'llm' in user_config:
                config['llm'] = {**defaults['llm'], **user_config['llm']}
            if 'output' in user_config:
                config['output'] = {**defaults['output'], **user_config['output']}
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return defaults
    
    return defaults


def save_config(config: Dict):
    """Save unified configuration."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def is_configured() -> bool:
    """Check if UI is configured (has papers_dir and interests_file set)."""
    config = load_config()
    return 'papers_dir' in config and 'interests_file' in config and config.get('papers_dir') and config.get('interests_file')


# Removed get_config_file_path - we now use unified JSON config


class TeeOutput:
    """Tee output to both original stream and queue for UI streaming."""
    def __init__(self, original_stream, queue_obj):
        self.original = original_stream
        self.queue = queue_obj
        self.buffer = []
    
    def write(self, text):
        if text:
            # Add to buffer
            self.buffer.append(text)
            # Also write to original stream
            if self.original:
                self.original.write(text)
                self.original.flush()
            # Send to queue for UI
            try:
                self.queue.put_nowait(('output', text))
            except queue.Full:
                pass  # Queue full, skip
    
    def flush(self):
        if self.original:
            self.original.flush()
    
    def getvalue(self):
        return ''.join(self.buffer)


def run_processing_job(config_path: Optional[Path], interests_file: Optional[Path], max_papers: Optional[int], skip_cache: bool = False):
    """Run the paper processing in a background thread."""
    global job_status, kill_flag
    
    with job_lock:
        if job_status['running']:
            return  # Already running
    
        job_status = {
            'running': True,
            'status': 'running',
            'message': 'Starting paper processing...',
            'error': None,
            'progress': []
        }
        kill_flag.clear()
        # Clear progress queue
        while not progress_queue.empty():
            try:
                progress_queue.get_nowait()
            except queue.Empty:
                break
    
    try:
        # Capture stdout/stderr with tee to queue
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        tee_stdout = TeeOutput(old_stdout, progress_queue)
        tee_stderr = TeeOutput(old_stderr, progress_queue)
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        
        with job_lock:
            job_status['message'] = 'Initializing components...'
            progress_queue.put_nowait(('status', 'Initializing components...'))
        
        # Check kill flag before starting
        if kill_flag.is_set():
            raise KeyboardInterrupt("Process killed by user")
        
        # Run the main processing - pass None for config_path to use JSON config
        # We'll need to check kill_flag periodically in main, but for now we'll catch KeyboardInterrupt
        exit_code = 0
        try:
            exit_code = run_main(
                config_path=None,  # Use JSON config instead
                interests_file=interests_file,
                max_papers=max_papers,
                skip_cache=skip_cache
            )
        except KeyboardInterrupt:
            progress_queue.put_nowait(('status', 'Process terminated by user'))
            raise
        
        output = tee_stdout.getvalue()
        errors = tee_stderr.getvalue()
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        with job_lock:
            if kill_flag.is_set():
                job_status = {
                    'running': False,
                    'status': 'killed',
                    'message': 'Process was terminated by user',
                    'error': None,
                    'progress': []
                }
            elif exit_code == 0:
                job_status = {
                    'running': False,
                    'status': 'completed',
                    'message': 'Paper processing completed successfully!',
                    'error': None,
                    'progress': []
                }
            else:
                job_status = {
                    'running': False,
                    'status': 'error',
                    'message': 'Paper processing completed with errors.',
                    'error': errors or output,
                    'progress': []
                }
    except KeyboardInterrupt:
        sys.stdout = old_stdout if 'old_stdout' in locals() else sys.stdout
        sys.stderr = old_stderr if 'old_stderr' in locals() else sys.stderr
        
        with job_lock:
            job_status = {
                'running': False,
                'status': 'killed',
                'message': 'Process was terminated by user',
                'error': None,
                'progress': []
            }
    except Exception as e:
        sys.stdout = old_stdout if 'old_stdout' in locals() else sys.stdout
        sys.stderr = old_stderr if 'old_stderr' in locals() else sys.stderr
        
        with job_lock:
            job_status = {
                'running': False,
                'status': 'error',
                'message': 'Error during paper processing',
                'error': str(e),
                'progress': []
            }
    finally:
        kill_flag.clear()


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
        
        # Load current config and update with setup values
        config = load_config()
        config['papers_dir'] = str(papers_path)
        config['interests_file'] = str(interests_path)
        config['configured_at'] = datetime.now().isoformat()
        
        save_config(config)
        
        return redirect(url_for('dashboard'))
    
    return render_template('setup.html')


@app.route('/dashboard')
def dashboard():
    """Main dashboard."""
    if not is_configured():
        return redirect(url_for('setup'))
    
    config = load_config()
    papers_dir = Path(config['papers_dir'])
    
    # Count papers
    paper_count = 0
    if papers_dir.exists():
        # Count markdown files (excluding index.md)
        paper_count = len([f for f in papers_dir.rglob('*.md') if f.name != 'index.md'])
    
    # Get job status
    with job_lock:
        job_running = job_status['running']
        job_status_msg = job_status['status']
    
    return render_template('dashboard.html', 
                         paper_count=paper_count,
                         job_running=job_running,
                         job_status=job_status_msg)


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


@app.route('/config')
def config_page():
    """Unified config editor page."""
    if not is_configured():
        return redirect(url_for('setup'))
    return render_template('config.html')


@app.route('/api/interests', methods=['GET', 'POST'])
def interests():
    """Get or update interests file."""
    if not is_configured():
        return jsonify({'error': 'Not configured'}), 400
    
    config = load_config()
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
    
    config = load_config()
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
    
    config = load_config()
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
    
    config = load_config()
    papers_dir = Path(config['papers_dir'])
    
    # Security: ensure filename is within papers_dir
    file_path = (papers_dir / filename).resolve()
    if not str(file_path).startswith(str(papers_dir.resolve())):
        return jsonify({'error': 'Invalid path'}), 403
    
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    return send_from_directory(papers_dir, filename)


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get or update unified JSON config file."""
    if not is_configured():
        return jsonify({'error': 'Not configured'}), 400
    
    if request.method == 'GET':
        try:
            config = load_config()
            return jsonify({
                'content': json.dumps(config, indent=2),
                'path': str(CONFIG_FILE)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'POST':
        content = request.json.get('content', '')
        try:
            # Validate JSON before saving
            config_data = json.loads(content)
            
            # Validate structure
            required_keys = ['papers_dir', 'interests_file']
            for key in required_keys:
                if key not in config_data:
                    return jsonify({'error': f'Missing required key: {key}'}), 400
            
            # Save configuration
            save_config(config_data)
            
            return jsonify({'success': True})
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/api/run', methods=['POST'])
def api_run():
    """Start paper processing job."""
    if not is_configured():
        return jsonify({'error': 'Not configured'}), 400
    
    with job_lock:
        if job_status['running']:
            return jsonify({'error': 'Processing is already running'}), 400
    
    # Get parameters
    data = request.json or {}
    max_papers = data.get('max_papers')
    if max_papers:
        try:
            max_papers = int(max_papers)
        except:
            max_papers = None
    
    # Get config and interests paths
    config = load_config()
    interests_file = Path(config['interests_file'])
    
    # Start job in background thread (config_path=None means use JSON config)
    global job_thread
    job_thread = threading.Thread(
        target=run_processing_job,
        args=(None, interests_file, max_papers, False),  # skip_cache=False for normal run
        daemon=True
    )
    job_thread.start()
    
    return jsonify({'success': True, 'message': 'Processing started'})


@app.route('/api/rerun', methods=['POST'])
def api_rerun():
    """Start paper processing job with cache skipping (re-run mode)."""
    if not is_configured():
        return jsonify({'error': 'Not configured'}), 400
    
    with job_lock:
        if job_status['running']:
            return jsonify({'error': 'Processing is already running'}), 400
    
    # Get parameters
    data = request.json or {}
    max_papers = data.get('max_papers')
    if max_papers:
        try:
            max_papers = int(max_papers)
        except:
            max_papers = None
    
    # Get config and interests paths
    config = load_config()
    interests_file = Path(config['interests_file'])
    
    # Start job in background thread with skip_cache=True
    global job_thread
    job_thread = threading.Thread(
        target=run_processing_job,
        args=(None, interests_file, max_papers, True),  # skip_cache=True for re-run
        daemon=True
    )
    job_thread.start()
    
    return jsonify({'success': True, 'message': 'Re-run processing started (cache will be overwritten)'})


@app.route('/api/run/status')
def api_run_status():
    """Get status of paper processing job."""
    # Collect progress messages from queue
    progress_messages = []
    max_messages = 100  # Limit to prevent memory issues
    while len(progress_messages) < max_messages:
        try:
            msg_type, msg_content = progress_queue.get_nowait()
            progress_messages.append({
                'type': msg_type,
                'content': msg_content,
                'timestamp': datetime.now().isoformat()
            })
        except queue.Empty:
            break
    
    with job_lock:
        status = job_status.copy()
        # Add new progress messages (don't store in job_status to avoid memory bloat)
        status['progress'] = progress_messages
        return jsonify(status)


@app.route('/api/run/kill', methods=['POST'])
def api_run_kill():
    """Kill the running paper processing job."""
    global kill_flag, job_thread
    
    with job_lock:
        if not job_status['running']:
            return jsonify({'error': 'No process is currently running'}), 400
        
        # Set kill flag
        kill_flag.set()
        
        # Try to interrupt the thread (Python limitation - can't forcefully kill)
        # The process will check kill_flag and exit gracefully
        return jsonify({'success': True, 'message': 'Termination signal sent'})


@app.route('/cache')
def cache_page():
    """Cache browser page."""
    if not is_configured():
        return redirect(url_for('setup'))
    return render_template('cache.html')


@app.route('/api/cache')
def api_cache():
    """Get cache file contents."""
    if not is_configured():
        return jsonify({'error': 'Not configured'}), 400
    
    config = load_config()
    papers_dir = Path(config['papers_dir'])
    cache_file = papers_dir / "doi_cache.json"
    
    if not cache_file.exists():
        return jsonify({
            'exists': False,
            'path': str(cache_file),
            'entries': [],
            'stats': {
                'total': 0,
                'accepted': 0,
                'rejected': 0
            }
        })
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Convert to list format for easier display
        entries = []
        stats = {'total': 0, 'accepted': 0, 'rejected': 0}
        
        for doi, data in cache_data.items():
            entries.append({
                'doi': doi,
                'title': data.get('title', 'Unknown'),
                'status': data.get('status', 'unknown'),
                'relevance': data.get('relevance', 0.0),
                'confidence': data.get('confidence', 0.0),
                'impact': data.get('impact', data.get('estimated_impact', 0.0)),
                'reasoning': data.get('reasoning', ''),
                'timestamp': data.get('timestamp', '')
            })
            stats['total'] += 1
            if data.get('status') == 'accept':
                stats['accepted'] += 1
            elif data.get('status') == 'reject':
                stats['rejected'] += 1
        
        # Sort by timestamp (most recent first)
        entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'exists': True,
            'path': str(cache_file),
            'entries': entries,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

