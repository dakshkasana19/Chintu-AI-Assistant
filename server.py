from flask import Flask, render_template, request, jsonify
import json
import os
import subprocess
import threading
import time

app = Flask(__name__)
REGISTRY_FILE = 'app_registry.json'

# --- Data Handling Functions ---

def load_registry():
    """Loads the application registry from the JSON file."""
    if not os.path.exists(REGISTRY_FILE):
        return {}
    try:
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: {REGISTRY_FILE} is corrupted. Starting fresh.")
        return {}
    except Exception:
        return {}

def save_registry(registry):
    """Saves the application registry to the JSON file."""
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main assistant UI (index.html)."""
    return render_template('index.html')

@app.route('/register_app', methods=['POST'])
def register_app():
    """Receives application path and name from the UI and saves it."""
    data = request.get_json()
    app_path = data.get('path', '').strip()
    app_name = data.get('name', '').lower().strip()

    if not app_path or not app_name:
        return jsonify({'success': False, 'message': 'Path and name are required.'}), 400

    # Check for simple path validity (rudimentary check)
    if not (app_path.endswith('.exe') or app_path.endswith('.lnk') or os.path.isdir(app_path) or app_path.startswith("ms-settings:")):
         return jsonify({'success': False, 'message': 'Path must lead to an EXE, link, folder, or Windows setting protocol.'}), 400

    registry = load_registry()
    registry[app_name] = app_path

    try:
        save_registry(registry)
        return jsonify({'success': True, 'message': f"'{app_name}' registered successfully!"})
    except Exception as e:
        return jsonify({'success': False, 'message': f"Error saving file: {str(e)}"}), 500

def start_ui():
    """Launches the UI in the default web browser after the server starts."""
    time.sleep(1) # Wait a moment for the server to spin up
    # Opens the local URL in the default web browser
    subprocess.Popen(['start', 'http://127.0.0.1:5000'], shell=True)

def run_flask():
    """Starts the Flask server."""
    app.run(debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start the UI launch in a separate thread
    threading.Thread(target=start_ui).start()
    # Run the Flask server
    run_flask()