"""
IPL Score Prediction - Application Entry Point
Run this file to start the Flask server
"""

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.app import app, load_models

if __name__ == "__main__":
    print("=" * 70)
    print("IPL SCORE PREDICTION")
    print("=" * 70)

    if load_models():
        print("Models loaded successfully!")
        print("\nStarting Flask server...")
        print("Open http://localhost:5000 in your browser\n")
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("ERROR: Models could not be loaded.")
        print("Please run 'python src/train.py' first.")
