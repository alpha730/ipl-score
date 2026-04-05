"""
IPL Score Prediction - Flask Backend API
Uses core parameters: runs, overs, wickets
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import json
import os

app = Flask(__name__,
            static_folder='../static',
            template_folder='../templates')
CORS(app)

models = None
preprocessing_params = None
evaluation_results = None


def load_models():
    """Load trained models and preprocessing parameters"""
    global models, preprocessing_params, evaluation_results

    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_path = os.path.join(base_dir, 'models', 'trained_models.pkl')
        params_path = os.path.join(base_dir, 'models', 'preprocessing_params.pkl')
        results_path = os.path.join(base_dir, 'models', 'evaluation_results.json')

        with open(models_path, 'rb') as f:
            models = pickle.load(f)

        with open(params_path, 'rb') as f:
            preprocessing_params = pickle.load(f)

        with open(results_path, 'r') as f:
            evaluation_results = json.load(f)

        print("Models loaded successfully!")
        return True
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please run 'python src/train.py' first to train the models.")
        return False


@app.route('/')
def index():
    return send_from_directory('../templates', 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('../static', filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict match score based on core parameters

    Expected JSON body:
    {
        "batting_team": "Mumbai Indians",
        "bowling_team": "Chennai Super Kings",
        "city": "Mumbai",
        "runs": 85,
        "overs": 10.0,
        "wickets": 3,
        "model": "Linear Regression"
    }
    """
    if models is None:
        return jsonify({'error': 'Models not loaded'}), 500

    data = request.get_json()

    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    required_fields = [
        'batting_team', 'bowling_team', 'city',
        'runs', 'overs', 'wickets'
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    model_name = data.get('model', evaluation_results['best_model'] if evaluation_results else 'Linear Regression')

    if model_name not in models:
        model_name = 'Linear Regression'

    encoders = preprocessing_params['encoders']
    bat_encoded = encoders.get('batting_team', {}).get(data['batting_team'], 0)
    bowl_encoded = encoders.get('bowling_team', {}).get(data['bowling_team'], 0)
    city_encoded = encoders.get('city', {}).get(data['city'], 0)

    features = np.array([
        bat_encoded, bowl_encoded, city_encoded,
        float(data['runs']),
        float(data['overs']),
        float(data['wickets'])
    ]).reshape(1, -1)

    mean = np.array(preprocessing_params['mean'])
    std = np.array(preprocessing_params['std'])
    features_normalized = (features - mean) / std

    model = models[model_name]['model']
    prediction = model.predict(features_normalized)
    predicted_score = max(0, prediction[0])

    if evaluation_results and model_name in evaluation_results['model_comparison']:
        rmse = evaluation_results['model_comparison'][model_name]['rmse']
        lower_bound = max(0, predicted_score - rmse)
        upper_bound = predicted_score + rmse
    else:
        rmse = 15
        lower_bound = max(0, predicted_score - rmse)
        upper_bound = predicted_score + rmse

    return jsonify({
        'success': True,
        'prediction': {
            'predicted_score': round(predicted_score, 1),
            'confidence_range': {
                'lower': round(lower_bound, 1),
                'upper': round(upper_bound, 1)
            },
            'model_used': model_name
        },
        'input': {
            'batting_team': data['batting_team'],
            'bowling_team': data['bowling_team'],
            'city': data['city'],
            'runs': data['runs'],
            'overs': data['overs'],
            'wickets': data['wickets']
        }
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    if models is None:
        return jsonify({'error': 'Models not loaded'}), 500

    model_info = {}
    for name in models.keys():
        model_info[name] = {'available': True}
        if evaluation_results and name in evaluation_results['model_comparison']:
            model_info[name].update(evaluation_results['model_comparison'][name])

    return jsonify({
        'success': True,
        'models': model_info,
        'best_model': evaluation_results['best_model'] if evaluation_results else None
    })


@app.route('/api/comparison', methods=['GET'])
def get_model_comparison():
    if evaluation_results is None:
        return jsonify({'error': 'Evaluation results not loaded'}), 500

    return jsonify({
        'success': True,
        'comparison': evaluation_results['model_comparison'],
        'best_model': {
            'name': evaluation_results['best_model'],
            'r2': evaluation_results['best_r2']
        }
    })


@app.route('/api/teams', methods=['GET'])
def get_teams():
    if preprocessing_params is None:
        return jsonify({'error': 'Preprocessing params not loaded'}), 500

    encoders = preprocessing_params['encoders']
    teams = list(encoders.get('batting_team', {}).keys())

    return jsonify({
        'success': True,
        'teams': sorted(teams)
    })


@app.route('/api/cities', methods=['GET'])
def get_cities():
    if preprocessing_params is None:
        return jsonify({'error': 'Preprocessing params not loaded'}), 500

    encoders = preprocessing_params['encoders']
    cities = list(encoders.get('city', {}).keys())

    return jsonify({
        'success': True,
        'cities': sorted(cities)
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if models is not None else 'models_not_loaded',
        'models_loaded': models is not None,
        'preprocessing_loaded': preprocessing_params is not None,
        'evaluation_loaded': evaluation_results is not None
    })


if __name__ == '__main__':
    print("=" * 70)
    print("IPL SCORE PREDICTION - FLASK API SERVER")
    print("=" * 70)
    print("Using core parameters: RUNS | OVERS | WICKETS")

    if load_models():
        print("\nStarting Flask server...")
        print("Open http://localhost:5000 in your browser")
    else:
        print("\nWARNING: Models not loaded.")

    app.run(debug=True, host='0.0.0.0', port=5000)
