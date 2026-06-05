"""
serve_model.py
==============
Custom Flask-based model serving for Bank Marketing ML Model.
Uses predict_proba() to return actual confidence scores alongside predictions.

Endpoints:
    POST /invocations  - Predict with confidence scores
    GET  /health       - Health check
"""

import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = os.environ.get('MODEL_PATH', '/model_artifact/model.pkl')

print(f"Loading model from {MODEL_PATH}...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print(f"Model loaded: {type(model).__name__}")
print(f"Model has predict_proba: {hasattr(model, 'predict_proba')}")

# Class mapping (Bank Marketing: no=0, yes=1)
CLASS_NAMES = {0: 'no', 1: 'yes'}

# Feature names (20 features after preprocessing)
FEATURE_NAMES = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
    'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m', 'nr.employed'
]

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_type': type(model).__name__})


@app.route('/invocations', methods=['POST'])
def invocations():
    """
    Prediction endpoint.
    Returns predictions with actual confidence scores from predict_proba().
    """
    try:
        data = request.get_json(force=True)
        
        # Parse input
        if 'dataframe_split' in data:
            input_data = np.array(data['dataframe_split']['data'])
        elif 'instances' in data:
            input_data = np.array(data['instances'])
        elif 'data' in data:
            input_data = np.array(data['data'])
        else:
            return jsonify({'error': 'Unsupported input format. Use dataframe_split.'}), 400
        
        # Validate input shape
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        if input_data.shape[1] != 20:
            return jsonify({
                'error': f'Expected 20 features, got {input_data.shape[1]}'
            }), 400
        
        # Get predictions
        predictions = model.predict(input_data).tolist()
        
        # Get actual confidence scores using predict_proba
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data).tolist()
            confidences = [float(max(prob)) for prob in probabilities]
        else:
            probabilities = []
            confidences = [1.0] * len(predictions)
        
        return jsonify({
            'predictions': predictions,
            'probabilities': probabilities,
            'confidences': confidences
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting model server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
