"""
3.prometheus_exporter.py
========================
Custom Prometheus Exporter for Bank Marketing ML Model.
Exposes 10 custom metrics for monitoring the ML model serving endpoint.

Metrics:
1. request_count_total (Counter) — Total requests
2. request_latency_seconds (Histogram) — Request latency
3. prediction_count_total (Counter) — Predictions per class
4. error_count_total (Counter) — Total errors
5. cpu_usage_percent (Gauge) — CPU usage
6. memory_usage_bytes (Gauge) — Memory usage
7. prediction_distribution (Histogram) — Prediction confidence distribution
8. response_size_bytes (Histogram) — Response size
9. active_connections (Gauge) — Active connections
10. model_drift_score (Gauge) — Model drift score (PSI-based)

Usage:
    python 3.prometheus_exporter.py

    Metrics endpoint: http://localhost:8000/metrics
    Prediction proxy: http://localhost:8000/predict
"""

import os
import time
import json
import threading
import numpy as np
import psutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import requests

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
)

# ============================================================
# CONFIGURATION
# ============================================================
EXPORTER_PORT = 8000
MODEL_SERVING_URL = os.environ.get('MODEL_SERVING_URL', 'http://localhost:5001/invocations')
METRICS_UPDATE_INTERVAL = 5  # seconds

# Prometheus Registry
registry = CollectorRegistry()

# ============================================================
# METRIC DEFINITIONS (10 metrics)
# ============================================================

# 1. Total Request Count
request_count = Counter(
    'ml_request_count_total',
    'Total number of prediction requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

# 2. Request Latency
request_latency = Histogram(
    'ml_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

# 3. Prediction Count per Class
prediction_count = Counter(
    'ml_prediction_count_total',
    'Total predictions per class',
    ['predicted_class'],
    registry=registry
)

# 4. Error Count
error_count = Counter(
    'ml_error_count_total',
    'Total number of errors',
    ['error_type'],
    registry=registry
)

# 5. CPU Usage
cpu_usage = Gauge(
    'ml_cpu_usage_percent',
    'Current CPU usage percentage',
    registry=registry
)

# 6. Memory Usage
memory_usage = Gauge(
    'ml_memory_usage_bytes',
    'Current memory usage in bytes',
    registry=registry
)

# 7. Prediction Confidence Distribution
prediction_confidence = Histogram(
    'ml_prediction_confidence',
    'Distribution of prediction confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry
)

# 8. Response Size
response_size = Histogram(
    'ml_response_size_bytes',
    'Size of responses in bytes',
    buckets=[100, 500, 1000, 5000, 10000, 50000],
    registry=registry
)

# 9. Active Connections
active_connections = Gauge(
    'ml_active_connections',
    'Number of currently active connections',
    registry=registry
)

# 10. Model Drift Score
model_drift_score = Gauge(
    'ml_model_drift_score',
    'Model drift score based on PSI (Population Stability Index)',
    registry=registry
)

# ============================================================
# TRACKING STATE
# ============================================================
# For drift calculation
reference_distribution = np.array([0.887, 0.113])  # no, yes from training
recent_predictions = []
DRIFT_WINDOW_SIZE = 100


def calculate_psi(expected, actual, epsilon=1e-4):
    """Calculate Population Stability Index (PSI)."""
    expected = np.clip(expected, epsilon, 1 - epsilon)
    actual = np.clip(actual, epsilon, 1 - epsilon)
    psi = np.sum((actual - expected) * np.log(actual / expected))
    return psi


def update_system_metrics():
    """Update CPU and memory metrics periodically."""
    while True:
        try:
            cpu_usage.set(psutil.cpu_percent(interval=1))
            memory_info = psutil.Process().memory_info()
            memory_usage.set(memory_info.rss)
            
            # Update drift score
            if len(recent_predictions) >= DRIFT_WINDOW_SIZE:
                window = recent_predictions[-DRIFT_WINDOW_SIZE:]
                unique, counts = np.unique(window, return_counts=True)
                actual_dist = np.zeros(2)
                for u, c in zip(unique, counts):
                    if u < 2:
                        actual_dist[u] = c / len(window)
                psi = calculate_psi(reference_distribution, actual_dist)
                model_drift_score.set(psi)
            
        except Exception as e:
            print(f"Error updating system metrics: {e}")
        
        time.sleep(METRICS_UPDATE_INTERVAL)


# ============================================================
# HTTP HANDLER
# ============================================================
class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoint and prediction proxy."""
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        
        if parsed.path == '/metrics':
            self._serve_metrics()
        elif parsed.path == '/health':
            self._serve_health()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def do_POST(self):
        """Handle POST requests (prediction proxy)."""
        parsed = urlparse(self.path)
        
        if parsed.path == '/predict':
            self._handle_prediction()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def _serve_metrics(self):
        """Serve Prometheus metrics."""
        self.send_response(200)
        self.send_header('Content-Type', CONTENT_TYPE_LATEST)
        self.end_headers()
        self.wfile.write(generate_latest(registry))
    
    def _serve_health(self):
        """Serve health check."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'healthy'}).encode())
    
    def _handle_prediction(self):
        """Proxy prediction requests to model serving and record metrics."""
        active_connections.inc()
        start_time = time.time()
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            # Forward to model serving endpoint
            response = requests.post(
                MODEL_SERVING_URL,
                data=body,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            latency = time.time() - start_time
            
            # Record metrics
            request_count.labels(
                method='POST', endpoint='/predict', status=str(response.status_code)
            ).inc()
            request_latency.labels(endpoint='/predict').observe(latency)
            response_size.observe(len(response.content))
            
            if response.status_code == 200:
                # Parse predictions and actual confidence from model
                try:
                    result = response.json()
                    predictions = result.get('predictions', [])
                    confidences = result.get('confidences', [])
                    
                    class_names = {0: 'no', 1: 'yes'}
                    for idx, pred in enumerate(predictions):
                        pred_class = pred if isinstance(pred, int) else int(pred)
                        prediction_count.labels(
                            predicted_class=class_names.get(pred_class, str(pred_class))
                        ).inc()
                        recent_predictions.append(pred_class)
                        
                        # Use ACTUAL confidence from model's predict_proba()
                        if idx < len(confidences):
                            prediction_confidence.observe(confidences[idx])
                        
                except Exception:
                    pass
            else:
                error_count.labels(error_type='model_error').inc()
            
            # Send response back
            self.send_response(response.status_code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response.content)
            
        except requests.exceptions.ConnectionError:
            error_count.labels(error_type='connection_error').inc()
            request_count.labels(method='POST', endpoint='/predict', status='503').inc()
            
            self.send_response(503)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': 'Model serving endpoint not available',
                'detail': f'Could not connect to {MODEL_SERVING_URL}'
            }).encode())
            
        except Exception as e:
            error_count.labels(error_type='internal_error').inc()
            request_count.labels(method='POST', endpoint='/predict', status='500').inc()
            
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
            
        finally:
            active_connections.dec()


# ============================================================
# MAIN
# ============================================================
def main():
    """Start the Prometheus exporter server."""
    print("=" * 60)
    print("ML MODEL - PROMETHEUS EXPORTER")
    print("=" * 60)
    print(f"Metrics endpoint: http://localhost:{EXPORTER_PORT}/metrics")
    print(f"Prediction proxy: http://localhost:{EXPORTER_PORT}/predict")
    print(f"Health check:     http://localhost:{EXPORTER_PORT}/health")
    print(f"Model serving:    {MODEL_SERVING_URL}")
    print("=" * 60)
    
    # Start system metrics updater thread
    metrics_thread = threading.Thread(target=update_system_metrics, daemon=True)
    metrics_thread.start()
    print("✅ System metrics updater started")
    
    # Start HTTP server
    server = HTTPServer(('0.0.0.0', EXPORTER_PORT), MetricsHandler)
    print(f"✅ Exporter server started on port {EXPORTER_PORT}")
    print("\nPress Ctrl+C to stop...")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down exporter...")
        server.shutdown()


if __name__ == "__main__":
    main()
