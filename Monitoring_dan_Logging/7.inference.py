"""
7.inference.py
==============
Inference script to send prediction requests to the ML model serving endpoint.
Uses REAL data from the preprocessed Bank Marketing dataset (not dummy/random values).

Usage:
    python 7.inference.py
    python 7.inference.py --url http://localhost:8000/predict --count 100
"""

import os
import sys
import json
import time
import random
import argparse
import requests
import csv

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_URL = os.environ.get('PREDICT_URL', 'http://localhost:8000/predict')
TARGET_NAMES = {0: 'no', 1: 'yes'}

# Path to the preprocessed dataset (real data, not dummy)
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bankmarketing_test.csv')

# Feature names (20 features after preprocessing)
FEATURE_NAMES = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
    'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m', 'nr.employed'
]


def load_real_dataset(path):
    """
    Load real preprocessed Bank Marketing dataset from CSV.
    Returns list of feature vectors (without the label column).
    """
    data = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = [float(row[col]) for col in FEATURE_NAMES]
            data.append(sample)
    print(f"  Loaded {len(data)} real samples from {os.path.basename(path)}")
    return data


def get_real_batch(dataset, n=5):
    """
    Get a random batch of REAL samples from the preprocessed dataset.
    """
    return random.sample(dataset, min(n, len(dataset)))


def send_prediction(url, samples):
    """
    Send prediction request to the model serving endpoint.
    """
    payload = {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": samples
        }
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        return response
    except requests.exceptions.ConnectionError:
        print(f"  Cannot connect to {url}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_normal_traffic(url, dataset, count=50, batch_size=5, delay=0.5):
    """
    Send normal traffic using REAL data from the preprocessed dataset.
    """
    print(f"\n[INFO] Sending {count} normal requests (batch_size={batch_size})...")
    print(f"[INFO] Using REAL data from preprocessed dataset ({len(dataset)} samples available)")
    
    success = 0
    errors = 0
    total_latency = 0
    predictions_count = {name: 0 for name in TARGET_NAMES.values()}
    
    for i in range(count):
        samples = get_real_batch(dataset, batch_size)
        
        start = time.time()
        response = send_prediction(url, samples)
        latency = time.time() - start
        total_latency += latency
        
        if response and response.status_code == 200:
            success += 1
            try:
                result = response.json()
                preds = result.get('predictions', [])
                for p in preds:
                    pred_class = int(p) if isinstance(p, (int, float)) else p
                    class_name = TARGET_NAMES.get(pred_class, str(pred_class))
                    predictions_count[class_name] = predictions_count.get(class_name, 0) + 1
            except Exception:
                pass
            
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{count}] OK Status: {response.status_code}, Latency: {latency:.3f}s")
        else:
            errors += 1
            status = response.status_code if response else "N/A"
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{count}] FAIL Status: {status}, Latency: {latency:.3f}s")
        
        time.sleep(delay + random.uniform(0, 0.3))
    
    print(f"\n[SUMMARY] Normal Traffic:")
    print(f"  Requests: {count}")
    print(f"  Success: {success}")
    print(f"  Errors: {errors}")
    print(f"  Avg Latency: {total_latency/max(count,1):.3f}s")
    print(f"  Predictions: {predictions_count}")


def run_burst_traffic(url, dataset, burst_size=20, batch_size=10):
    """
    Simulate burst traffic using REAL data.
    """
    print(f"\n[INFO] Sending burst of {burst_size} rapid requests (REAL data)...")
    
    for i in range(burst_size):
        samples = get_real_batch(dataset, batch_size)
        response = send_prediction(url, samples)
        status = response.status_code if response else "N/A"
        print(f"  Burst [{i+1}/{burst_size}] Status: {status}")
        time.sleep(0.05)
    
    print("  Burst completed")


def run_error_traffic(url, count=10):
    """
    Simulate error scenarios by sending malformed requests.
    """
    print(f"\n[INFO] Sending {count} error-inducing requests...")
    
    error_payloads = [
        {},
        {"invalid": "data"},
        {"dataframe_split": {"columns": ["wrong"], "data": [[1]]}},
        "not json",
    ]
    
    for i in range(count):
        payload = random.choice(error_payloads)
        
        try:
            response = requests.post(
                url,
                json=payload if isinstance(payload, dict) else None,
                data=payload if isinstance(payload, str) else None,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            print(f"  Error [{i+1}/{count}] Status: {response.status_code}")
        except Exception as e:
            print(f"  Error [{i+1}/{count}] Exception: {type(e).__name__}")
        
        time.sleep(0.2)
    
    print("  Error traffic completed")


def main():
    parser = argparse.ArgumentParser(description='Bank Marketing ML Model Inference (REAL data)')
    parser.add_argument('--url', default=DEFAULT_URL, help='Prediction endpoint URL')
    parser.add_argument('--count', type=int, default=50, help='Number of normal requests')
    parser.add_argument('--batch-size', type=int, default=5, help='Samples per request')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between requests')
    parser.add_argument('--scenario', choices=['normal', 'burst', 'error', 'all'], 
                        default='all', help='Traffic scenario')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BANK MARKETING MODEL - INFERENCE CLIENT")
    print("=" * 60)
    print(f"Endpoint: {args.url}")
    print(f"Scenario: {args.scenario}")
    print(f"Data source: {DATASET_PATH}")
    
    # Load REAL dataset
    print(f"\n[INFO] Loading real preprocessed dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"  ERROR: Dataset not found at {DATASET_PATH}")
        print(f"  Please ensure bankmarketing_test.csv is in the same directory.")
        sys.exit(1)
    
    dataset = load_real_dataset(DATASET_PATH)
    
    # Check connectivity
    print(f"\n[INFO] Checking endpoint connectivity...")
    try:
        resp = requests.get(args.url.replace('/predict', '/health'), timeout=5)
        print(f"  Health check: {resp.status_code}")
    except Exception:
        print(f"  WARNING: Health check failed")
        print(f"  Continuing anyway...")
    
    if args.scenario in ('normal', 'all'):
        run_normal_traffic(args.url, dataset, args.count, args.batch_size, args.delay)
    
    if args.scenario in ('burst', 'all'):
        run_burst_traffic(args.url, dataset)
    
    if args.scenario in ('error', 'all'):
        run_error_traffic(args.url)
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)
    print("\nCheck metrics at: http://localhost:8000/metrics")
    print("Check Prometheus at: http://localhost:9090")
    print("Check Grafana at: http://localhost:3000")


if __name__ == "__main__":
    main()
