import json
import urllib.request
import urllib.error
import time
import random

# API endpoint URL
API_URL = "http://127.0.0.1:8000"

def generate_simulated_emg_window(num_samples: int = 200) -> list[list[float]]:
    """
    Simulates a 200ms window of raw EMG data (11 channels, sampled at 1000 Hz).
    Includes simulated electrical noise and muscle activation bursts.
    """
    window = []
    # Set muscle activation multipliers (some muscles more active than others)
    multipliers = [random.uniform(0.05, 0.8) for _ in range(11)]
    
    for i in range(num_samples):
        sample = []
        for ch in range(11):
            # Baseline high-frequency noise + occasional activation burst
            noise = random.uniform(-0.02, 0.02)
            burst = 0.3 * multipliers[ch] * (1.0 + random.uniform(-0.1, 0.1))
            val = noise + burst
            sample.append(val)
        window.append(sample)
    return window

def check_health() -> bool:
    """Queries the FastAPI server health endpoint."""
    url = f"{API_URL}/health"
    print(f"Checking service health at {url}...")
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"Health status: {data['status']} - {data['details']}")
                return data['status'] == "OK"
    except Exception as e:
        print(f"Health check failed (server might not be running yet): {e}")
    return False

def test_prediction_endpoint():
    """Sends simulated EMG windows to /predict and evaluates performance."""
    url = f"{API_URL}/predict"
    print(f"\n--- Testing Inference Endpoint at {url} ---")
    
    # Run 5 test queries to evaluate stability and average latency
    num_tests = 5
    latencies = []
    predicted_angles = []
    
    for i in range(num_tests):
        print(f"Query {i+1}/{num_tests}: Generating raw EMG window...")
        emg_window = generate_simulated_emg_window(200)
        
        payload = {"emg_window": emg_window}
        data_bytes = json.dumps(payload).encode('utf-8')
        
        req = urllib.request.Request(
            url, 
            data=data_bytes, 
            headers={'Content-Type': 'application/json'}
        )
        
        start_client_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                end_client_time = time.perf_counter()
                
                if response.status == 200:
                    resp_data = json.loads(response.read().decode())
                    server_latency = resp_data['latency_ms']
                    client_latency = (end_client_time - start_client_time) * 1000.0
                    predicted_angle = resp_data['predicted_knee_angle']
                    features = resp_data['features_extracted']
                    
                    print(f"  -> Prediction: {predicted_angle:.4f}°")
                    print(f"  -> Server Latency: {server_latency:.2f} ms")
                    print(f"  -> Client Roundtrip Latency: {client_latency:.2f} ms")
                    print(f"  -> Extracted features sample: VM_rms={features['vastusmedialis_rms']:.4f}, BF_wl={features['bicepsfemoris_wl']:.4f}")
                    
                    latencies.append(client_latency)
                    predicted_angles.append(predicted_angle)
                else:
                    print(f"  -> Query failed: HTTP {response.status}")
        except urllib.error.HTTPError as e:
            print(f"  -> HTTP Error: {e.code} - {e.read().decode()}")
        except Exception as e:
            print(f"  -> Network/Client Error: {e}")
            
        time.sleep(0.5) # Gap between queries
        
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print("\n--- Performance Summary ---")
        print(f"Successful Runs: {len(latencies)}/{num_tests}")
        print(f"Average Roundtrip Inference Latency: {avg_latency:.2f} ms")
        print(f"Range of Predicted Knee Angles: [{min(predicted_angles):.2f}°, {max(predicted_angles):.2f}°]")
        
        if avg_latency < 10.0:
            print("Speed check: EXCELLENT (meets <10ms requirement for real-time exoskeleton loop)")
        elif avg_latency < 50.0:
            print("Speed check: GOOD (suitable for slow/interactive movement loop)")
        else:
            print("Speed check: WARNING (latency might be too high for sub-20ms exoskeleton control)")
    else:
        print("\nVerification failed: No queries completed successfully.")

def main():
    print("EMG Exoskeleton Deployment Verification Client")
    print("=============================================")
    if check_health():
        test_prediction_endpoint()
    else:
        print("\nService is not ready. Please make sure uvicorn is running on port 8000 and the model has been trained.")

if __name__ == '__main__':
    main()
