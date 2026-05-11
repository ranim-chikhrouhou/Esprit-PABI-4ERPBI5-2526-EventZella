# -*- coding: utf-8 -*-
"""
EventZilla MLOps - Monitoring Simulation Script
Week S13: Simulate production scenarios for testing monitoring system

This script simulates:
1. High traffic scenario
2. API errors scenario
3. Model drift scenario
4. Data quality issues
"""
import requests
import time
import random
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configuration
API_BASE_URL = "http://localhost:8000"
LOGIN_CREDENTIALS = {
    "login": "naima_sarraj",
    "password": "Naima@Finance2025!"
}


def get_auth_token():
    """Get JWT authentication token"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            json=LOGIN_CREDENTIALS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return None


def make_prediction_request(token, endpoint, data, expect_error=False):
    """Make a prediction request"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        if endpoint == "/predict/timeseries":
            response = requests.get(
                f"{API_BASE_URL}{endpoint}",
                headers=headers,
                params=data,
                timeout=5
            )
        else:
            response = requests.post(
                f"{API_BASE_URL}{endpoint}",
                headers=headers,
                json=data,
                timeout=5
            )
        
        if expect_error:
            return {"status": "error", "code": response.status_code}
        
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    
    except requests.exceptions.Timeout:
        return {"status": "timeout"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 1: HIGH TRAFFIC
# ═══════════════════════════════════════════════════════════════════

def scenario_high_traffic(token, duration_seconds=60, requests_per_second=10):
    """
    Simulate high traffic scenario
    
    Expected monitoring impact:
    - Request rate increases
    - Latency may increase
    - System resources (CPU, memory) increase
    """
    print("\n" + "="*70)
    print("🚀 SCENARIO 1: HIGH TRAFFIC")
    print("="*70)
    print(f"Duration: {duration_seconds}s | Target: {requests_per_second} req/s")
    print("Expected: Increased latency, higher CPU/memory usage")
    print("-"*70)
    
    endpoints = [
        ("/predict/classification", {
            "id_date": 1, "id_event": 42, "id_servicecategory": 3,
            "id_benchmark": 2, "id_provider": 7, "final_price": 1500,
            "service_price": 1200, "benchmark_avg_price": 1300,
            "event_budget": 2000, "cal_month": 4, "cal_year": 2024, "quarter": 2
        }),
        ("/predict/regression", {
            "id_date": 1, "id_event": 42, "id_servicecategory": 3,
            "id_benchmark": 2, "id_provider": 7, "service_price": 1200,
            "benchmark_avg_price": 1300, "event_budget": 2000,
            "cal_month": 4, "cal_year": 2024, "quarter": 2, "commission_margin": 150
        }),
        ("/predict/timeseries", {"horizon": 3}),
    ]
    
    start_time = time.time()
    total_requests = 0
    successful_requests = 0
    failed_requests = 0
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        while time.time() - start_time < duration_seconds:
            futures = []
            
            # Submit batch of requests
            for _ in range(requests_per_second):
                endpoint, data = random.choice(endpoints)
                future = executor.submit(make_prediction_request, token, endpoint, data)
                futures.append(future)
            
            # Wait for batch to complete
            for future in as_completed(futures):
                result = future.result()
                total_requests += 1
                if result["status"] == "success":
                    successful_requests += 1
                else:
                    failed_requests += 1
            
            # Progress update
            elapsed = time.time() - start_time
            current_rate = total_requests / elapsed if elapsed > 0 else 0
            print(f"⏱️  {elapsed:.1f}s | Requests: {total_requests} | Rate: {current_rate:.1f} req/s | Success: {successful_requests} | Failed: {failed_requests}", end="\r")
            
            # Sleep to maintain target rate
            time.sleep(0.1)
    
    print(f"\n✅ Scenario completed!")
    print(f"   Total requests: {total_requests}")
    print(f"   Successful: {successful_requests} ({successful_requests/total_requests*100:.1f}%)")
    print(f"   Failed: {failed_requests} ({failed_requests/total_requests*100:.1f}%)")
    print(f"   Average rate: {total_requests/duration_seconds:.1f} req/s")


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 2: API ERRORS
# ═══════════════════════════════════════════════════════════════════

def scenario_api_errors(token, duration_seconds=30, error_rate=0.5):
    """
    Simulate API errors scenario
    
    Expected monitoring impact:
    - Error rate increases
    - Alerts triggered for high error rate
    """
    print("\n" + "="*70)
    print("💥 SCENARIO 2: API ERRORS")
    print("="*70)
    print(f"Duration: {duration_seconds}s | Error rate: {error_rate*100:.0f}%")
    print("Expected: High error rate alerts, increased error counter")
    print("-"*70)
    
    start_time = time.time()
    total_requests = 0
    error_requests = 0
    
    while time.time() - start_time < duration_seconds:
        # Randomly decide if this request should error
        should_error = random.random() < error_rate
        
        if should_error:
            # Send invalid data to trigger errors
            invalid_data = {
                "id_date": "invalid",  # Wrong type
                "id_event": None,  # Missing required field
            }
            result = make_prediction_request(
                token,
                "/predict/classification",
                invalid_data,
                expect_error=True
            )
            error_requests += 1
        else:
            # Send valid request
            valid_data = {
                "id_date": 1, "id_event": 42, "id_servicecategory": 3,
                "id_benchmark": 2, "id_provider": 7, "final_price": 1500,
                "service_price": 1200, "benchmark_avg_price": 1300,
                "event_budget": 2000, "cal_month": 4, "cal_year": 2024, "quarter": 2
            }
            result = make_prediction_request(token, "/predict/classification", valid_data)
        
        total_requests += 1
        
        # Progress update
        elapsed = time.time() - start_time
        current_error_rate = error_requests / total_requests if total_requests > 0 else 0
        print(f"⏱️  {elapsed:.1f}s | Requests: {total_requests} | Errors: {error_requests} | Error rate: {current_error_rate*100:.1f}%", end="\r")
        
        time.sleep(0.1)
    
    print(f"\n✅ Scenario completed!")
    print(f"   Total requests: {total_requests}")
    print(f"   Errors: {error_requests} ({error_requests/total_requests*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 3: MODEL DRIFT
# ═══════════════════════════════════════════════════════════════════

def scenario_model_drift(token, duration_seconds=45):
    """
    Simulate model drift scenario by sending data with shifted distribution
    
    Expected monitoring impact:
    - Data drift detection triggered
    - Feature distribution changes
    - Potential accuracy degradation alerts
    """
    print("\n" + "="*70)
    print("📉 SCENARIO 3: MODEL DRIFT (Data Distribution Shift)")
    print("="*70)
    print(f"Duration: {duration_seconds}s")
    print("Expected: Data drift alerts, distribution shift detection")
    print("-"*70)
    
    start_time = time.time()
    total_requests = 0
    
    # Phase 1: Normal data (first 15 seconds)
    print("\n📊 Phase 1: Normal data distribution...")
    while time.time() - start_time < duration_seconds / 3:
        data = {
            "id_date": random.randint(1, 100),
            "id_event": random.randint(1, 50),
            "id_servicecategory": random.randint(1, 10),
            "id_benchmark": random.randint(1, 5),
            "id_provider": random.randint(1, 20),
            "service_price": random.uniform(500, 2000),
            "benchmark_avg_price": random.uniform(600, 1800),
            "event_budget": random.uniform(1000, 5000),
            "cal_month": random.randint(1, 12),
            "cal_year": 2024,
            "quarter": random.randint(1, 4),
            "commission_margin": random.uniform(50, 300)
        }
        
        make_prediction_request(token, "/predict/regression", data)
        total_requests += 1
        print(f"⏱️  Requests: {total_requests} | Phase: Normal", end="\r")
        time.sleep(0.2)
    
    # Phase 2: Shifted data (next 15 seconds)
    print(f"\n📊 Phase 2: Shifted data distribution (drift simulation)...")
    while time.time() - start_time < 2 * duration_seconds / 3:
        # Shift all numerical features by 50%
        data = {
            "id_date": random.randint(50, 150),  # Shifted up
            "id_event": random.randint(25, 75),  # Shifted up
            "id_servicecategory": random.randint(5, 15),  # Shifted up
            "id_benchmark": random.randint(3, 8),  # Shifted up
            "id_provider": random.randint(10, 30),  # Shifted up
            "service_price": random.uniform(1500, 4000),  # Shifted up significantly
            "benchmark_avg_price": random.uniform(1600, 3800),  # Shifted up
            "event_budget": random.uniform(3000, 8000),  # Shifted up
            "cal_month": random.randint(1, 12),
            "cal_year": 2024,
            "quarter": random.randint(1, 4),
            "commission_margin": random.uniform(200, 600)  # Shifted up
        }
        
        make_prediction_request(token, "/predict/regression", data)
        total_requests += 1
        print(f"⏱️  Requests: {total_requests} | Phase: DRIFT (shifted +50%)", end="\r")
        time.sleep(0.2)
    
    # Phase 3: Return to normal (last 15 seconds)
    print(f"\n📊 Phase 3: Return to normal distribution...")
    while time.time() - start_time < duration_seconds:
        data = {
            "id_date": random.randint(1, 100),
            "id_event": random.randint(1, 50),
            "id_servicecategory": random.randint(1, 10),
            "id_benchmark": random.randint(1, 5),
            "id_provider": random.randint(1, 20),
            "service_price": random.uniform(500, 2000),
            "benchmark_avg_price": random.uniform(600, 1800),
            "event_budget": random.uniform(1000, 5000),
            "cal_month": random.randint(1, 12),
            "cal_year": 2024,
            "quarter": random.randint(1, 4),
            "commission_margin": random.uniform(50, 300)
        }
        
        make_prediction_request(token, "/predict/regression", data)
        total_requests += 1
        print(f"⏱️  Requests: {total_requests} | Phase: Normal (recovered)", end="\r")
        time.sleep(0.2)
    
    print(f"\n✅ Scenario completed!")
    print(f"   Total requests: {total_requests}")
    print(f"   Drift simulation: 3 phases (normal → drift → normal)")


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    """Run all simulation scenarios"""
    print("\n" + "="*70)
    print("🎯 EventZilla MLOps - Monitoring Simulation")
    print("Week S13: Production Scenario Testing")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_BASE_URL}")
    print("="*70)
    
    # Authenticate
    print("\n🔐 Authenticating...")
    token = get_auth_token()
    if not token:
        print("❌ Failed to authenticate. Exiting.")
        return
    print("✅ Authentication successful!")
    
    # Wait for user confirmation
    print("\n" + "="*70)
    print("⚠️  IMPORTANT: Make sure Prometheus and Grafana are running!")
    print("   - Prometheus: http://localhost:9090")
    print("   - Grafana: http://localhost:3000")
    print("="*70)
    input("\nPress ENTER to start simulations...")
    
    try:
        # Scenario 1: High Traffic
        scenario_high_traffic(token, duration_seconds=60, requests_per_second=10)
        print("\n⏸️  Waiting 10 seconds before next scenario...")
        time.sleep(10)
        
        # Scenario 2: API Errors
        scenario_api_errors(token, duration_seconds=30, error_rate=0.3)
        print("\n⏸️  Waiting 10 seconds before next scenario...")
        time.sleep(10)
        
        # Scenario 3: Model Drift
        scenario_model_drift(token, duration_seconds=45)
        
        # Final summary
        print("\n" + "="*70)
        print("✅ ALL SCENARIOS COMPLETED!")
        print("="*70)
        print("\n📊 Next steps:")
        print("   1. Open Grafana: http://localhost:3000")
        print("   2. View 'EventZilla MLOps - Production Monitoring' dashboard")
        print("   3. Observe the metrics and alerts from the simulations")
        print("   4. Check Prometheus alerts: http://localhost:9090/alerts")
        print("\n" + "="*70)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during simulation: {e}")


if __name__ == "__main__":
    main()
