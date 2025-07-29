from flask import Flask, jsonify
import numpy as np
from supabase import create_client
import time
import datetime
import logging

app = Flask(__name__)
SUPABASE_URL = "https://zckiwulodojgcfwyjrcx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTMxNDQ3NDQsImV4cCI6MjA2ODcyMDc0NH0.glM0KT1bfV_5BgQbOLS5JxhjTjJR5sLNn7nuoNpBtBc"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

MIN_GREEN = 30
MAX_GREEN = 80

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

def process_traffic_data():
    response = supabase.table("traffic_observations")\
        .select("*")\
        .order("created_at", desc=True)\
        .limit(10)\
        .execute()
    
    if not response.data:
        return

    for record in response.data:
        try:
            tls_id = record['tls_id']
            obs = record['obs']
            lane_data = {lane['id']: lane for lane in obs['lanes']}
            record_id = record['id']

            # Compute reward
            reward, _, _ = compute_reward(lane_data)
            
            # Calculate delta_t
            _, delta_t, _ = calculate_delta_t(reward)

            # Load existing phases
            phases = supabase.table("phases")\
                .select("phase_idx, state, duration")\
                .eq("tls_id", tls_id)\
                .execute()
            if not phases.data:
                phases = [{'phase_idx': 0, 'duration': MIN_GREEN, 'state': 'Grrr'}]
            else:
                phases = phases.data

            # Update phases (skip any with None as phase_idx)
            for phase in phases:
                if phase.get('phase_idx') is None:
                    print(f"[WARNING] Skipping phase with None phase_idx for tls_id={tls_id}")
                    continue
                new_duration = int(np.clip(phase['duration'] + delta_t, MIN_GREEN, MAX_GREEN))
                supabase.table("phases").update({"duration": new_duration, "updated_at": datetime.datetime.now().isoformat()})\
                    .eq("tls_id", tls_id).eq("phase_idx", phase['phase_idx']).execute()

            # Store phase recommendation (only if phase_idx is not None)
            valid_phases = [p for p in phases if p.get('phase_idx') is not None]
            if valid_phases:
                latest_phase = max(valid_phases, key=lambda x: x['phase_idx'])
                supabase.table("phase_recommendations").insert({
                    "tls_id": tls_id,
                    "phase_idx": latest_phase['phase_idx'],
                    "duration": latest_phase['duration']
                }).execute()

            print(f"[INFO] Processed traffic data for tls_id: {tls_id}, record_id: {record_id}")
        except Exception as e:
            print(f"[ERROR] Error processing record {record_id}: {e}")

def compute_reward(lane_data):
    metrics = np.zeros(4)  # [density, speed, wait, queue]
    valid_lanes = 0
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    MAX_VALUES = [0.2, 13.89, 300, 50]

    for data in lane_data.values():
        queue, wait, speed, density = data['queue_length'], data['waiting_time'], data['mean_speed'], data['density']
        metrics += [
            min(density, MAX_VALUES[0]) / MAX_VALUES[0],
            min(speed, MAX_VALUES[1]) / MAX_VALUES[1],
            min(wait, MAX_VALUES[2]) / MAX_VALUES[2],
            min(queue, MAX_VALUES[3]) / MAX_VALUES[3]
        ]
        valid_lanes += 1

    avg_metrics = metrics / valid_lanes if valid_lanes > 0 else np.zeros(4)
    reward = 100 * (
        -weights[0] * avg_metrics[0] +
        weights[1] * avg_metrics[1] -
        weights[2] * avg_metrics[2] -
        weights[3] * avg_metrics[3]
    )
    return np.clip(reward, -100, 100), 0, 0

def calculate_delta_t(reward):
    raw_delta_t = 1.0 * (reward - 0.5)
    delta_t = np.clip(10 * np.tanh(raw_delta_t / 20), -10, 10)
    penalty = max(0, abs(raw_delta_t) - 10)
    return raw_delta_t, delta_t, penalty

def run_polling():
    while True:
        try:
            process_traffic_data()
        except Exception as e:
            print(f"[ERROR] Polling error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    import threading
    threading.Thread(target=run_polling, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)