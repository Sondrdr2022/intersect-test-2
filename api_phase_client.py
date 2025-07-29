import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

SUPABASE_EDGE_URL = os.getenv(
    "SUPABASE_EDGE_URL",
    "https://zckiwulodojgcfwyjrcx.supabase.co/functions/v1/clever-worker"
)
SUPABASE_REST_URL = os.getenv(
    "SUPABASE_REST_URL",
    "https://zckiwulodojgcfwyjrcx.supabase.co/rest/v1/phases"
)
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzE0NDc0NCwiZXhwIjoyMDY4NzIwNzQ0fQ.FLthh_xzdGy3BiuC2PBhRQUcH6QZ1K5mt_dYQtMT2Sc"
)

app = FastAPI()

class LaneData(BaseModel):
    lane_id: str
    queue: float
    wait: float
    speed: float

class APCRequest(BaseModel):
    tls_id: str
    traffic: List[LaneData]
    expected_state_length: Optional[int] = None

def calc_reward(traffic):
    queue_sum = sum(l.queue for l in traffic)
    wait_sum = sum(l.wait for l in traffic)
    speed_sum = sum(l.speed for l in traffic)
    avg_speed = speed_sum / len(traffic) if traffic else 0
    return -queue_sum * 1.0 - wait_sum * 0.5 + avg_speed * 2.0

def calc_delta_t(reward, last_reward, alpha=1.0):
    raw_delta = alpha * (reward - last_reward)
    return max(-10, min(10, raw_delta))

def get_last_reward_from_supabase(tls_id):
    url = "https://zckiwulodojgcfwyjrcx.supabase.co/rest/v1/phase_meta"
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    params = {"tls_id": f"eq.{tls_id}"}
    resp = requests.get(url, headers=headers, params=params)
    if resp.ok and resp.json():
        return resp.json()[0].get("last_reward", 0)
    return 0

@app.post("/apc")
async def apc_endpoint(req: APCRequest):
    tls_id = req.tls_id
    traffic = req.traffic
    expected_state_length = req.expected_state_length or len(traffic)
    last_reward = get_last_reward_from_supabase(tls_id)
    reward = calc_reward(traffic)
    delta_t = calc_delta_t(reward, last_reward)
    base_duration = 30

    all_lane_ids = [l.lane_id for l in traffic]
    phases = []
    for i, lane in enumerate(traffic):
        state_arr = ["G" if j == i else "r" for j in range(len(all_lane_ids))]
        while len(state_arr) < expected_state_length:
            state_arr.append("r")
        if len(state_arr) > expected_state_length:
            state_arr = state_arr[:expected_state_length]
        state_str = "".join(state_arr)
        lane_queue = lane.queue
        duration = max(10, min(80, base_duration + delta_t + lane_queue))
        phases.append({
            "tls_id": tls_id,
            "phase_idx": i,
            "state": state_str,
            "duration": duration
        })
    # Add all-red phase for safety
    phases.append({
        "tls_id": tls_id,
        "phase_idx": len(all_lane_ids),
        "state": "r" * expected_state_length,
        "duration": 7
    })

    # Optionally: sync with Supabase
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    try:
        requests.post(SUPABASE_REST_URL, json=phases, headers=headers)
        # Optionally: update meta
        requests.post("https://zckiwulodojgcfwyjrcx.supabase.co/rest/v1/phase_meta",
                      json={"tls_id": tls_id, "last_reward": reward}, headers=headers)
    except Exception as e:
        # Log error; continue to return phases to client
        print(f"Error syncing with Supabase: {e}")

    return {"phases": phases, "reward": reward}