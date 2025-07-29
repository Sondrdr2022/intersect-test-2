import os
import requests

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


def post_traffic_to_api(tls_id, traffic_data, expected_state_length=None):
    payload = {
        "tls_id": tls_id,
        "traffic": traffic_data
    }
    if expected_state_length is not None:
        payload["expectedStateLength"] = expected_state_length
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(SUPABASE_EDGE_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json().get("phases", [])
    except Exception as e:
        print(f"API POST error: {e}")
        return get_phases_from_api(tls_id)

def get_phases_from_api(tls_id):
    params = {"tls_id": f"eq.{tls_id}", "order": "phase_idx"}
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    try:
        resp = requests.get(SUPABASE_REST_URL, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"API GET error: {e}")
        return []

def create_new_phase_in_api(tls_id, state_str, duration):
    new_phase = {
        "tls_id": tls_id,
        "state": state_str,
        "duration": duration
    }
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    try:
        resp = requests.post(SUPABASE_REST_URL, json=new_phase, headers=headers)
        resp.raise_for_status()
        return resp.json()[0] if resp.json() else None
    except Exception as e:
        print(f"API phase creation error: {e}")
        return None