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

def post_traffic_to_api(tls_id, traffic_data):
    """
    Send current traffic state to the Supabase Edge Function.
    The Edge Function will perform all adaptive phase logic (reward, delta_t, phase creation, duration extension, etc),
    update the phases in Supabase, and return the current list of phases for this intersection.

    Args:
        tls_id (str): Intersection id.
        traffic_data (list of dict): List of lane traffic dicts (queue, wait, speed, etc).

    Returns:
        List of phase objects, each with at least keys: phase_idx, state, duration.
    """
    payload = {
        "tls_id": tls_id,
        "traffic": traffic_data
    }
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.post(SUPABASE_EDGE_URL, json=payload, headers=headers)
    resp.raise_for_status()
    # The Edge Function should return: {"phases": [...]}
    return resp.json().get("phases", [])

def get_phases_from_api(tls_id):
    """
    Fetch all phases for the given intersection from Supabase.

    Args:
        tls_id (str): Intersection id.

    Returns:
        List of phase objects, each with at least keys: phase_idx, state, duration.
    """
    params = {"tls_id": f"eq.{tls_id}"}
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    resp = requests.get(SUPABASE_REST_URL, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()