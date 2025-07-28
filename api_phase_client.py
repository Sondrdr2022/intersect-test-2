import os
import requests

SUPABASE_EDGE_URL = os.getenv("SUPABASE_EDGE_URL", "https://zckiwulodojgcfwyjrcx.supabase.co/functions/v1/clever-worker")
SUPABASE_REST_URL = os.getenv("SUPABASE_REST_URL", "https://zckiwulodojgcfwyjrcx.supabase.co/rest/v1/phases")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzE0NDc0NCwiZXhwIjoyMDY4NzIwNzQ0fQ.FLthh_xzdGy3BiuC2PBhRQUcH6QZ1K5mt_dYQtMT2Sc"
)  # Use anon/public or service key as needed

def post_traffic_to_api(tls_id, traffic_data):
    """
    Sends current traffic data for a given tls_id to the clever-worker Edge Function,
    which is expected to process and upsert the phases for that intersection.

    Returns:
        List of phase objects for the intersection.
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
    return resp.json().get("phases", [])

def get_phases_from_api(tls_id):
    """
    Queries the 'phases' table for all phases for a given tls_id.

    Returns:
        List of phase objects.
    """
    params = {"tls_id": f"eq.{tls_id}"}
    headers = {"apikey": SUPABASE_KEY}
    resp = requests.get(SUPABASE_REST_URL, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()