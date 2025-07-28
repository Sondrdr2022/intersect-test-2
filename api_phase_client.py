import os
import requests

SUPABASE_EDGE_URL = os.getenv("SUPABASE_EDGE_URL", "https://zckiwulodojgcfwyjrcx.supabase.co/functions/v1/create_phases")
SUPABASE_REST_URL = os.getenv("SUPABASE_REST_URL", "https://zckiwulodojgcfwyjrcx.supabase.co/rest/v1/phases")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "<YOUR_SUPABASE_KEY>")  # Use anon/public if safe

def post_traffic_to_api(tls_id, traffic_data):
    payload = {
        "tls_id": tls_id,
        "traffic": traffic_data
    }
    resp = requests.post(SUPABASE_EDGE_URL + "/create_phases", json=payload)
    resp.raise_for_status()
    return resp.json().get("phases", [])

def get_phases_from_api(tls_id):
    params = {"tls_id": f"eq.{tls_id}"}
    headers = {"apikey": SUPABASE_KEY}
    resp = requests.get(SUPABASE_REST_URL, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()