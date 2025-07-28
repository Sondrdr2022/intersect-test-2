import os
from supabase import create_client
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

@app.post("/create_phases")
async def create_phases(request: Request):
    body = await request.json()
    tls_id = body["tls_id"]
    traffic = body["traffic"]  # List of dicts: [{"lane_id":..., "queue":..., ...}, ...]

    # Example phase logic: green for all lanes with queue > 0
    green_lanes = [lane["lane_id"] for lane in traffic if lane["queue"] > 0]
    all_lanes = [lane["lane_id"] for lane in traffic]
    phase_state = ""
    for lane_id in all_lanes:
        phase_state += "G" if lane_id in green_lanes else "r"

    # Always phase_idx=0, duration=30 for demo
    phase = {
        "tls_id": tls_id,
        "phase_idx": 0,
        "state": phase_state,
        "duration": 30
    }
    # Upsert into phases table
    supabase.table("phases").upsert([phase], on_conflict="tls_id,phase_idx").execute()

    # Return all phases for this tls_id
    phases = (
        supabase.table("phases")
        .select("*")
        .eq("tls_id", tls_id)
        .order("phase_idx")
        .execute()
    )
    return JSONResponse({"phases": phases.data})