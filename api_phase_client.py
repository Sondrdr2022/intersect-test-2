import time
import datetime
import logging
from supabase import create_client

SUPABASE_URL = "https://zckiwulodojgcfwyjrcx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzE0NDc0NCwiZXhwIjoyMDY4NzIwNzQ0fQ.FLthh_xzdGy3BiuC2PBhRQUcH6QZ1K5mt_dYQtMT2Sc"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
logger = logging.getLogger("api_phase_client")

class APIPhaseClient:
    def __init__(self, tls_id, max_retries=3, retry_delay=1):
        self.tls_id = tls_id
        self.last_submission = 0
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def submit_traffic_data(self, lane_data, sim_time=None):
        now = time.time()
        if now - self.last_submission < 2.0:
            return None

        required_keys = ['vehicle_count', 'mean_speed', 'queue_length', 'waiting_time', 'density', 'vehicle_ids']
        for lane_id, data in lane_data.items():
            if not all(key in data for key in required_keys):
                print(f"[ERROR] Invalid lane_data for {lane_id}: missing keys")
                return None

        payload = {
            "tls_id": self.tls_id,
            "obs": {"lanes": [{k: v for k, v in data.items()} for k, data in lane_data.items()],
                    "timestamp": datetime.datetime.now().isoformat(),
                    "sim_time": sim_time}
        }
        for attempt in range(self.max_retries):
            try:
                supabase.table("traffic_observations").insert(payload).execute()
                self.last_submission = now
                return True
            except Exception as e:
                print(f"[ERROR] API submission failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        print("[ERROR] All retry attempts failed")
        return False

    def get_phase_recommendation(self):
        try:
            response = supabase.table("phase_recommendations")\
                .select("tls_id, phase_idx, duration")\
                .eq("tls_id", self.tls_id)\
                .order("created_at", desc=True).limit(1).execute()
            if response.data:
                rec = response.data[0]
                # Only allow valid phase indices (e.g., phase_idx >= 0)
                if rec.get("phase_idx") is not None and rec["phase_idx"] >= 0:
                    return rec
        except Exception as e:
            print(f"[ERROR] API fetch failed: {e}")
        return None