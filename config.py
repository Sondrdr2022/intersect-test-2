import os
import threading
import time
import random
import logging

# Use environment variables as default, fallback to hardcoded (for development)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://zizihmglxsobyxvgzosa.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InppemlobWdseHNvYnl4dmd6b3NhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzY0NDY2NSwiZXhwIjoyMDczMjIwNjY1fQ.76nGiK9483TXczGWvH0y1IpmQaA01nASAtRHfzJUR5Q")

# Logging config
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # or "DEBUG", etc.

# SUMO HOME
SUMO_HOME = os.getenv("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")

LOGIC_MUTATION_COOLDOWN_S = 3.0     # rate-limit for add/overwrite phase ops
MAX_PENDING_DB_OPS = 200            # cap the supabase queue

# How long we tolerate a "yellow-only" or non-green state before forcing rotation
YELLOW_MAX_HOLD_S = 4.0

# --- SAFETY: Dilemma zone and yellow logic parameters (conservative defaults) ---
MIN_GREEN_HOLD_S = 5.0          # Minimum time a phase must stay green before preemption (↑ from 3.0)
DZ_EXTENSION_SLICE_S = 1.0      # Extension added when a dilemma vehicle blocks switch
DZ_MAX_CUM_EXT_S = 4.0          # Maximum cumulative extension due to dilemma gating
DZ_SPEED_FILTER = 0.5           # Below this speed (m/s) we treat vehicle as already stopped
DZ_TIME_BUFFER = 4.0            # Time buffer (s) for dilemma zone detection (↑ from 2.5)
DZ_DIST_FALLBACK = 50.0         # Fallback distance if no controller attribute threshold (↑ from 12.0)
DYNAMIC_YELLOW = True           # Enable dynamic yellow computation
REACTION_TIME_S = 2.5           # Reaction time used for dynamic yellow (↑ from 1.0)
COMFORT_DECEL = 2.0             # Comfortable decel (m/s^2) (↓ from 4.5)
MIN_YELLOW_S = 5.0              # Minimum yellow duration (↑ from 3.0)
MAX_YELLOW_S = 12.0             # Cap yellow duration (↑ from 6.0)

# Extra approach safety parameters
HIGH_SPEED_THRESHOLD = 10.0     # m/s (36 km/h) threshold for "high speed"
CRITICAL_APPROACH_TIME = 4.0    # seconds: high-speed vehicle within v*t is critical
SAFETY_MARGIN_FACTOR = 1.2      # Additional margin multiplier on stop distance

# New: DB behavior toggles and timeouts expected by Lane8.py
DB_MODE = os.getenv("DB_MODE", "supabase").lower()       # "supabase" | "file" | "disabled"
DB_HTTP_TIMEOUT_S = float(os.getenv("DB_HTTP_TIMEOUT_S", "3.0"))
FALLBACK_DIR = os.getenv("DB_FALLBACK_DIR", "offline_db")

# Write/flush settings (kept from your current config)
DB_WRITE_INTERVAL = 300  # Write every 5 minutes
DB_BATCH_SIZE = 500      # Larger batches
USE_MEMORY_CACHE = True  # Use in-memory caching
PHASE_CAP = int(os.getenv("SUMO_PHASE_CAP", "32"))

# Throttle repeated [YELLOW AUDIT] logs for the same (tls, from, to) pair
YELLOW_AUDIT_SUPPRESS_WINDOW_S = float(os.getenv("YELLOW_AUDIT_SUPPRESS_WINDOW_S", "5.0"))

# Optional: hard on/off switch for strict enforcement (kept on by default)
STRICT_YELLOW_ENFORCEMENT = os.getenv("STRICT_YELLOW_ENFORCEMENT", "true").lower() == "true"
# --- PATCH: Non-blocking Async Supabase Writer implementation ---
class PatchedAsyncSupabaseWriter(threading.Thread):
    """
    Replacement for AsyncSupabaseWriter.
    Ensures all DB writes are done asynchronously and isolates Supabase calls.
    Prevents simulation freeze by never calling traci or simulation logic from the DB thread.
    """
    def __init__(self, apc, interval=DB_WRITE_INTERVAL, max_batch=DB_BATCH_SIZE):
        super().__init__(daemon=True)
        self.apc = apc
        self.interval = interval
        self.max_batch = max_batch
        self.running = True
        self.logger = logging.getLogger("controller")

    def run(self):
        while self.running:
            try:
                self._flush_all()
            except Exception as e:
                self.logger.info(f"[PatchedSupabaseWriter] flush error: {e}")
            time.sleep(self.interval)

    def _flush_all(self):
        # Only perform DB calls here; never call traci or simulation logic.
        self._safe_flush(self.apc.flush_pending_supabase_writes, "apc_states")
        self._safe_flush(self.apc.flush_pending_phase_records, "phase_records")
        self._safe_flush(self.apc.flush_pending_events, "simulation_events")

    def _safe_flush(self, flush_func, name):
        try:
            # If the flush function supports a timeout parameter, pass it
            code = getattr(flush_func, "__code__", None)
            if code and "timeout" in code.co_varnames:
                flush_func(max_retries=3, max_batch=self.max_batch, timeout=DB_HTTP_TIMEOUT_S)
            else:
                flush_func(max_retries=3, max_batch=self.max_batch)
        except Exception as e:
            self.logger.info(f"[PatchedSupabaseWriter] {name} flush error: {e}")

    def stop(self):
        self.running = False

# When config is imported, patch the AsyncSupabaseWriter globally if not already patched
def _patch_async_supabase_writer():
    try:
        import Lane7b
        Lane7b.AsyncSupabaseWriter = PatchedAsyncSupabaseWriter
        logging.getLogger("controller").info(
            "[CONFIG PATCH] AsyncSupabaseWriter replaced with non-blocking async version."
        )
    except Exception as e:
        logging.getLogger("controller").info(
            f"[CONFIG PATCH] Could not patch AsyncSupabaseWriter: {e}"
        )

_patch_async_supabase_writer()