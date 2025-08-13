import os

# Use environment variables as default, fallback to hardcoded (for development)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://lkozpmizzagordbpmioq.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxrb3pwbWl6emFnb3JkYnBtaW9xIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDg4MDI2OSwiZXhwIjoyMDcwNDU2MjY5fQ.1m_ANZcHO8S_EZ94ACYdBleMarkEeoY9BAduxTJke_U")

# Logging config
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # or "DEBUG", etc.

# SUMO HOME
SUMO_HOME = os.getenv("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")

LOGIC_MUTATION_COOLDOWN_S = 5.0     # rate-limit for add/overwrite phase ops
MAX_PENDING_DB_OPS = 200            # cap the supabase queue
YELLOW_MAX_HOLD_S = 4.0 
