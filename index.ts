import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
// --- CONFIGURE THIS FOR YOUR INTERSECTIONS ---
const EXPECTED_LANE_COUNTS = {
  "E3": 22
};
// ---------------------------------------------
const SUPABASE_URL = Deno.env.get("SUPABASE_URL");
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
function calcReward(traffic) {
  // Lane 7-style reward: penalize queue and wait, reward higher speed
  let queueSum = 0, waitSum = 0, speedSum = 0;
  for (const lane of traffic){
    queueSum += lane.queue || 0;
    waitSum += lane.wait || 0;
    speedSum += lane.speed || 0;
  }
  const avgSpeed = traffic.length ? speedSum / traffic.length : 0;
  return -queueSum * 1.0 - waitSum * 0.5 + avgSpeed * 2.0;
}
function calcDeltaT(reward, lastReward, alpha = 1.0) {
  // Extend/shorten phase based on reward improvement
  const rawDelta = alpha * (reward - lastReward);
  return Math.max(-10, Math.min(10, rawDelta));
}
serve(async (req)=>{
  try {
    if (req.method !== "POST") {
      return new Response("Method Not Allowed", {
        status: 405
      });
    }
    const body = await req.json();
    const tls_id = body.tls_id;
    const traffic = body.traffic;
    if (!tls_id || !Array.isArray(traffic) || traffic.length === 0) {
      return new Response(JSON.stringify({
        error: "Missing or invalid tls_id or traffic"
      }), {
        status: 400,
        headers: {
          "Content-Type": "application/json"
        }
      });
    }
    // Use full expected lane count for phase state
    const expectedStateLength = EXPECTED_LANE_COUNTS[tls_id] || traffic.length;
    const all_lane_ids = traffic.map((lane)=>lane.lane_id);
    // 1. Fetch previous reward for this intersection (simple version: store in a 'phase_meta' table)
    let lastReward = 0;
    const { data: meta } = await supabase.from("phase_meta").select("*").eq("tls_id", tls_id).single();
    if (meta && typeof meta.last_reward === "number") lastReward = meta.last_reward;
    // 2. Compute current reward
    const reward = calcReward(traffic);
    // 3. Compute delta_t (phase duration adjustment)
    const delta_t = calcDeltaT(reward, lastReward);
    // 4. Generate phases (per-lane green, protected lefts, all red)
    const baseDuration = 30;
    const phases = [];
    // Per-lane phases (one green per lane)
    for(let i = 0; i < all_lane_ids.length; i++){
      let stateArr = [];
      for(let j = 0; j < all_lane_ids.length; j++){
        stateArr.push(i === j ? "G" : "r");
      }
      while(stateArr.length < expectedStateLength)stateArr.push("r");
      if (stateArr.length > expectedStateLength) stateArr = stateArr.slice(0, expectedStateLength);
      const stateStr = stateArr.join("");
      const laneQueue = traffic[i]?.queue ?? 0;
      // Smart phase time: longer for congested lanes
      const duration = Math.max(10, Math.min(80, baseDuration + delta_t + laneQueue));
      phases.push({
        tls_id,
        phase_idx: i,
        state: stateStr,
        duration
      });
    }
    // All-red phase for safety
    phases.push({
      tls_id,
      phase_idx: all_lane_ids.length,
      state: "r".repeat(expectedStateLength),
      duration: 7
    });
    // 5. Upsert all phases
    const { error: upsertError } = await supabase.from("phases").upsert(phases, {
      onConflict: "tls_id,phase_idx"
    });
    if (upsertError) {
      console.error("Upsert error:", upsertError);
      return new Response(JSON.stringify({
        error: upsertError.message
      }), {
        status: 500,
        headers: {
          "Content-Type": "application/json"
        }
      });
    }
    // 6. Save last reward for next step (meta table)
    await supabase.from("phase_meta").upsert({
      tls_id,
      last_reward: reward
    }, {
      onConflict: "tls_id"
    });
    // 7. Return all phases for this intersection
    const { data: dbPhases, error: selectError } = await supabase.from("phases").select("*").eq("tls_id", tls_id).order("phase_idx");
    if (selectError) {
      console.error("Select error:", selectError);
      return new Response(JSON.stringify({
        error: selectError.message
      }), {
        status: 500,
        headers: {
          "Content-Type": "application/json"
        }
      });
    }
    return new Response(JSON.stringify({
      phases: dbPhases
    }), {
      status: 200,
      headers: {
        "Content-Type": "application/json"
      }
    });
  } catch (e) {
    console.error("Unhandled error:", e);
    return new Response(JSON.stringify({
      error: String(e)
    }), {
      status: 500,
      headers: {
        "Content-Type": "application/json"
      }
    });
  }
});
