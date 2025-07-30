from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import datetime
import os
import json
from supabase import create_client, Client

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://zckiwulodojgcfwyjrcx.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpja2l3dWxvZG9qZ2Nmd3lqcmN4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzE0NDc0NCwiZXhwIjoyMDY4NzIwNzQ0fQ.FLthh_xzdGy3BiuC2PBhRQUcH6QZ1K5mt_dYQtMT2Sc')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

class LaneData(BaseModel):
    id: str
    queue: float
    wait: float
    speed: float
    density: float
    vehicle_ids: List[str]
    vehicle_classes: List[str]
    is_left: bool = False
    is_right: bool = False

class PhaseData(BaseModel):
    index: int
    state: str
    duration: float

class IntersectionState(BaseModel):
    tls_id: str
    current_phase: int
    phases: List[PhaseData]
    lanes: List[LaneData]
    sim_time: float

class RLStepRequest(BaseModel):
    intersection: IntersectionState

class ControlCommand(BaseModel):
    action: str
    phase_index: Optional[int] = None
    state: Optional[str] = None
    duration: Optional[float] = None

class RLStepResponse(BaseModel):
    commands: List[ControlCommand]
    reward: float

class RLController:
    def __init__(self, tls_id):
        self.tls_id = tls_id
        self.min_green = 10
        self.max_green = 60
        self.epsilon = 0.05
        self.learning_rate = 0.1
        self.discount = 0.95
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self.last_reward = 0
        self.starvation = {}

    def get_state_vector(self, intersection):
        queues = [l.queue for l in intersection.lanes]
        waits = [l.wait for l in intersection.lanes]
        speeds = [l.speed for l in intersection.lanes]
        current_phase = intersection.current_phase
        n_phases = len(intersection.phases)
        state = np.array([
            np.max(queues) if queues else 0,
            np.mean(queues) if queues else 0,
            np.max(waits) if waits else 0,
            np.mean(waits) if waits else 0,
            np.min(speeds) if speeds else 0,
            np.mean(speeds) if speeds else 0,
            current_phase,
            n_phases
        ])
        return tuple(np.round(state,2))

    def emergency_detection(self, intersection):
        for lane in intersection.lanes:
            if any(vc in ("emergency", "authority") for vc in lane.vehicle_classes):
                return lane.id
        return None

    def starvation_detection(self, intersection, sim_time):
        for lane in intersection.lanes:
            last = self.starvation.get(lane.id, sim_time)
            if lane.queue > 0 and sim_time - last > 90:
                return lane.id
        return None

    def protected_left_detection(self, intersection):
        for lane in intersection.lanes:
            if lane.is_left and lane.queue >= 2 and lane.speed < 0.1:
                return lane.id
        return None

    def reward(self, intersection):
        queues = [l.queue for l in intersection.lanes]
        waits = [l.wait for l in intersection.lanes]
        speeds = [l.speed for l in intersection.lanes]
        r = -2 * np.mean(queues) - 0.5 * np.mean(waits) + 0.3 * np.mean(speeds)
        return float(np.clip(r, -100, 100))

    def select_action(self, state, n_phases):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(n_phases)
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(n_phases))
        return int(np.argmax(self.q_table[state]))
    
    def update_q(self, prev_state, action, reward, new_state, n_phases):
        if prev_state not in self.q_table:
            self.q_table[prev_state] = np.zeros(n_phases)
        if new_state not in self.q_table:
            self.q_table[new_state] = np.zeros(n_phases)
        q = self.q_table[prev_state][action]
        max_next = np.max(self.q_table[new_state])
        self.q_table[prev_state][action] = q + self.learning_rate * (reward + self.discount * max_next - q)

    def step(self, intersection: IntersectionState):
        sim_time = intersection.sim_time
        cmds = []
        emerg_lane = self.emergency_detection(intersection)
        if emerg_lane:
            target = None
            for ph in intersection.phases:
                idx = intersection.lanes.index(next(l for l in intersection.lanes if l.id == emerg_lane))
                if idx < len(ph.state) and ph.state[idx].upper() == "G":
                    target = ph.index
                    break
            if target is not None:
                cmds.append(ControlCommand(action="switch_phase", phase_index=target))
                cmds.append(ControlCommand(action="set_duration", phase_index=target, duration=30))
                return cmds, 90
        block_left = self.protected_left_detection(intersection)
        if block_left:
            idx = intersection.lanes.index(next(l for l in intersection.lanes if l.id == block_left))
            state = ''.join(['G' if i == idx else 'r' for i in range(len(intersection.lanes))])
            found = None
            for ph in intersection.phases:
                if ph.state == state:
                    found = ph.index
                    break
            if found is None:
                cmds.append(ControlCommand(action="add_phase", state=state, duration=20))
                new_idx = len(intersection.phases)
                cmds.append(ControlCommand(action="switch_phase", phase_index=new_idx))
                cmds.append(ControlCommand(action="set_duration", phase_index=new_idx, duration=20))
                return cmds, 70
            else:
                cmds.append(ControlCommand(action="switch_phase", phase_index=found))
                cmds.append(ControlCommand(action="set_duration", phase_index=found, duration=20))
                return cmds, 70
        starved_lane = self.starvation_detection(intersection, sim_time)
        if starved_lane:
            idx = intersection.lanes.index(next(l for l in intersection.lanes if l.id == starved_lane))
            for ph in intersection.phases:
                if idx < len(ph.state) and ph.state[idx].upper() == "G":
                    cmds.append(ControlCommand(action="switch_phase", phase_index=ph.index))
                    cmds.append(ControlCommand(action="set_duration", phase_index=ph.index, duration=20))
                    self.starvation[starved_lane] = sim_time
                    return cmds, 50
        state = self.get_state_vector(intersection)
        n_phases = len(intersection.phases)
        if self.last_state is not None and self.last_action is not None:
            self.update_q(self.last_state, self.last_action, self.last_reward, state, n_phases)
        action = self.select_action(state, n_phases)
        reward = self.reward(intersection)
        self.last_state = state
        self.last_action = action
        self.last_reward = reward
        cmds.append(ControlCommand(action="switch_phase", phase_index=action))
        cmds.append(ControlCommand(action="set_duration", phase_index=action, duration=self.min_green))
        return cmds, reward

controllers: Dict[str, RLController] = {}

@app.websocket("/apc/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            req_dict = json.loads(data)
            req = RLStepRequest(**req_dict)
            tls_id = req.intersection.tls_id
            if tls_id not in controllers:
                controllers[tls_id] = RLController(tls_id)
            controller = controllers[tls_id]
            cmds, reward = controller.step(req.intersection)
            resp = RLStepResponse(commands=cmds, reward=reward)
            await websocket.send_text(resp.json())
    except WebSocketDisconnect:
        print("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_phase_client:app", host="0.0.0.0", port=8000, reload=True)