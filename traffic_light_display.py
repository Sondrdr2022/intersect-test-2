import tkinter as tk
from tkinter import ttk
import traci

class TrafficLightPhaseDisplay:
    def __init__(self, event_log, poll_interval=1000):
        self.root = tk.Tk()
        self.root.title("Phase Time")
        self.tree = ttk.Treeview(
            self.root,
            columns=(
                "tl_id",
                "current time",
                "next",
                "event type",
                "phase index",
                "phase state",
                "action taken",
                "protected left",
                "blocked"
            ),
            show="headings"
        )
        self.tree.heading("tl_id", text="tl_id")
        self.tree.heading("current time", text="current time")
        self.tree.heading("next", text="next")
        self.tree.heading("event type", text="Event Type")
        self.tree.heading("phase index", text="Phase Index")
        self.tree.heading("phase state", text="Phase State")
        self.tree.heading("action taken", text="Action Taken")
        self.tree.heading("protected left", text="Protected Left Turn")
        self.tree.heading("blocked", text="Is Blocked")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.poll_interval = poll_interval
        self.running = False
        self.ready = True
        self.event_log = event_log

        # For direct phase update support:
        self.phase_idx = None
        self.duration = None
        self.current_time = None
        self.next_switch_time = None
        self.elapsed = None
        self.remaining = None

    def update_table(self):
        if not getattr(self, "ready", True):
            self.root.after(self.poll_interval, self.update_table)
            return
        try:
            for row in self.tree.get_children():
                self.tree.delete(row)
            for tl_id in traci.trafficlight.getIDList():
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
                current_phase = traci.trafficlight.getPhase(tl_id)
                phases = logic.getPhases()
                curr_phase_record = None
                next_phase_record = None
                next_phase = (current_phase + 1) % len(phases)

                # Find latest event for current and next phase
                for p in reversed(self.event_log):
                    if p.get("tls_id") == tl_id and p.get("phase_idx") == current_phase:
                        curr_phase_record = p
                        break
                for p in reversed(self.event_log):
                    if p.get("tls_id") == tl_id and p.get("phase_idx") == next_phase:
                        next_phase_record = p
                        break

                # Prefer live values if present
                if self.phase_idx == current_phase and self.duration is not None and \
                self.current_time is not None and self.next_switch_time is not None:
                    curr_time = self.elapsed
                    next_time = self.remaining
                else:
                    # PATCH: Always use live SUMO phase duration for current phase
                    curr_time = traci.trafficlight.getPhaseDuration(tl_id)
                    # Next phase duration: still try event log, else live, else phase object
                    if next_phase_record and "duration" in next_phase_record:
                        next_time = next_phase_record["duration"]
                    else:
                        if next_phase < len(phases):
                            next_time = phases[next_phase].duration
                        else:
                            next_time = "-"

                event_type = curr_phase_record["action"] if curr_phase_record and "action" in curr_phase_record else self.get_event_type_for(tl_id, current_phase)
                action_taken = curr_phase_record.get("action_taken", event_type) if curr_phase_record else self.get_action_taken_for(tl_id, current_phase)
                phase_state = curr_phase_record["state"] if curr_phase_record and "state" in curr_phase_record else (phases[current_phase].state if current_phase < len(phases) else "-")
                protected_left = curr_phase_record.get("protected_left", False) if curr_phase_record else self.is_protected_left(tl_id, current_phase)
                blocked = curr_phase_record.get("blocked", False) if curr_phase_record else self.is_blocked(tl_id, current_phase)
                self.tree.insert(
                    "", "end",
                    values=(
                        tl_id,
                        curr_time,
                        next_time,
                        event_type,
                        current_phase,
                        phase_state,
                        action_taken,
                        "Yes" if protected_left else "No",
                        "Yes" if blocked else "No"
                    )
                )
        except Exception as e:
            print("[TrafficLightPhaseDisplay ERROR]:", e)
        if self.running:
            self.root.after(self.poll_interval, self.update_table)    
    
    def get_event_type_for(self, tl_id, phase_index):
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                return event.get("action", "Unknown")
        return "Unknown"

    def update_phase_duration(self, phase_idx, duration, current_time, next_switch_time):
        """
        Update the phase time display with the latest timing info.
        """
        self.phase_idx = phase_idx
        self.duration = duration
        self.current_time = current_time
        self.next_switch_time = next_switch_time

        # Calculate elapsed and remaining time for display
        self.elapsed = max(0, current_time - (next_switch_time - duration))
        self.remaining = max(0, next_switch_time - current_time)

        # ADD THIS LINE TO FORCE IMMEDIATE REDRAW
        self.redraw()

    def redraw(self):
        """Force the table to update with the new values."""
        self.update_table()

    def get_action_taken_for(self, tl_id, phase_index):
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                return event.get("action_taken", event.get("action", "Unknown"))
        return "Unknown"

    def is_protected_left(self, tl_id, phase_index):
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                return event.get("protected_left", False) or event.get("action", "").startswith("add_true_protected_left")
        return False

    def is_blocked(self, tl_id, phase_index):
        for event in reversed(self.event_log):
            if event.get("tls_id") == tl_id and event.get("phase_idx") == phase_index:
                return event.get("blocked", False)
        return False

    def start(self):
        self.running = True
        self.update_table()
        self.root.mainloop()

    def stop(self):
        self.running = False
        self.root.quit()