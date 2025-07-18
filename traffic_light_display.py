import tkinter as tk
from tkinter import ttk
import threading
import traci

class TrafficLightPhaseDisplay:
    def __init__(self, poll_interval=1000):
        self.root = tk.Tk()
        self.root.title("Phase Time")
        self.tree = ttk.Treeview(self.root, columns=("tl_id", "current time", "next"), show="headings")
        self.tree.heading("tl_id", text="tl_id")
        self.tree.heading("current time", text="current time")
        self.tree.heading("next", text="next")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.poll_interval = poll_interval
        self.running = False

    def update_table(self):
        # Remove old rows
        for row in self.tree.get_children():
            self.tree.delete(row)
        # Query TraCI for all traffic lights
        for tl_id in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
            current_phase = traci.trafficlight.getPhase(tl_id)
            phases = logic.getPhases()
            curr_time = phases[current_phase].duration if current_phase < len(phases) else "-"
            next_phase = (current_phase + 1) % len(phases)
            next_time = phases[next_phase].duration if next_phase < len(phases) else "-"
            self.tree.insert("", "end", values=(tl_id, curr_time, next_time))
        if self.running:
            self.root.after(self.poll_interval, self.update_table)

    def start(self):
        self.running = True
        self.update_table()
        threading.Thread(target=self.root.mainloop, daemon=True).start()

    def stop(self):
        self.running = False
        self.root.quit()