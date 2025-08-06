import os
import sys
import traci
import sumolib
import tkinter as tk
from tkinter import ttk
import datetime

class SmartIntersectionTrafficDisplay:
    def __init__(self, event_log, poll_interval=500):
        self.root = tk.Tk()
        self.root.title("Smart Intersection Traffic Light Display")
        self.root.minsize(1300, 300)  # Reduced width since removing a column

        # Columns to display
        self.columns = (
            "tl_id", "lane", "time_left", "base_duration", "extended_time",
            "phase_index", "phase_state", "queue_length", "waiting_time", "mean_speed"
        )
        
        self.tree = ttk.Treeview(
            self.root,
            columns=self.columns,
            show="headings"
        )
        
        column_config = {
            "tl_id": ("TL ID", 60),
            "lane": ("Lane", 100),
            "time_left": ("Time Left", 80),
            "base_duration": ("Base Duration", 100),
            "extended_time": ("Extended Time", 100),
            "phase_index": ("Phase Index", 80),
            "phase_state": ("Phase State", 200),
            "queue_length": ("Queue Length", 100),
            "waiting_time": ("Waiting Time", 100),
            "mean_speed": ("Mean Speed", 100)
        }
        
        for col in self.columns:
            heading, width = column_config[col]
            self.tree.heading(col, text=heading)
            self.tree.column(col, minwidth=60, width=width, anchor="center")
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(self.root, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.poll_interval = poll_interval
        self.running = False
        self.event_log = event_log
        self.no_data_label = tk.Label(self.root, text="Waiting for traffic light data...", font=("Arial", 16))

    def get_phase_data(self, tl_id, current_phase):
        """Get phase data from event log with better matching"""
        base_duration = None
        extended_time = None
        
        # Look for the most recent matching event
        for rec in reversed(self.event_log):
            if (rec.get("tls_id") == tl_id and 
                rec.get("phase_idx") == current_phase):
                base_duration = rec.get("base_duration")
                extended_time = rec.get("extended_time")
                break
        
        # If not found in event_log, try to calculate from current data
        if base_duration is None or extended_time is None:
            try:
                current_duration = traci.trafficlight.getPhaseDuration(tl_id)
                base_duration = base_duration if base_duration is not None else current_duration
                extended_time = extended_time if extended_time is not None else 0
            except Exception:
                base_duration = "-"
                extended_time = "-"
        
        return base_duration, extended_time

    def update_table(self):
        try:
            # Clear existing rows
            for row in self.tree.get_children():
                self.tree.delete(row)
            
            # Check if TraCI is loaded and get traffic lights
            tls_list = traci.trafficlight.getIDList() if traci.isLoaded() else []
            if not tls_list:
                self.show_no_data_message("No traffic lights found in the simulation. Waiting for data...")
                self.root.after(self.poll_interval, self.update_table)
                return
            else:
                self.hide_no_data_message()
            
            inserted = False
            
            for tl_id in tls_list:
                try:
                    logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    phases = logic.getPhases()
                    if not phases:
                        continue
                    
                    now = traci.simulation.getTime()
                    next_switch = traci.trafficlight.getNextSwitch(tl_id)
                    phase_state = phases[current_phase].state if current_phase < len(phases) else "-"

                    # Get base_duration and extended_time using improved method
                    base_duration, extended_time = self.get_phase_data(tl_id, current_phase)

                    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                    unique_lanes = sorted(set(controlled_lanes), key=controlled_lanes.index)
                    
                    for lane in unique_lanes:
                        orig_idx = controlled_lanes.index(lane)
                        lane_time_left = max(0, round(next_switch - now, 1))
                        
                        # Get lane statistics safely
                        try:
                            queue_length = traci.lane.getLastStepHaltingNumber(lane)
                            waiting_time = round(traci.lane.getWaitingTime(lane), 1)
                            mean_speed = round(traci.lane.getLastStepMeanSpeed(lane), 2)
                        except Exception:
                            queue_length = 0
                            waiting_time = 0.0
                            mean_speed = 0.0
                        
                        # Format values for display
                        def format_value(val):
                            if val is None:
                                return "-"
                            if isinstance(val, (int, float)):
                                return round(val, 1)
                            return val
                        
                        self.tree.insert(
                            "", "end",
                            values=(
                                tl_id,
                                lane,
                                lane_time_left,
                                format_value(base_duration),
                                format_value(extended_time),
                                current_phase,
                                phase_state,
                                queue_length,
                                waiting_time,
                                mean_speed
                            )
                        )
                        inserted = True
                        
                except Exception as e:
                    print(f"[SmartIntersectionTrafficDisplay ERROR]: Could not update for {tl_id}: {e}")
            
            if not inserted:
                self.show_no_data_message("Waiting for traffic light data...")
            else:
                self.hide_no_data_message()
                
        except Exception as e:
            print("[SmartIntersectionTrafficDisplay ERROR]:", e)
        
        if self.running:
            self.root.after(self.poll_interval, self.update_table)

    def show_no_data_message(self, message):
        self.no_data_label.config(text=message)
        self.no_data_label.grid(row=2, column=0, columnspan=2, pady=20)

    def hide_no_data_message(self):
        self.no_data_label.grid_remove()

    def start(self):
        self.running = True
        self.update_table()
        self.root.mainloop()

    def stop(self):
        self.running = False
        try:
            self.root.quit()
        except Exception:
            pass

def run_simulation_step(display, event_log):
    try:
        if traci.isLoaded():
            traci.simulationStep()
            current_step = traci.simulation.getTime()
            
            # Keep last 1000 entries
            if len(event_log) > 1000:
                event_log.pop(0)
            
            for tl_id in traci.trafficlight.getIDList():
                current_phase = traci.trafficlight.getPhase(tl_id)
                phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
                next_switch = traci.trafficlight.getNextSwitch(tl_id)
                current_time = traci.simulation.getTime()
                logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                phases = logic.getPhases()
                phase_state = phases[current_phase].state if current_phase < len(phases) else "-"
                
                # Try to get extended_time and base_duration if available from event_log
                base_duration = phase_duration
                extended_time = 0
                for rec in reversed(event_log):
                    if rec.get("tls_id") == tl_id and rec.get("phase_idx") == current_phase:
                        base_duration = rec.get("base_duration", rec.get("duration", phase_duration))
                        extended_time = rec.get("extended_time", 0)
                        break
                
                event_log.append({
                    "tls_id": tl_id,
                    "phase_idx": current_phase,
                    "state": phase_state,
                    "duration": phase_duration,
                    "base_duration": base_duration,
                    "current_time": current_time,
                    "next_switch_time": next_switch,
                    "extended_time": extended_time
                })
            
            # Schedule the next step
            if traci.simulation.getMinExpectedNumber() > 0:
                display.root.after(50, run_simulation_step, display, event_log)
            else:
                display.stop()
        else:
            display.stop()
    except traci.exceptions.FatalTraCIError:
        print("[TraCI ERROR]: SUMO-GUI disconnected or closed.")
        display.stop()
    except Exception as e:
        print(f"[Simulation ERROR]: {e}")
        display.stop()

def run_sumo_with_display(sumocfg_file, event_log=None):
    """This function is only used when running the display standalone"""
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    sumo_binary = sumolib.checkBinary("sumo-gui")
    if not os.path.exists(sumocfg_file):
        sys.exit(f"Error: SUMO configuration file '{sumocfg_file}' not found")
    if event_log is None:
        event_log = []
    try:
        traci.start([sumo_binary, "-c", sumocfg_file, "--start", "--quit-on-end"])
        display = SmartIntersectionTrafficDisplay(event_log=event_log, poll_interval=500)
        display.root.after(50, run_simulation_step, display, event_log)
        display.start()
    except Exception as e:
        print(f"[SUMO Simulation ERROR]: {e}")
    finally:
        if traci.isLoaded():
            traci.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sumocfg_file = sys.argv[1]
    else:
        print("Usage: python traffic_light_display.py <path_to_sumocfg_file>")
        print("Or import this module to use with your controller")
        sys.exit(1)
    run_sumo_with_display(sumocfg_file)