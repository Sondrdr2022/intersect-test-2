import os
import sys
import traci

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

def detect_turning_lanes_with_traci(sumo_cfg_path):
    sumo_binary = "sumo"
    sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], "bin", sumo_binary), "-c", sumo_cfg_path, "--start", "--quit-on-end"]
    traci.start(sumo_cmd)
    left_turn_lanes = set()
    right_turn_lanes = set()
    for lane_id in traci.lane.getIDList():
        links = traci.lane.getLinks(lane_id)
        for conn in links:
            # Use index 6 for direction string
            if len(conn) > 6:
                if conn[6] == 'l':
                    left_turn_lanes.add(lane_id)
                if conn[6] == 'r':
                    right_turn_lanes.add(lane_id)
    traci.close()
    return left_turn_lanes, right_turn_lanes

if __name__ == "__main__":
    sumo_cfg = r"C:\Users\Admin\Downloads\New-folder--6-\dataset1.sumocfg" # Replace with your .sumocfg path
    left_lanes, right_lanes = detect_turning_lanes_with_traci(sumo_cfg)
    print("Left-turn lanes detected:")
    for lane in sorted(left_lanes):
        print(lane)
    print("\nRight-turn lanes detected:")
    for lane in sorted(right_lanes):
        print(lane)