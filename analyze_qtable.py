import pickle
import json
import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_state_data(state_data: Union[str, list, np.ndarray]) -> Dict[str, float]:
    if isinstance(state_data, str):
        try:
            state_array = json.loads(state_data)
        except json.JSONDecodeError:
            logger.debug("Failed to decode JSON state data")
            return {}
    elif isinstance(state_data, (list, np.ndarray)):
        state_array = state_data
    else:
        logger.debug(f"Unsupported state data type: {type(state_data)}")
        return {}
    
    if len(state_array) >= 12:  # Updated for 12-element state vector
        try:
            return {
                'queue_norm': round(float(state_array[0]), 4),
                'wait_norm': round(float(state_array[1]), 4),
                'density_norm': round(float(state_array[2]), 4),
                'speed_norm': round(float(state_array[3]), 4),
                'flow_norm': round(float(state_array[4]), 4),
                'route_queue_norm': round(float(state_array[5]), 4),
                'route_flow_norm': round(float(state_array[6]), 4),
                'phase_norm': round(float(state_array[7]), 4),
                'time_since_green_norm': round(float(state_array[8]), 4),
                'has_ambulance': round(float(state_array[9])),  # Binary flag
                'is_left_turn': round(float(state_array[10])),  # Binary flag
                'lane_score_norm': round(float(state_array[11]), 4)
            }
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting state values: {e}")
            return {}
    return {}

def clean_reward_components(reward_data: Union[Dict, str]) -> Dict[str, float]:
    """
    Clean and normalize reward components data.
    
    Args:
        reward_data: Either a dictionary of reward components or a JSON string
        
    Returns:
        Cleaned dictionary of reward components
    """
    if isinstance(reward_data, str):
        try:
            reward_data = json.loads(reward_data)
        except json.JSONDecodeError:
            return {}
    
    if not isinstance(reward_data, dict):
        return {}
    
    # Clean each component
    cleaned = {}
    for key, value in reward_data.items():
        try:
            cleaned[key] = round(float(value), 4)
        except (ValueError, TypeError):
            continue
    
    return cleaned

def export_training_data_xlsx_only(training_data: List[Dict[str, Any]], base_filename: str) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Enhanced export function to handle reward components and new state format.
    
    Args:
        training_data: List of dictionaries containing training data
        base_filename: Base filename for output (will be converted to .xlsx)
        
    Returns:
        Tuple containing:
            - Cleaned DataFrame
            - List of exported files with their types and paths
    """
    clean_data = []
    logger.info(f"Processing {len(training_data)} training entries...")
    
    # Predefine expected columns and default values
    DEFAULT_VALUES = {
        'episode': 0,
        'simulation_time': 0.0,
        'action': 0,
        'action_name': '',
        'reward': 0.0,
        'q_value': 0.0,
        'lane_id': '',
        'edge_id': '',
        'route_id': '',
        'queue_length': 0,
        'waiting_time': 0,
        'density': 0.0,
        'mean_speed': 0.0,
        'flow': 0,
        'queue_route': 0,
        'flow_route': 0,
        'ambulance': False,
        'left_turn': False,
        'tl_id': '',
        'phase_id': -1,
        'epsilon': 0.1,
        'learning_rate': 0.1,
        'state': [],
        'next_state': [],
        'reward_components': {},
        'raw_reward': 0
    }
    
    for i, entry in enumerate(training_data):
        if i % 1000 == 0:  # Reduced frequency of progress updates
            logger.info(f"Processed {i}/{len(training_data)} entries")
        
        # Create clean entry with defaults
        clean_entry = {
            key: entry.get(key, default)
            for key, default in DEFAULT_VALUES.items()
            if key not in ['state', 'next_state', 'reward_components']  # Handle these separately
        }
        
        # Convert specific fields to proper types
        clean_entry['episode'] = int(clean_entry.get('episode', 0))
        clean_entry['simulation_time'] = float(clean_entry['simulation_time'])
        clean_entry['action'] = int(clean_entry['action'])
        clean_entry['reward'] = float(clean_entry['reward'])
        clean_entry['q_value'] = float(clean_entry['q_value'])
        clean_entry['density'] = float(clean_entry['density'])
        clean_entry['mean_speed'] = float(clean_entry['mean_speed'])
        clean_entry['ambulance'] = bool(clean_entry['ambulance'])
        clean_entry['left_turn'] = bool(clean_entry['left_turn'])
        clean_entry['epsilon'] = float(clean_entry['epsilon'])
        clean_entry['learning_rate'] = float(clean_entry['learning_rate'])
        clean_entry['raw_reward'] = int(entry.get('raw_reward', 0))
        
        # Add state data
        state_cols = clean_state_data(entry.get('state', []))
        clean_entry.update({f'state_{k}': v for k, v in state_cols.items()})
        
        # Add next_state data
        next_state_cols = clean_state_data(entry.get('next_state', []))
        clean_entry.update({f'next_state_{k}': v for k, v in next_state_cols.items()})
        
        # Add reward components
        reward_components = clean_reward_components(entry.get('reward_components', {}))
        clean_entry.update({f'reward_{k}': v for k, v in reward_components.items()})
        
        clean_data.append(clean_entry)
    
    # Create DataFrame more efficiently
    df = pd.DataFrame(clean_data)
    
    # Sort and reset index
    sort_columns = ['episode', 'simulation_time', 'lane_id', 'edge_id']
    df = df.sort_values([col for col in sort_columns if col in df.columns])
    df = df.reset_index(drop=True)
    
    # Define and reorder columns
    column_order = [
        'episode', 'simulation_time', 'lane_id', 'edge_id', 'route_id',
        'action', 'action_name', 'reward', 'raw_reward', 'q_value', 
        'queue_length', 'waiting_time', 'density', 'mean_speed', 'flow',
        'queue_route', 'flow_route', 'ambulance', 'left_turn',
        'tl_id', 'phase_id', 'epsilon', 'learning_rate',
        # State columns
        'state_queue_norm', 'state_wait_norm', 'state_density_norm', 
        'state_speed_norm', 'state_flow_norm', 'state_route_queue_norm',
        'state_route_flow_norm', 'state_phase_norm', 'state_time_since_green_norm',
        'state_has_ambulance', 'state_is_left_turn', 'state_lane_score_norm',
        # Next state columns
        'next_state_queue_norm', 'next_state_wait_norm', 'next_state_density_norm',
        'next_state_speed_norm', 'next_state_flow_norm', 'next_state_route_queue_norm',
        'next_state_route_flow_norm', 'next_state_phase_norm', 'next_state_time_since_green_norm',
        'next_state_has_ambulance', 'next_state_is_left_turn', 'next_state_lane_score_norm',
        # Reward components
        'reward_queue_penalty', 'reward_wait_penalty', 'reward_throughput_reward',
        'reward_speed_reward', 'reward_action_bonus', 'reward_starvation_penalty',
        'reward_ambulance_bonus', 'reward_left_turn_bonus', 'reward_total_raw',
        'reward_normalized'
    ]
    
    # Filter to available columns only
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    # Export to Excel with error handling
    exported_files = []
    excel_file = Path(base_filename).with_suffix('.xlsx')
    
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, float_format='%.6f')
        exported_files.append(('Excel file', str(excel_file)))
        logger.info(f"Successfully exported Excel file: {excel_file}")
    except Exception as e:
        logger.error(f"Excel export failed: {e}", exc_info=True)
    
    return df, exported_files

def aggregate_by_episode_and_lane(df: pd.DataFrame, save_path: str = None) -> Union[pd.DataFrame, None]:
    """
    Enhanced aggregation function that groups by episode and lane.
    
    Args:
        df: Input DataFrame containing training data
        save_path: Optional path to save aggregated data
        
    Returns:
        Aggregated DataFrame if successful, None otherwise
    """
    required_columns = {'episode', 'lane_id'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logger.error(f"DataFrame missing required columns: {missing}")
        return None
    
    # Set default save path if not provided
    if save_path is None:
        save_path = "aggregated_by_episode_and_lane.xlsx"
    
    # Define aggregation functions
    aggregations = {
        "reward": ["mean", "sum", "max", "min", "std"],
        "raw_reward": ["mean", "sum", "max", "min", "std"],
        "q_value": ["mean", "max", "min", "std"],
        "queue_length": ["mean", "max"],
        "waiting_time": ["mean", "max"],
        "flow": ["mean", "sum"],
        "action": ["nunique"],
        # Reward components
        **{col: ["mean", "sum"] for col in df.columns if col.startswith('reward_')}
    }
    
    # Filter to only columns that exist in the DataFrame
    valid_aggregations = {
        col: funcs for col, funcs in aggregations.items()
        if col in df.columns
    }
    
    try:
        # Group by episode and lane
        agg_df = df.groupby(['episode', 'lane_id']).agg(valid_aggregations)
        
        # Flatten multi-level columns
        agg_df.columns = ['_'.join(filter(None, col)).strip('_') 
                         for col in agg_df.columns.values]
        
        # Reset index to make episode and lane_id regular columns
        agg_df = agg_df.reset_index()
        
        # Calculate some derived metrics
        if 'reward_mean' in agg_df.columns and 'reward_sum' in agg_df.columns:
            agg_df['reward_per_action'] = agg_df['reward_sum'] / agg_df['action_nunique']
        
        # Save to Excel if requested
        if save_path:
            try:
                agg_df.to_excel(save_path, index=False)
                logger.info(f"Aggregated data saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save aggregated data: {e}")
        
        return agg_df
    
    except Exception as e:
        logger.error(f"Error during aggregation: {e}", exc_info=True)
        return None

def analyze_qtable(pkl_file: str = 'enhanced_q_table.pkl') -> None:
    """
    Enhanced analysis function to handle the new data structure.
    
    Args:
        pkl_file: Path to the pickle file containing Q-table data
    """
    try:
        logger.info(f"Loading {pkl_file}...")
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        logger.info("Successfully loaded pickle file!")
        logger.info("="*60)
        
        # File structure analysis
        logger.info("File structure analysis:")
        logger.info(f"Keys in file: {list(data.keys())}")
        
        # Q-table analysis
        if 'q_table' in data:
            q_table = data['q_table']
            logger.info("\nQ-TABLE ANALYSIS:")
            logger.info(f"Number of states learned: {len(q_table)}")
            
            total_q_values = sum(len(q_values) for q_values in q_table.values())
            non_zero_q_values = sum(
                sum(1 for q_val in q_values if abs(q_val) > 1e-10)
                for q_values in q_table.values()
            )
            
            logger.info(f"Total Q-values: {total_q_values}")
            logger.info(f"Non-zero Q-values: {non_zero_q_values}")
            if total_q_values > 0:
                logger.info(f"Learning progress: {non_zero_q_values/total_q_values*100:.2f}%")
        
        # Training data analysis
        if 'training_data' in data:
            training_data = data['training_data']
            logger.info("\nTRAINING DATA ANALYSIS:")
            logger.info(f"Total training entries: {len(training_data)}")
            
            if training_data:
                # Action distribution
                actions = [entry.get('action', 0) for entry in training_data]
                action_names = [entry.get('action_name', 'Unknown') for entry in training_data]
                unique_actions = sorted(set(zip(actions, action_names)), key=lambda x: x[0])
                
                logger.info("\nACTION DISTRIBUTION:")
                for action, name in unique_actions:
                    count = actions.count(action)
                    percentage = (count / len(actions)) * 100
                    logger.info(f"  Action {action} ({name}): {count} times ({percentage:.1f}%)")
                
                # Reward statistics
                rewards = [entry.get('reward', 0) for entry in training_data]
                raw_rewards = [entry.get('raw_reward', 0) for entry in training_data]
                logger.info("\nREWARD STATISTICS:")
                logger.info(f"  Average reward: {np.mean(rewards):.6f}")
                logger.info(f"  Max reward: {np.max(rewards):.6f}")
                logger.info(f"  Min reward: {np.min(rewards):.6f}")
                logger.info(f"  Reward std dev: {np.std(rewards):.6f}")
                logger.info(f"  Average raw reward: {np.mean(raw_rewards):.2f}")
                logger.info(f"  Max raw reward: {np.max(raw_rewards):.2f}")
                logger.info(f"  Min raw reward: {np.min(raw_rewards):.2f}")
                logger.info(f"  Raw reward std dev: {np.std(raw_rewards):.2f}")
                
                # Reward components analysis if available
                if any('reward_components' in entry for entry in training_data):
                    logger.info("\nREWARD COMPONENTS ANALYSIS:")
                    reward_components = defaultdict(list)
                    for entry in training_data:
                        components = entry.get('reward_components', {})
                        for key, value in components.items():
                            reward_components[key].append(float(value))
                    
                    for component, values in reward_components.items():
                        logger.info(f"  {component}:")
                        logger.info(f"    Avg: {np.mean(values):.4f}")
                        logger.info(f"    Min: {np.min(values):.4f}")
                        logger.info(f"    Max: {np.max(values):.4f}")
                        if np.mean(rewards) != 0:
                            logger.info(f"    Contribution: {np.mean(values)/np.mean(rewards)*100:.1f}% of normalized reward")
        
        # Parameter analysis
        if 'params' in data:
            params = data['params']
            logger.info("\nLEARNING PARAMETERS:")
            for key, value in params.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.6f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        # Adaptive parameters analysis
        if 'adaptive_params' in data:
            adaptive_params = data['adaptive_params']
            logger.info("\nADAPTIVE PARAMETERS:")
            for key, value in adaptive_params.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        # Export training data
        if 'training_data' in data and data['training_data']:
            export = input("\nExport training data as Excel (.xlsx)? (y/n): ").lower().strip()
            if export == 'y':
                base_filename = str(Path(pkl_file).with_suffix('')) + "_training.xlsx"
                df, exported_files = export_training_data_xlsx_only(data['training_data'], base_filename)
                
                logger.info("\nEXPORT SUMMARY:")
                logger.info(f"Data shape: {df.shape}")
                if exported_files:
                    logger.info(f"Exported to: {exported_files[0][1]}")
                
                # Offer aggregation
                aggregate = input("\nAggregate by episode and lane? (y/n): ").lower().strip()
                if aggregate == 'y':
                    save_path = str(Path(base_filename).with_name(
                        Path(base_filename).stem + "_aggregated.xlsx"
                    ))
                    agg_df = aggregate_by_episode_and_lane(df, save_path=save_path)
                    if agg_df is not None:
                        logger.info(f"Aggregation complete. Saved to {save_path}")
    
    except FileNotFoundError:
        logger.error(f"File not found: {pkl_file}")
    except Exception as e:
        logger.error(f"Error loading file: {e}", exc_info=True)

if __name__ == "__main__":
    # Configure more detailed logging when run as script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('qtable_analysis.log')
        ]
    )
    
    # Use first command line argument as file if provided
    import sys
    pkl_file = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Admin\Downloads\New-folder--7-\New folder (2)\enhanced_q_table.pkl"
    
    analyze_qtable(pkl_file)