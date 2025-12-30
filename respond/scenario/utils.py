from typing import List, Tuple, Optional, Union, Dict

from highway_env.vehicle.controller import ControlledVehicle
import numpy as np
from rich import print 

def vehicleKeyInfo(car: ControlledVehicle) -> Dict[str, any]:

    x = car.position[0]
    y = car.position[1]
    speedVec = car.speed
    headRatio = car.heading
    return {
        'carId': id(car) % 1000,
        'x': x, 
        'shift_y': y, 
        'speed': speedVec,
        'heading': headRatio
    }

def vehicleKeyInfoMore(car: ControlledVehicle) -> Dict[str, any]:
  
    x = car.position[0]
    y = car.position[1]
    speedVec = car.speed
    headRatio = car.heading
    laneId = car.lane_index[2]
    return {
        'carId': id(car) % 1000,
        'x': x, 
        'shift_y': y, 
        'speed': speedVec,
        'heading': headRatio,
        'laneId': laneId
    }

def vehicleInfo(car: ControlledVehicle, distance:float) -> Dict[str, any]:
    if car is None:
        return None
    x = car.position[0]
    y = car.position[1]
    speedVec = car.speed
    headRatio = car.heading
    return {
        'carId': id(car) % 1000,
        'x': x, 
        'shift_y': y, 
        'speed': speedVec,
        'heading': headRatio,
        'laneId': car.lane_index[2],
        'dis': distance
    }

def risk_of_two(main: np.ndarray, other: np.ndarray, threshold: float = 0.01) -> float:
    if other is None or main is None:
        return 0
    total = np.sum(other > threshold)
    if total == 0:
        return 0
    
    if other.shape != main.shape:
        other = other.reshape(main.shape)  
        print("warning, the shape is different for cpu/gpu tensor mix using together for debugging")

    intersection = np.sum((main > threshold) & (other > threshold))

    max_value1 = np.max(main)
    max_value2 = np.max(other)
    # print("max_value1: ", max_value1)
    # print("max_value2: ", max_value2)
    # print( "enter risk_of_two, the threshold is: ", threshold)
    # print(f"first np parameter's area: {np.sum(main > threshold)}, total(the second np parameter's area): {total}, overlap_area: {intersection}")

    return intersection / total

def printCarInfo(lst: List[Dict[str, any]]) -> None:

    # print(f"[bold yellow]Total number of vehicles: {len(lst)}[/bold yellow]")

    for i, car in enumerate(lst):
  
        car_id = car.get('carId', 'N/A')
        x = car.get('x', 'N/A')
        shift_y = car.get('shift_y', 'N/A')
        speed = car.get('speed', 'N/A')
        heading = car.get('heading', 'N/A')
        lane_id = car.get('laneId', 'N/A')

 
        if i == 0:
           
            print(f"[green]Ego Vehicle - Car {i + 1}:[/green]")
            print(f"[green]  Car ID:< {car_id}>[/green]")
            print(f"[green]  X Position: {x:.2f}[/green]")
            print(f"[green]  Y Position (shift_y): {shift_y:.2f}[/green]")
            print(f"[green]  Speed: {speed:.2f} m/s[/green]")
            print(f"[green]  Heading: {heading:.2f} radians[/green]")
            print(f"[green]  Lane ID: {lane_id}[/green]")
        else:
          
            print(f"[cyan]Vehicle {i + 1}:[/cyan]")
            print(f"[cyan]  Car ID: <{car_id}>[/cyan]")
            print(f"[cyan]  X Position: {x:.2f}[/cyan]")
            print(f"[cyan]  Y Position (shift_y): {shift_y:.2f}[/cyan]")
            print(f"[cyan]  Speed: {speed:.2f} m/s[/cyan]")
            print(f"[cyan]  Heading: {heading:.2f} radians[/cyan]")
            print(f"[cyan]  Lane ID: {lane_id}[/cyan]")



def remove_actions_from_string(actions_string: str, actions_to_remove: List[int]) -> str:

    lines = actions_string.split("\n")


    filtered_lines = [
        line for line in lines
        if not any(f"Action_id: {action_id}" in line for action_id in actions_to_remove)
    ]


    return "\n".join(filtered_lines)


def is_matched_action_the_only_one_available(
    matched_action_id: int,
    available_action_str: str
) -> Tuple[bool, bool]:

    import re
    action_ids = re.findall(r"Action_id: (\d+)", available_action_str)
    action_ids = list(map(int, action_ids)) 


    is_in_available_actions = matched_action_id in action_ids


    lane_change_actions = [action_id for action_id in action_ids if action_id in [0, 2]]
    is_unique_lane_change_action = len(lane_change_actions) == 1 and matched_action_id in lane_change_actions

    return is_in_available_actions, is_unique_lane_change_action

def change_lane_solution(risk_pattern: List[int]) -> Tuple[int, int]:

    solution_count = 2 
    changed_lane_action_id = 1 

    if risk_pattern[3] == 3 and risk_pattern[4] == 3 and risk_pattern[5] == 3:
        solution_count -= 1

    if risk_pattern[9] == 3 and risk_pattern[10] == 3 and risk_pattern[11] == 3:
        solution_count -= 1

    if solution_count == 0:
        return solution_count, -1

    if solution_count == 1:
        if risk_pattern[3] == 3 and risk_pattern[4] == 3 and risk_pattern[5] == 3:

            if risk_pattern[10] == 3:
                return 0, -1 
            return solution_count, 2  
        if risk_pattern[9] == 3 and risk_pattern[10] == 3 and risk_pattern[11] == 3:

            if risk_pattern[4] == 3:
                return 0, -1  
            return solution_count, 0  


    if solution_count == 2:

        if risk_pattern[4] > risk_pattern[10]: 
            return 1, 2 
        elif risk_pattern[4] < risk_pattern[10]:    
            return 1, 0 
        elif risk_pattern[4] == 3 and risk_pattern[10] == 3:    
            return 0, -1  
        elif risk_pattern[4] == risk_pattern[10]:
           
            left_sum = risk_pattern[3] + risk_pattern[4] + risk_pattern[5]
            right_sum = risk_pattern[9] + risk_pattern[10] + risk_pattern[11]
            if left_sum < right_sum:
                return 1, 0  
            elif left_sum > right_sum:
                return 1, 2  
            else:
                return 2, 1  

    print("[WARNING] Should not be here, sth must be wrong!")
    return solution_count, changed_lane_action_id
