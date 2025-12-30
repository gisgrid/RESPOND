import os
from rich import print
import numpy as np
import matplotlib.pyplot as plt
import respond.risk_pattern.risk as risk
from scipy.interpolate import griddata
from typing import Optional, Tuple, Dict, Union
from respond.scenario.utils import vehicleKeyInfo

import torch
import torch.nn.functional as F
from typing import List, Tuple

from highway_env.road.road import Road, RoadNetwork, LaneIndex



RISK_WARNING_DISTANCE = 15  
SAFE_DISTANCE_THRESHOLD = 30  
LOOK_SURROUND_V_NUM = 15

def calculate_overlap_area(risk_field, X_common, Y_common, risk_field1, X_common1, Y_common1, threshold=0.01):

    # print("enter calculate_overlap_area")

    # Calculate the binary masks for the threshold
    mask_ego = risk_field >= threshold
    mask_other = risk_field1 >= threshold

    # Interpolate mask_other onto the grid of mask_ego (X_common, Y_common)

    mask_other_interp = griddata(
        (X_common1.flatten(), Y_common1.flatten()), 
        mask_other.flatten().astype(float), 
        (X_common, Y_common), 
        method='nearest', 
        fill_value=0
    )

    # Calculate the overlapping area
    overlap_mask = (mask_ego > 0) & (mask_other_interp > 0)
    overlap_area = np.sum(overlap_mask)

    # Calculate individual areas
    ego_area = np.sum(mask_ego > 0)
    other_area = np.sum(mask_other_interp > 0)

    # Print the results
    # print(f"Ego car risk field area: {ego_area}")
    # print(f"Other car risk field area (interpolated): {other_area}")
    # print(f"Overlapping area: {overlap_area}")
    # print("leave calculate_overlap_area")
    return overlap_area, overlap_mask


def calculate_overlap_area_optimized(
    risk_field:      np.ndarray,
    X_common:        np.ndarray,
    Y_common:        np.ndarray,
    risk_field1:     np.ndarray,
    X_common1:       np.ndarray,
    Y_common1:       np.ndarray,
    threshold: float = 0.01
) -> int:

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    mask_ego_np   = (risk_field  >= threshold).astype(np.float32)
    mask_other_np = (risk_field1 >= threshold).astype(np.float32)


    mask_other = torch.as_tensor(mask_other_np, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,H1,W1]
    mask_ego   = torch.as_tensor(mask_ego_np,   device=device)                           # [H ,W ]
    # print(f"in optimized function, device type is: {mask_ego.device}")


    x_min1, x_max1 = X_common1.min(), X_common1.max()
    y_min1, y_max1 = Y_common1.min(), Y_common1.max()


    grid_x = 2.0 * (torch.as_tensor(X_common, device=device) - x_min1) / (x_max1 - x_min1) - 1.0
    grid_y = 2.0 * (torch.as_tensor(Y_common, device=device) - y_min1) / (y_max1 - y_min1) - 1.0
    grid   = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1,H,W,2]
    grid = grid.to(mask_other.dtype)     


    mask_other_interp = F.grid_sample(
        mask_other, grid, mode='nearest',
        padding_mode='zeros', align_corners=True
    ).squeeze()        # -> [H,W]


    overlap_mask = (mask_ego > 0.0) & (mask_other_interp > 0.0)
    overlap_area = int(overlap_mask.sum().item())  # 标量 int

    return overlap_area

def riskValueCal(
    frame_name: str, 
    carId_ego: int = 911,
    x_ego: float = 0,
    y_ego: float = 0,
    speed_ego: float = 10,
    heading_angle_ego: float = 0,
    carId1: int = 111, 
    x1: float = 0, 
    y1: float = 0, 
    speed1: float = 10, 
    heading_angle1: float = 0,
    str_pos: str = "no_pos_input"
) -> Tuple[float, float, float]:
    
    """Calculate and visualize the overlapping area of risk fields for the ego car and another car."""
    
    # Define the threshold
    threshold = 0.01
    # print(f"Contour threshold: {threshold}")
    
    if speed_ego < 5: 
        X_common1, Y_common1, risk_field1 = risk.field_distribution_Optimized(
        x=x1, y=y1, speed=speed1, heading_angle=heading_angle1,
        turning_angle=0.1, vehicle_length=5, common_grid=None
        )

        mask_other = risk_field1 >= threshold
        other_area = np.sum(mask_other) 
        print( f"speed_ego < 5, means ego car is stationary, speed_ego: {speed_ego}, other_area: {other_area}")
        
        return 0.0, other_area, 0.0    

    if speed1 < 5: 
        X_common, Y_common, risk_field = risk.field_distribution_Optimized(
        x=x_ego, y=y_ego, speed=speed_ego, heading_angle=heading_angle_ego,
        turning_angle=0.1, vehicle_length=5, common_grid=None
        )

        mask_ego = risk_field >= threshold
        ego_area = np.sum(mask_ego)
        print( f"speed1 < 1, means other car is stationary, speed1: {speed1}, ego_area: {ego_area}")

        return ego_area, 0.0, 0.0   
    

 
    X_common, Y_common, risk_field = risk.field_distribution_Optimized(
        x=x_ego, y=y_ego, speed=speed_ego, heading_angle=heading_angle_ego,
        turning_angle=0.1, vehicle_length=5, common_grid=None
    )
    X_common1, Y_common1, risk_field1 = risk.field_distribution_Optimized(
        x=x1, y=y1, speed=speed1, heading_angle=heading_angle1,
        turning_angle=0.1, vehicle_length=5, common_grid=None
    )

    
    max_value = np.max(risk_field)


    max_index = np.unravel_index(np.argmax(risk_field), risk_field.shape)
    row, column = max_index


    max_value = np.max(risk_field)
    max_value1 = np.max(risk_field1)
    min_value = np.min(risk_field)

    retrieved_value = risk_field[row][column]
    
    
    lap_area = calculate_overlap_area_optimized(
        risk_field, X_common, Y_common, risk_field1, X_common1, Y_common1, threshold=threshold
    )

    #print(f"compare lap_area: {lap_area} and lap_area1: {lap_area1}, the old & new way's calculating result")

    # print(f"mask_ego type: {type(lap_mask)}, shape: {getattr(lap_mask, 'shape', 'N/A')}")

    mask_ego = risk_field >= threshold
    mask_other = risk_field1 >= threshold

    # Calculate the areas
    ego_area = np.sum(mask_ego)
    other_area = np.sum(mask_other)

    return ego_area, other_area, lap_area


def calculate_risk(
    decisionFrame: int,
    ego_info: Dict[str, Union[int, float]],
    position: str,
    vehicle: Optional[Dict[str, Union[int, float]]]
) -> Optional[Tuple[float, float, float]]:

    if vehicle:
        
        if position in ["Left Front", "Left Behind"]:
            ego_shift_y = ego_info['shift_y'] - 4.0  
        elif position in ["Right Front", "Right Behind"]:
            ego_shift_y = ego_info['shift_y'] + 4.0  
        else:
            ego_shift_y = ego_info['shift_y']

        ego_area, other_area, lap_area = riskValueCal(
            decisionFrame,
            ego_info['carId'],
            ego_info['x'],
            ego_shift_y,
            ego_info['speed'],
            ego_info['heading'],
            vehicle['carId'],
            vehicle['x'],
            vehicle['shift_y'],
            vehicle['speed'],
            vehicle['heading'],
            position
        )
        return ego_area, other_area, lap_area
    return None, None, None

def update_ego_pos_risk_road_crashed(env_scenario, ego_vehicle, front_vehicle, behind_vehicle, ego_pos_risk):

    if ego_pos_risk >= 1.0:
        return ego_pos_risk, "", -1, -1
    
    crashed_car_prompt = ""
    returnValue = ego_pos_risk  

    returnLeftFrontRisk = -1
    returnRightFrontRisk = -1

    v_ego = ego_vehicle.speed 
    x_ego = ego_vehicle.position[0]  
    s_ego = 1 * v_ego 
    currentLaneIndex: LaneIndex = env_scenario.ego.lane_index   #self.ego.lane_index
    egoLaneIndex = currentLaneIndex[2]
    
    sideLanes = env_scenario.network.all_side_lanes(currentLaneIndex)
    numLanes = len(sideLanes) 
    
    egoLeftLaneIndex = egoLaneIndex - 1
    egoRightLaneIndex = egoLaneIndex + 1
    
    egoLaneRank = currentLaneIndex[2]
    if egoLaneRank == 0:

        egoLeftLaneIndex = -1
    elif egoLaneRank == numLanes - 1:

        egoRightLaneIndex = -1

    abnormal_vehicles = []
    abnormal_ego_distance = 1000.0      
    abnormal_left_distance = 1000.0     
    abnormal_right_distance = 1000.0    

    surround_vehicles = env_scenario.getSurroundVehicles(LOOK_SURROUND_V_NUM + 10)  # look extra vehicles in crashed scenario
    for sv in surround_vehicles:
        if (sv.speed < 5) and (sv.position[0] > x_ego + RISK_WARNING_DISTANCE) :    
            abnormal_vehicles.append({
                "carId": id(sv) % 1000,
                "position_x": sv.position[0],
                "position_y": sv.position[1],
                "lane_index": sv.lane_index[2],
                "speed": sv.speed,
                "distance_to_ego": sv.position[0] - x_ego
            })
            if sv.lane_index[2] == egoLaneIndex: 
                returnValue = 1.0    
                if abnormal_ego_distance > (sv.position[0] - x_ego):
                    abnormal_ego_distance = sv.position[0] - x_ego
            elif sv.lane_index[2] == egoLeftLaneIndex:  
                if abnormal_left_distance > (sv.position[0] - x_ego):
                    abnormal_left_distance = sv.position[0] - x_ego
            elif sv.lane_index[2] == egoRightLaneIndex:  
                if abnormal_right_distance > (sv.position[0] - x_ego):
                    abnormal_right_distance = sv.position[0] - x_ego

    abnormal_distance = 0.0
    abnormal_threshold = 0.0
    if egoLeftLaneIndex == -1:
        abnormal_distance = abnormal_ego_distance + abnormal_right_distance
        abnormal_threshold = 2000.0
    if egoRightLaneIndex == -1:
        abnormal_distance = abnormal_ego_distance + abnormal_left_distance
        abnormal_threshold = 2000.0
    if (egoLeftLaneIndex != -1) and (egoRightLaneIndex != -1):
        abnormal_distance = abnormal_ego_distance + abnormal_left_distance + abnormal_right_distance
        abnormal_threshold = 3000.0
    
    if (len(abnormal_vehicles) > 0) and (abnormal_distance < abnormal_threshold): 
        crashed_car_prompt = "Be careful! There are crashed vehicles ahead of you and they have stopped in the road."


    for abn_vehicle in abnormal_vehicles:

        '''
        print(f"CarId: {abn_vehicle['carId']}, "
              f"Position: ({abn_vehicle['position_x']:.2f}, {abn_vehicle['position_y']:.2f}), "
              f"Lane Index: {abn_vehicle['lane_index']}, "
              f"Speed: {abn_vehicle['speed']:.2f}, "
              f"Distance to Ego: {abn_vehicle['distance_to_ego']:.2f}")
        '''
        if abn_vehicle['lane_index'] == egoLaneIndex: 
            crashed_car_prompt += f"There are crashed Cars in your lane and has stopped ahead at a distance of {abn_vehicle['distance_to_ego']:.2f} meters. you should consider to decelerate first, so that you might still have chance to change lane in case the left or right risk is not 1.0 (1.0 is the highest risk)\n"
        if abn_vehicle['lane_index'] == egoLeftLaneIndex:
            returnLeftFrontRisk = 1
            # crashed_car_prompt += f"The crashed Car is in your Left lane and has stopped ahead at a distance of {abn_vehicle['distance_to_ego']:.2f} meters.\n"
        if abn_vehicle['lane_index'] == egoRightLaneIndex:
            returnRightFrontRisk = 1
            # crashed_car_prompt += f"The crashed Car is in your Right lane and has stopped ahead at a distance of {abn_vehicle['distance_to_ego']:.2f} meters.\n"
    
    return returnValue, crashed_car_prompt, returnLeftFrontRisk, returnRightFrontRisk

def calculate_ego_pos_risk(env_scenario, ego_vehicle, front_vehicle, behind_vehicle):


    v_ego = ego_vehicle.speed  
    x_ego = ego_vehicle.position[0]  

    s_ego = 1 * v_ego


    risk_ego_f = 0.0  
    risk_ego_b = 0.0  


    if front_vehicle is not None:
        v_f = front_vehicle['speed']    
        x_f = front_vehicle['x']    

        v_rf = v_ego - v_f
        s_rf = 1 * v_rf  
        distance_to_front = x_f - x_ego  


        if distance_to_front <= 0:
            print(f"Exception: distance_to_front <= 0, distance_to_front={distance_to_front}")
            return 1.0  
        

        if v_rf < 0:
            v_rf = 0
            s_rf = 0


        if distance_to_front >= SAFE_DISTANCE_THRESHOLD:
            risk_ego_f = 0
        elif s_rf > 2 * distance_to_front or distance_to_front < RISK_WARNING_DISTANCE:  # 风险警示距离
            risk_ego_f = 1
        else:
            risk_ego_f = 0.34 # just keep it simple. the number 0.34 refer to edge definition


    if behind_vehicle is not None:
        v_b = behind_vehicle['speed'] 
        x_b = behind_vehicle['x']     


        v_rb = v_b - v_ego
        s_rb = 1 * abs(v_rb)  
        distance_to_behind = x_ego - x_b  


        if distance_to_behind <= 0:
            print(f"Exception: distance_to_behind <= 0, distance_to_behind={distance_to_behind}")
            return 1.0 

        if v_rb < 0:
            v_rb = 0
            s_rb = 0


        if distance_to_behind >= SAFE_DISTANCE_THRESHOLD:
            risk_ego_b = 0
        elif s_rb > 2 * distance_to_behind or distance_to_behind < RISK_WARNING_DISTANCE:  
            risk_ego_b = 1
        else:

            risk_ego_b = 0.34  


    risk_ego = risk_ego_f + risk_ego_b
    if risk_ego > 1:
        risk_ego = 1.0


    if front_vehicle is None and behind_vehicle is None:
        risk_ego = 0

    return risk_ego

def getEgoRiskLevel(
    env_scenario, 
    decisionFrame: int
) -> float:
   
    front_vehicle = env_scenario.getFrontV()
    behind_vehicle = env_scenario.getBehindV()

    
    ego_info = vehicleKeyInfo(env_scenario.ego)

    
    ego_area1, other_area1, lap_area1 = calculate_risk(decisionFrame, ego_info, "Front", front_vehicle)
    ego_area2, other_area2, lap_area2 = calculate_risk(decisionFrame, ego_info, "Behind", behind_vehicle)
    
    front_risk = 0.0
    behind_risk = 0.0

   
    if any(value is not None for value in [ego_area1, other_area1, lap_area1]):
        # print("getEgoRiskLevel call calculate_risk: At least one returned value is not None.")
        if other_area1 > 0: 
            
            front_risk = lap_area1 / other_area1
        else:
            print("Warning: Front vehicle is stationary.", front_vehicle)
            
            front_risk = riskCal4StationarySituation(
                ego_info, "Front", front_vehicle
            )
    else: 
        front_risk = 0  
    
 
    distance_to_front = front_vehicle['x'] - ego_info['x'] if front_vehicle else float('inf')
    if distance_to_front <= 10:
        print("Warning: Front vehicle is too close, the distance is: ", distance_to_front)
        front_risk = 1  


    if any(value is not None for value in [ego_area2, other_area2, lap_area2]):
        # print("getEgoRiskLevel call calculate_risk: At least one returned value is not None.")
        if ego_area2 > 0: 

            behind_risk = lap_area2 / ego_area2
        else:
            print("Warning: ego vehicle might be stationary.", behind_vehicle)
            behind_risk = riskCal4StationarySituation(
                ego_info, "Behind", behind_vehicle
            )
            
    else: 
        behind_risk = 0.0  
    
    distance_to_behind = ego_info['x'] - behind_vehicle['x'] if behind_vehicle else float('inf')
    if distance_to_behind <= 10:
        print("Warning: Behind vehicle is too close, the distance is: ", distance_to_behind)
        behind_risk = 1.0 


    overall_risk_level = max(front_risk, behind_risk)
    # print(f"Overall risk level calculated as the maximum of front_risk ({front_risk:.2f}) and behind_risk ({behind_risk:.2f}): {overall_risk_level:.2f}")

    return overall_risk_level


def riskCal4StationarySituation(
    ego_info: Dict[str, Union[int, float]],
    position: str,
    vehicle: Optional[Dict[str, Union[int, float]]]
) -> float:


    # print("<red>in riskCal4StationarySituation, means stationary situation happens</red>")


    if vehicle is None:
        return 0.0


    speed_ego = ego_info.get("speed", 0.0)
    speed_vehicle = vehicle.get("speed", 0.0)


    if speed_ego < 1:
        speed_ego = 0.0
    if speed_vehicle < 1:
        speed_vehicle = 0.0

    relative_speed = 0.0  

    if position in ["Front", "Left Front", "Right Front"]:
        relative_speed = speed_ego - speed_vehicle
    elif position in ["Behind", "Left Behind", "Right Behind"]:
        relative_speed = speed_vehicle - speed_ego
    else:
        raise ValueError(f"Invalid position: {position}")

    if relative_speed <= 0:
        return 0.0


    distance = abs(vehicle.get("x", 0.0) - ego_info.get("x", 0.0))


    if distance <= RISK_WARNING_DISTANCE:
        return 1.0


    ttc = distance / relative_speed


    if ttc <= 3.0:
        risk_value = 1.0
    elif ttc >= 6.0:
        risk_value = 0.0
    else:
        risk_value = (6.0 - ttc) / 3.0 

    return risk_value

# 
def adjust_risk_with_heading(
    vehicles: Dict[str, Optional[Dict[str, Union[int, float]]]],  
    ego_info: Dict[str, Union[int, float]],  
    risks: Dict[str, float]  
) -> Dict[str, float]:


    which_car_risk_is_adjusted = []


    for position, vehicle in vehicles.items():
        if vehicle is None:
            continue  


        heading = vehicle.get("heading")
        lane_id = vehicle.get("laneId")
        y_position = vehicle.get("shift_y") 


        if heading < -0.02:  
            print(f"Handling a right turn vehicle at position: {position}, lane_id: {lane_id}, y_position: {y_position}, heading: {heading}")
            if lane_id * 4 < y_position:  
                print(f"lane_id * 4 < y_position, postion: {position}, lane_id: {lane_id}, y_position: {y_position}, heading: {heading}")
                continue
            elif lane_id * 4 > y_position:  
                which_car_risk_is_adjusted.append(position)
              
                if position in ["Front"]:
                    risks["Left Front"] += risks.get(position, 0.0)
                    if risks["Left Front"] > 1.0:
                        risks["Left Front"] = 1.0
                elif position in ["Behind"]:
                    risks["Left Behind"] += risks.get(position, 0.0)
                    if risks["Left Behind"] > 1.0:
                        risks["Left Behind"] = 1.0
                elif position in ["Right Front"]:
                    risks["Front"] += risks.get(position, 0.0)
                    if risks["Front"] > 1.0:
                        risks["Front"] = 1.0
                elif position in ["Right Behind"]:
                    risks["Behind"] += risks.get(position, 0.0)
                    if risks["Behind"] > 1.0:
                        risks["Behind"] = 1.0
                elif position in ["Left Front"]:
                    pass  
                elif position in ["Left Behind"]:
                    pass
                else: 
                    print(f"Warning: Unexpected situation in handling a car about to turn left, The position is: {position}. The vehicle info is : {vehicle}")
            else:
                print(f"Warning: Unexpected situation, The position is: {position}. The vehicle info is : {vehicle}")

        elif heading > 0.02: 
            print(f"Handling a right turn vehicle at position: {position}, lane_id: {lane_id}, y_position: {y_position}, heading: {heading}")
            if lane_id * 4 > y_position: 
                print(f"lane_id * 4 > y_position, postion: {position}, lane_id: {lane_id}, y_position: {y_position}, heading: {heading}")
                continue
            elif lane_id * 4 < y_position:  
                which_car_risk_is_adjusted.append(position)
              
                if position in ["Front"]:
                    risks["Right Front"] += risks.get(position, 0.0)
                    if risks["Right Front"] > 1.0:
                        risks["Right Front"] = 1.0
                elif position in ["Behind"]:
                    risks["Right Behind"] += risks.get(position, 0.0)
                    if risks["Right Behind"] > 1.0:
                        risks["Right Behind"] = 1.0
                elif position in ["Left Front"]:
                    risks["Front"] += risks.get(position, 0.0)
                    if risks["Front"] > 1.0:
                        risks["Front"] = 1.0
                elif position in ["Left Behind"]:
                    risks["Behind"] += risks.get(position, 0.0)
                    if risks["Behind"] > 1.0:
                        risks["Behind"] = 1.0
                elif position in ["Right Front"]:
                    pass  
                elif position in ["Right Behind"]:
                    pass
                else:
                    print(f"Warning: Unexpected situation in handling a car about to turn right, The position is: {position}. The vehicle info is : {vehicle}")
            else:
                print(f"Warning: Unexpected situation, The position is: {position}. The vehicle info is : {vehicle}")

    # 
    if which_car_risk_is_adjusted:
        print(f"[red]Adjusted risks based on heading: {which_car_risk_is_adjusted}[/red]")
    return risks


def getSurroundingRiskPattern(
    env_scenario,  
    decisionFrame: int
) -> Tuple[str, np.ndarray, np.ndarray]:


    front_vehicle = env_scenario.getFrontV()
    behind_vehicle = env_scenario.getBehindV()
    left_front_vehicle = env_scenario.getLeftFrontV()
    left_behind_vehicle = env_scenario.getLeftBehindV()
    right_front_vehicle = env_scenario.getRightFrontV()
    right_behind_vehicle = env_scenario.getRightBehindV()

    left_left_front_vehicle = env_scenario.getLeftLeftFrontV()
    left_left_behind_vehicle = env_scenario.getLeftLeftBehindV()
    right_right_front_vehicle = env_scenario.getRightRightFrontV()
    right_right_behind_vehicle = env_scenario.getRightRightBehindV()


    ego_info = vehicleKeyInfo(env_scenario.ego)


    risks = {
        "Front": calculate_risk(decisionFrame, ego_info, "Front", front_vehicle),
        "Behind": calculate_risk(decisionFrame, ego_info, "Behind", behind_vehicle),
        "Left Front": calculate_risk(decisionFrame, ego_info, "Left Front", left_front_vehicle),
        "Left Behind": calculate_risk(decisionFrame, ego_info, "Left Behind", left_behind_vehicle),
        "Right Front": calculate_risk(decisionFrame, ego_info, "Right Front", right_front_vehicle),
        "Right Behind": calculate_risk(decisionFrame, ego_info, "Right Behind", right_behind_vehicle),
    }


    front_risk = 0
    behind_risk = 0
    left_front_risk = 0
    left_behind_risk = 0
    right_front_risk = 0
    right_behind_risk = 0
    
    left_risk = 0
    right_risk = 0


    for position, risk in risks.items():
        if risk and all(value is not None for value in risk):
            ego_area, other_area, lap_area = risk
            if position == "Front" and other_area >= 0:
                
                if other_area > 0:
                    front_risk = lap_area / other_area
                else: 
                    # print("Warning: Front vehicle is stationary", front_vehicle)
                    # print("other_area <= 0, other area = " + str(other_area))
                    # print("postion: ", position)
                    front_risk = riskCal4StationarySituation(
                        ego_info, position, front_vehicle)  
            elif position == "Behind" and ego_area >= 0:
                
                if ego_area > 0:
                    behind_risk = lap_area / ego_area
                else: 
                    # print("Warning: Ego vehicle is stationary", behind_vehicle)
                    # print("ego_area <= 0, ego area = " + str(ego_area))
                    # print("postion: ", position)
                    behind_risk = riskCal4StationarySituation(
                        ego_info, position, behind_vehicle)
            elif position == "Left Front" and other_area >= 0:
                #print(f"position: {position}, ego_area: {ego_area}, other_area: {other_area}, lap_area: {lap_area}")
                if other_area > 0:
                    left_front_risk = lap_area / other_area
                else: # 
                    # print("Warning: Left Front vehicle is stationary", left_front_vehicle)
                    # print("other_area <= 0, other area = " + str(other_area))
                    # print("postion: ", position)
                    left_front_risk = riskCal4StationarySituation(
                        ego_info, position, left_front_vehicle)
            elif position == "Left Behind" and ego_area >= 0:
                #print(f"position: {position}, ego_area: {ego_area}, other_area: {other_area}, lap_area: {lap_area}")
                if ego_area > 0:
                    left_behind_risk = lap_area / ego_area
                else: # ego_area <= 0 
                    print("Warning: Left Behind vehicle is stationary", left_behind_vehicle)
                    print("ego_area <= 0, ego area = " + str(ego_area))
                    print("postion: ", position)
                    left_behind_risk = riskCal4StationarySituation(
                        ego_info, position, left_behind_vehicle)
            elif position == "Right Front" and other_area >= 0:
                #print(f"position: {position}, ego_area: {ego_area}, other_area: {other_area}, lap_area: {lap_area}")
                if other_area > 0:
                    right_front_risk = lap_area / other_area
                else: # other_area <= 0
                    # print("Warning: Right Front vehicle is stationary", right_front_vehicle)
                    # print("other_area <= 0, other area = " + str(other_area))
                    # print("postion: ", position)
                    right_front_risk = riskCal4StationarySituation(
                        ego_info, position, right_front_vehicle)
            elif position == "Right Behind" and ego_area >= 0:
                #print(f"position: {position}, ego_area: {ego_area}, other_area: {other_area}, lap_area: {lap_area}")
                if ego_area > 0:
                    right_behind_risk = lap_area / ego_area
                else: 
                    print("Warning: Right Behind vehicle is stationary", right_behind_vehicle)
                    print("ego_area <= 0, ego area = " + str(ego_area))
                    print("postion: ", position)
                    right_behind_risk = riskCal4StationarySituation(
                        ego_info, position, right_behind_vehicle)
 

    vehicles = {
        "Front": front_vehicle,
        "Behind": behind_vehicle,
        "Left Front": left_front_vehicle,
        "Left Behind": left_behind_vehicle,
        "Right Front": right_front_vehicle,
        "Right Behind": right_behind_vehicle,
    }
    risk_values = {
        "Front": front_risk,
        "Behind": behind_risk,
        "Left Front": left_front_risk,
        "Left Behind": left_behind_risk,
        "Right Front": right_front_risk,
        "Right Behind": right_behind_risk,
    }
    updated_risks = adjust_risk_with_heading(vehicles, ego_info, risk_values)

    front_risk = updated_risks["Front"]
    behind_risk = updated_risks["Behind"]
    left_front_risk = updated_risks["Left Front"]
    left_behind_risk = updated_risks["Left Behind"]
    right_front_risk = updated_risks["Right Front"]
    right_behind_risk = updated_risks["Right Behind"]


    left_risk = left_front_risk + left_behind_risk
    if left_risk > 1:
        left_risk = 1
    right_risk = right_front_risk + right_behind_risk
    if right_risk > 1:
        right_risk = 1


    if env_scenario.ego.lane_index[2] == 0: 
        left_risk = 1
    if env_scenario.ego.lane_index[2] == len(env_scenario.network.all_side_lanes(env_scenario.ego.lane_index)) - 1: 
        right_risk = 1

    ego_pos_risk = calculate_ego_pos_risk(env_scenario, env_scenario.ego, front_vehicle, behind_vehicle)
    #print(f"ego_pos_risk: {ego_pos_risk}")

    if front_vehicle and front_vehicle['speed'] < 1:
        print(f"Warning: Front vehicle is stationary, speed: {front_vehicle['speed']}")
        ego_pos_risk = 1.0


    ego_pos_risk, crash_prompt, abnormalLeftRisk, abnormalRightRisk = update_ego_pos_risk_road_crashed(env_scenario, env_scenario.ego, front_vehicle, behind_vehicle, ego_pos_risk)
    print(f"After update_ego_pos_risk_road_crashed: outside, ego_pos_risk: {ego_pos_risk} \n")

    prompt = ""   # set the return prompt
    prompt_left_risk = left_risk
    prompt_right_risk = right_risk

    return_crash_happened = False

    if crash_prompt != "":
        return_crash_happened = True
        prompt = crash_prompt + "The driving Risk Values around you are provided as below:\n"
        if abnormalLeftRisk > 0:
            prompt_left_risk = abnormalLeftRisk 
        if abnormalRightRisk > 0:
            prompt_right_risk = abnormalRightRisk 
        prompt += f"(Overall Left Risk Value, {prompt_left_risk:.2f}), (Overall Right Risk Value, {prompt_right_risk:.2f}), (Front Risk Value, {front_risk:.2f}), (Behind Risk Value, {behind_risk:.2f}). "
        prompt += "The Risk Value is a float number from 0.00 to 1.00. The Value 0.00 means no risk, and 1.00 means the highest risk. "
        prompt += "You should consider your action and do reasoning, based on the scenario descriptions together with the Left, Right, Front, Behind Risk Values provided to you."
    else:
        prompt = "\nThe driving Risk Values around you are provided as below:\n"
        prompt += f"(Overall Left Risk Value, {left_risk:.2f}), (Overall Right Risk Value, {right_risk:.2f}), (Front Risk Value, {front_risk:.2f}), (Behind Risk Value, {behind_risk:.2f}). "
        prompt += "The Risk Value is a float number from 0.00 to 1.00. The Value 0.00 means no risk, and 1.00 means the highest risk. "
        prompt += "You should consider your action and do reasoning, based on the scenario descriptions together with the Left, Right, Front, Behind Risk Values provided to you."


    ego_posL_risk = calculate_ego_pos_risk(env_scenario, env_scenario.ego, left_front_vehicle, left_behind_vehicle)
    #print(f"ego_posL_risk: {ego_posL_risk}")
    ego_posR_risk = calculate_ego_pos_risk(env_scenario, env_scenario.ego, right_front_vehicle, right_behind_vehicle)
    #print(f"ego_posR_risk: {ego_posR_risk}")
    ego_posLL_risk = calculate_ego_pos_risk(env_scenario, env_scenario.ego, left_left_front_vehicle, left_left_behind_vehicle)
    #print(f"ego_posLL_risk: {ego_posLL_risk}")
    ego_posRR_risk = calculate_ego_pos_risk(env_scenario, env_scenario.ego, right_right_front_vehicle, right_right_behind_vehicle)
    #print(f"ego_posRR_risk: {ego_posRR_risk}")


    risk_vec4 = np.array([
        round(front_risk, 2),
        round(behind_risk, 2),
        round(left_risk, 2),
        round(right_risk, 2)
    ], dtype=np.float32)

    # construct risk_pattern, values in 0-1
    
    risk_pattern = np.array([
        [0.67, round(ego_posLL_risk, 2), 0.67],
        [round(left_behind_risk, 2), round(ego_posL_risk, 2), round(left_front_risk, 2)],
        [round(behind_risk, 2), round(ego_pos_risk, 2), round(front_risk, 2)],
        [round(right_behind_risk, 2), round(ego_posR_risk, 2), round(right_front_risk, 2)],
        [0.67, round(ego_posRR_risk, 2), 0.67]
    ], dtype=np.float32)


    if env_scenario.ego.lane_index[2] == 0:  
        #print("Ego is in the leftmost lane.")
        risk_pattern[0, 0] = 1.00
        risk_pattern[0, 1] = 1.00
        risk_pattern[0, 2] = 1.00
        risk_pattern[1, 0] = 1.00
        risk_pattern[1, 1] = 1.00
        risk_pattern[1, 2] = 1.00


    if env_scenario.ego.lane_index[2] == 1:  
        #print("Ego is in the left 2nd lane.")
        risk_pattern[0, 0] = 1.00
        risk_pattern[0, 1] = 1.00
        risk_pattern[0, 2] = 1.00


    if env_scenario.ego.lane_index[2] == len(env_scenario.network.all_side_lanes(env_scenario.ego.lane_index)) - 1:
        #print("Ego is in the rightmost lane.")
        risk_pattern[4, 0] = 1.00
        risk_pattern[4, 1] = 1.00
        risk_pattern[4, 2] = 1.00
        risk_pattern[3, 0] = 1.00
        risk_pattern[3, 1] = 1.00
        risk_pattern[3, 2] = 1.00
    

    if env_scenario.ego.lane_index[2] == len(env_scenario.network.all_side_lanes(env_scenario.ego.lane_index)) - 2:
        #print("Ego is in the right 2nd lane.")
        risk_pattern[4, 0] = 1.00
        risk_pattern[4, 1] = 1.00
        risk_pattern[4, 2] = 1.00

    
    # (optional) quantise to 4 bins
    edges = np.array([0.34, 0.67, 1.00], dtype=np.float32) #
    # edges = np.array([0.25, 0.50, 0.75], dtype=np.float32) # 
    quantised = np.digitize(risk_pattern, edges).astype(np.float32)   # 0,1,2,3
    risk_pattern_vec = quantised.flatten()          # shape (15,)
    #print(f"(Overall Left Risk Value, {left_risk:.2f}), (Overall Right Risk Value, {right_risk:.2f}), (Front Risk Value, {front_risk:.2f}), (Behind Risk Value, {behind_risk:.2f}). ")
    print(f"original risk_pattern:\n {risk_pattern}")
    print(f"quantised: \n{quantised}")
    # print(f"risk_pattern_vec: {risk_pattern_vec}")

    return prompt, risk_vec4, risk_pattern_vec, return_crash_happened

def isRiskPatternSymmetric(risk_pattern: np.ndarray) -> bool:

    left_side = risk_pattern[:, 0]  
    right_side = risk_pattern[:, 2]  


    is_symmetric = np.array_equal(left_side, right_side)
    
    return is_symmetric

def isFlatListedRiskPatternSymmetric(risk_pattern: List[int]) -> bool:

    if len(risk_pattern) % 3 != 0:
        raise ValueError("The flattened risk pattern list length must be a multiple of 3.")

    if len(risk_pattern) != 15:
        raise ValueError("The flattened risk pattern list must contain exactly 15 elements.")


    rows = len(risk_pattern) // 3  

 
    reshaped_pattern = np.array(risk_pattern, dtype=int).reshape(5, 3)


    if (
        not np.array_equal(reshaped_pattern[0], reshaped_pattern[4]) or  
        not np.array_equal(reshaped_pattern[1], reshaped_pattern[3])    
    ):
        return False

    return True

def axialSymmetryTransformRiskVec4(risk_vec4: List[float]) -> List[float]:

    if len(risk_vec4) != 4:
        raise ValueError("The input risk_vec4 must contain exactly 4 elements.")


    risk_vec4[2], risk_vec4[3] = risk_vec4[3], risk_vec4[2]
    return risk_vec4

def axialSymmetryTransformRiskPatternList(risk_pattern: List[int]) -> List[int]:


    if len(risk_pattern) != 15:
        raise ValueError("The input risk_pattern must contain exactly 15 elements.")

    transformed_pattern = risk_pattern[:]

    swap_positions = [
        (0, 12),
        (1, 13),
        (2, 14),
        (3, 9),
        (4, 10),
        (5, 11),
    ]
    for pos1, pos2 in swap_positions:
        transformed_pattern[pos1], transformed_pattern[pos2] = transformed_pattern[pos2], transformed_pattern[pos1]

    return transformed_pattern

def axialSymmetryTransformActionId(action_id: int) -> int:

    if action_id not in [0, 1, 2, 3, 4]:
        raise ValueError("The input action_id must be one of [0, 1, 2, 3, 4].")


    if action_id == 0:
        return 2
    elif action_id == 2:
        return 0
    else:
        return action_id