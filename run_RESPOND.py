"""
RESPOND runner.

This file is adapted from DiLu's `run_dilu.py` (PJLab-ADG/DiLu, Apache-2.0),
with substantial modifications for RESPOND (risk field / risk patterns / L1 & L2 risk memory / reflection based on 2 layer risk memory).

Modifications Copyright 2025 Dan Chen.
"""
import os
import sys
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DILU_ROOT = os.path.join(REPO_ROOT, "third_party", "dilu")

if DILU_ROOT not in sys.path:
    sys.path.insert(0, DILU_ROOT)

import copy
import random
import numpy as np
import yaml
from rich import print

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from respond.scenario.envScenario import EnvScenario
from respond.scenario.envScenario import LOOK_SURROUND_V_NUM
from respond.agent.driverAgent import DriverAgent
from respond.agent.risk_pattern_reflection_agent import RiskPatternReflectionAgent
from dilu.driver_agent.vectorStore import DrivingMemory         
from dilu.driver_agent.reflectionAgent import ReflectionAgent   

from respond.risk_pattern.riskDb import RiskMemory
from respond.risk_pattern.riskL2Db import DBLevel2RiskPattern
from respond.risk_pattern.risk_calculator import (
    getEgoRiskLevel, 
    isRiskPatternSymmetric,
    isFlatListedRiskPatternSymmetric,
    axialSymmetryTransformRiskVec4,
    axialSymmetryTransformRiskPatternList,
    axialSymmetryTransformActionId,
)


from respond.scenario.utils import change_lane_solution, remove_actions_from_string

import torch
from typing import List, Tuple

# Keep the same as DiLu
test_list_seed = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31]

def setup_env(config):
    if config['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
        os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
        os.environ["OPENAI_REFLECTION_MODEL"] = config['OPENAI_REFLECTION_MODEL']
        os.environ["OPENAI_BASE_URL"] = config['OPENAI_BASE_URL']
    else:
        raise ValueError("Unknown OPENAI_API_TYPE, should be azure or openai")

    # environment setting: keep the same as DiLu
    env_config = {
        'highway-v0':
        {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": config["vehicle_count"],
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(5, 32, 9),
            },
            "lanes_count": config["lanes_count"],
            "other_vehicles_type": config["other_vehicle_type"],
            "duration": config["simulation_duration"],
            "vehicles_density": config["vehicles_density"],
            "show_trajectories": True,
            "render_agent": True,
            "scaling": 4,
            'initial_lane_id': None,
            "ego_spacing": 4,
        }
    }

    return env_config

def get_action_from_mem(
    exact_matched_memory: dict,
) -> int:
    """
    根据精确匹配的风险记忆记录，决定下一步的行动。

    参数:
        exact_matched_memory: 精确匹配的风险记忆记录（从数据库中查到的）。

    返回:
        action: 对应pattern中，做出行动次数最多的action ID。
    """
    print("[cyan]Determining action from exact matched memory...[/cyan]")

    action_cnt = exact_matched_memory.get("action_cnt", [0, 0, 0, 0, 0])  
    print("[cyan]Action count from memory: [/cyan]", action_cnt)
    print("the (last time) action_id in the memory item: ", exact_matched_memory.get("action_id", -1))
    print("the count attribute in memory item: ", exact_matched_memory.get("count", -1))

    if all(cnt == 0 for cnt in action_cnt):
        print("[red]Error: action_cnt is all zeros. This indicates no actions have been recorded for the matched memory.[/red]")
        return -1  

    max_action_count = max(action_cnt)

    max_action_ids = [idx for idx, count in enumerate(action_cnt) if count == max_action_count]
    print("[yellow]Actions with maximum count:[/yellow]", max_action_ids)

    if len(max_action_ids) > 1:
        print("[yellow]Warning: Multiple actions have the same maximum count. Selecting the first one.[/yellow]")
        if 1 in max_action_ids:  
            print("[green]IDLE action found. Selecting IDLE (index 1).[/green]")
            max_action_id = 1
            return max_action_id  
        elif 3 in max_action_ids:  
            print("[green]Accelerate action found. Selecting Accelerate (index 3).[/green]")
            max_action_id = 3
            return max_action_id  
        else:  
            max_action_id = max_action_ids[-1]
            print(f"[yellow]No IDLE or Accelerate action found. Selecting the last action with maximum count: {max_action_id}[/yellow]")
            return max_action_id  
    else: 
        max_action_id = max_action_ids[0]
        print(f"[green]Single action with maximum count: {max_action_id}[/green]")

    return max_action_id

def action_id_to_action_str(action_id: int) -> str:
    """
    将 action_id 转换为对应的字符串描述。

    参数:
        action_id: int，动作 ID，取值范围为 0 到 4。

    返回:
        str: 动作的字符串描述。
    """
    action_mapping = {
        0: "Turn-left",
        1: "IDLE",
        2: "Turn-right",
        3: "Acceleration",
        4: "Deceleration",
    }

    if action_id not in action_mapping:
        raise ValueError(f"Invalid action_id: {action_id}. Must be between 0 and 4.")

    return action_mapping[action_id]


def addL2FrontorBehind(
    dbl2: DBLevel2RiskPattern,
    risk_pattern: List[int],
    sub_pattern: int
) -> None:

    if len(risk_pattern) != 15:
        print("[red]Error: in addL2FrontorBehind method: risk_pattern must be a list of 15 integers.[/red]")
        return

    if sub_pattern == 3:  
        level2_pattern = int(f"3{risk_pattern[7]}{risk_pattern[8]}")  
        dbl2.addLevel2RiskPattern(
            level2_pattern=level2_pattern,
            risk_pattern=str(risk_pattern),  
            action_id=None,  
            dont_action_id_1=None,
            dont_action_id_2=None,
            dont_action_id_3=None,
            sub_pattern=sub_pattern,  
        )
    elif sub_pattern == 4:  
        level2_pattern = int(f"4{risk_pattern[7]}{risk_pattern[6]}") 
        dbl2.addLevel2RiskPattern(
            level2_pattern=level2_pattern,
            risk_pattern=str(risk_pattern),  
            action_id=None,  
            dont_action_id_1=None,
            dont_action_id_2=None,
            dont_action_id_3=None,
            sub_pattern=sub_pattern,  
        )
    else:
        print("[red]Error: Invalid sub_pattern. Must be 3 (Front) or 4 (Behind).[/red]")
    
    return

# just for zero-shot decision with Risk Values in scenario description
def make_zero_shot_decision(
    risk_memory: RiskMemory,
    risk_pattern_vec: np.ndarray,
    ego_state: Tuple[float, float, int],
    scenario_description: str = "",
    available_action: List[int] = [],
    driving_intensions: str = "",
    risk_level: float = 0.0,    
    speed_level: float = 15.0,  
    previous_action: int = 1,   
) -> Tuple[int, str, str, str]:
    
    print("[cyan]Making zero-shot decision without risk memory, with Risk Values in Scenario Description only...[/cyan]")

    action_from = "0_SHOT" 
    
    zero_shot_action, zero_shot_response, human_question = DA.zero_shot_decision(
        scenario_description=scenario_description, available_actions=available_action,
        previous_decisions=action_id_to_action_str(previous_action), 
        driving_intensions=driving_intensions, 
    )
    return zero_shot_action, action_from, zero_shot_response, human_question
    

def make_decision(
    risk_memory: RiskMemory,
    risk_pattern_vec: np.ndarray,
    ego_state: Tuple[float, float, int],
    scenario_description: str = "",
    available_action: List[int] = [],
    driving_intensions: str = "",
    risk_level: float = 0.0,    
    speed_level: float = 15.0,  
    previous_action: int = 1,   
) -> Tuple[int, str, str, str]:

    # print("[cyan]Making decision based on risk memory...[/cyan]")

    action_from = "RISK_MEM"     # 0_SHOT, FEW_SHOT, RISK_MEM_IDLE, RISK_MEM_CONFIDENCE_1.0 or RISK_MEM(should not happen)
    
    if risk_memory.size("ok") < 4:
        action_from = "0_SHOT"
        print("[yellow]Not enough risk memories. Using :[/yellow]" + action_from)
        
    else:
        # print("[green]Sufficient risk memories available for decision making.[/green]")
        similar_memories = risk_memory.retrieve_risk_mem(
            risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
            top_k=3, 
            search_bad=False 
        )

        print("[cyan]The closest memory item is:[/cyan]")
        for idx, memory in enumerate(similar_memories, start=1):
            memory_risk_pattern = memory.get("risk_pattern", [])
            memory_action_id = memory.get("action_id", -1)
            memory_confidence = memory.get("confidence", 0.0)
            memory_distance = memory.get("distance", None)  
            print(f"Memory {idx}:")
            print(f"  Distance: {memory_distance}")
            print(f"  Risk Pattern: {memory_risk_pattern}")
            print(f"  Action ID: {memory_action_id}")
            print(f"  Confidence: {memory_confidence}")
            print("-" * 40)
            break #only print the closest one

        is_exact_matched = False
        exact_matched_memory = None
        for memory in similar_memories:
            # 获取 memory 中的 risk_pattern
            memory_risk_pattern = memory.get("risk_pattern", [])
            memory_distance = memory.get("distance", None)

            # print("memory_risk_pattern: ", memory_risk_pattern)
            # print("risk_pattern_vec: ", risk_pattern_vec.flatten().astype(int).tolist())
            # print("memory_distance: ", memory_distance)

            if memory_distance is not None and memory_distance < 1.0:
                is_exact_matched = True
                exact_matched_memory = memory
                print("[green]Exact match found in risk memory.[/green]")
                break

        if not is_exact_matched:
            print("[yellow]No exact match found in risk memory.[/yellow]")

        if risk_level < 0.75:
            print("[cyan]Risk level is below threshold. Checking for exact match in risk memory...[/cyan]")

            if is_exact_matched:
                print("[green]Exact match found in risk memory and risk level < 0.75[/green]")
                matched_action_id = get_action_from_mem(exact_matched_memory)

                if matched_action_id == 1:  
                    print("[green]Matched action is IDLE. Using IDLE action.[/green]")
                    action_from = action_from + "_IDLE"  
                    return (
                        matched_action_id,
                        action_from,
                        "Response based on RiskDb search, the exact risk pattern is: " + str(risk_pattern_vec.tolist()),
                        "Driving scenario description:\n" + scenario_description +
                        "Driving Intensions(since the decision is based on previous one in db, not consider this yet) :\n" + driving_intensions +
                        "Available Actions:\n" + available_action 
                    )
                else: 
                    # print("[yellow]Matched action is not IDLE. Using Zero Shot decision.[/yellow]. the matched action id is: ", matched_action_id)
                    action_from = "0_SHOT"
            else: 
                print("[yellow]No exact match found in risk memory.[/yellow]")
                action_from = "0_SHOT"  
        else:   # risk_level >= 0.75
            print("[green]Risk level is above threshold.[/green]")
            
            if is_exact_matched:
                print(f"[green]Exact match found in risk memory, in situation that risk level is high: {risk_level}.[/green]")
                
                confidence = exact_matched_memory.get("confidence", 0.0)  
                if confidence == 1.0: 
                    print("[green]Confidence is 1.0. Proceeding with action_id in exact matched memory.[/green]")
                    matched_action_id = exact_matched_memory.get("action_id", -1)  
                    action_from = action_from + "_CONFIDENCE_1.0"  
                    return (
                        matched_action_id,
                        action_from,
                        "Response based on RiskDb search, the exact risk pattern is: " + str(risk_pattern_vec.tolist()),
                        "Driving scenario description:\n" + scenario_description +
                        "Driving Intensions(since the decision is based on previous one in db, not consider this yet) :\n" + driving_intensions +
                        "Available Actions:\n" + available_action  
                    )
                else:

                    print(f"[yellow]Confidence is not 1.0. Use 0_SHOT [/yellow]")
                    action_from = "0_SHOT"  
            else: 
                action_from = "0_SHOT"  

    if( action_from == "0_SHOT"):
        print("[yellow]Action determined from Zero Shot[/yellow]")
        zero_shot_action, zero_shot_response, human_question = DA.zero_shot_decision(
            scenario_description=scenario_description, available_actions=available_action,
            previous_decisions=action_id_to_action_str(previous_action), 
            driving_intensions=driving_intensions, 
        )
        return zero_shot_action, action_from, zero_shot_response, human_question
    elif( action_from == "RISK_MEM"):
        print("[red]: Should not be here. Unknown exception happened[/red]")
    else:
        print("[red]Should not be here. Unknown action_from: ", action_from, "[/red]")
    
    print("[red]Should not be here, at the end or make_decision(...) function.[/red]")
    return -1, action_from, "ERROR", "ERROR" 


def get_human_feedback_and_save(
    risk_l2_memory: DBLevel2RiskPattern,
    risk_vec4: List[float],
    risk_pattern: List[int],
    next_action: int
) -> int:

    if len(risk_pattern) != 15:
        raise ValueError("Risk pattern must be a list of 15 integers.")
    
    if len(risk_vec4) != 4:
        raise ValueError("Risk vector must be a list of 4 floats (front, behind, left, right).")
    
    print("[cyan]Risk Pattern (5x3 Matrix):[/cyan]")
    for i in range(0, len(risk_pattern), 3):
        print(risk_pattern[i:i+3])

    print("[cyan]Risk Vector (front, behind, left, right):[/cyan]", risk_vec4)

    print("[cyan]Next Action ID: [/cyan]", next_action)

    while True:
        try:
            human_feedback = int(input("Please provide feedback (0~4) for the action ID: ").strip())
            if human_feedback < 0 or human_feedback > 5: # use No.5 as ignore human feedback
                raise ValueError("Feedback must be an integer between 0 and 5 (5 means ignore).")
            break
        except ValueError as e:
            print(f"[red]Invalid input: {e}. Please try again.[/red]")

    sub_pattern = None
    level2_pattern = None
    rp = risk_pattern 

    if human_feedback == 5:
        return next_action 

    if human_feedback == 3:
        sub_pattern = 9  
        level2_pattern=int(f"9{rp[8]}")
    
    if human_feedback == 0:
        sub_pattern = 5
        level2_pattern=int(f"5{rp[4]}{rp[5]}")

    if human_feedback == 2:
        sub_pattern = 6
        level2_pattern=int(f"6{rp[10]}{rp[11]}")

    if human_feedback == 4:
        sub_pattern = 10
        level2_pattern=int(f"10{rp[8]}")

    if human_feedback != next_action:
        print(f"Special log for risk values in sporty teaching: front: {risk_vec4[0]} left: {risk_vec4[2]} right: {risk_vec4[3]} action_id: {human_feedback} ")


    if sub_pattern is not None:

        risk_l2_memory.addLevel2RiskPattern(
            level2_pattern=level2_pattern,
            risk_pattern=str(risk_pattern),  
            action_id=human_feedback,  
            dont_action_id_1=None,  
            dont_action_id_2=None,
            dont_action_id_3=None,
            sub_pattern=sub_pattern
        )

    return human_feedback

def make_decision_with_level2(
    risk_memory: RiskMemory,
    risk_l2_memory: DBLevel2RiskPattern,
    risk_vec4: np.ndarray,
    risk_pattern_vec: np.ndarray,
    ego_state: Tuple[float, float, int],
    scenario_description: str = "",
    available_action: List[int] = [],
    driving_intensions: str = "",
    risk_level: float = 0.0,   
    speed_level: float = 15.0, 
    previous_action: int = 1,   
    sporty: bool = False,  
) -> Tuple[int, str, str, str]:

    print("[cyan]Making decision based on risk memory...[/cyan]")

    action_from = "RISK_MEM"     # 0_SHOT, FEW_SHOT, RISK_MEM_IDLE, RISK_MEM_CONFIDENCE_1.0 or RISK_MEM(should not happen)
    updated_available_action = copy.deepcopy(available_action)  

    if risk_memory.size("ok") < 4:  # 
        action_from = "0_SHOT"
        print("[yellow]Not enough risk memories. Using :[/yellow]" + action_from)
        
    else: 
        print("[green]Sufficient risk memories available for decision making.[/green]")

        similar_memories = risk_memory.retrieve_risk_mem(
            risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
            top_k=3,  
            search_bad=False  
        )

        print("[cyan]The closest memory item is:[/cyan]")
        for idx, memory in enumerate(similar_memories, start=1):
            memory_risk_pattern = memory.get("risk_pattern", [])
            memory_action_id = memory.get("action_id", -1)
            memory_confidence = memory.get("confidence", 0.0)
            memory_distance = memory.get("distance", None)  
            print(f"Memory {idx}:")
            print(f"  Distance: {memory_distance}")
            print(f"  Risk Pattern: {memory_risk_pattern}")
            print(f"  Action ID: {memory_action_id}")
            print(f"  Confidence: {memory_confidence}")
            print("-" * 40)
            break #only print the closest one

        is_exact_matched = False
        exact_matched_memory = None
        for memory in similar_memories:
            memory_risk_pattern = memory.get("risk_pattern", [])
            memory_distance = memory.get("distance", None)

            if memory_distance is not None and memory_distance < 1.0:
                is_exact_matched = True
                exact_matched_memory = memory
                # print("[green]Exact match found in risk memory level1.[/green]")
                break

        if not is_exact_matched:
            print("[yellow]No exact match found in risk memory level1.[/yellow]")

        if risk_level < 0.75:

            if sporty == True: 

                is_sporty_matched = risk_l2_memory.sporty_action(
                    risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                    sub_pattern=9,  
                )
                if is_sporty_matched:
                    print("[green]Sporty action accelerate found in risk L2 memory.[/green]")
                    action_from = "RISK_Level2_SPORTY"
                    return (
                        3,  # Accelerate action ID
                        action_from,
                        "Response based on DBLevel2RiskPattern search, the leve2 sub_pattern is Sporty-9, the level1 risk pattern is: " + str(risk_pattern_vec.tolist()),
                        "Driving scenario description:\n" + scenario_description +
                        "Driving Intensions(since the decision is based on previous one in db, not consider this yet) :\n" + driving_intensions +
                        "Available Actions:\n" + available_action  
                    )

                if risk_vec4[2] < risk_vec4[3] :
                    is_sporty_matched = risk_l2_memory.sporty_action(
                        risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                        sub_pattern=5,  
                    )
                    if is_sporty_matched:
                        print("[green]Sporty action turn-left found in risk L2 memory.[/green]")
                        action_from = "RISK_Level2_SPORTY"
                        return (
                            0,  # Accelerate action ID
                            action_from,
                            "Response based on DBLevel2RiskPattern search, the leve2 sub_pattern is Sporty-5, the level1 risk pattern is: " + str(risk_pattern_vec.tolist()),
                            "Driving scenario description:\n" + scenario_description +
                            "Driving Intensions(since the decision is based on previous one in db, not consider this yet) :\n" + driving_intensions +
                            "Available Actions:\n" + available_action  
                        )
                
                if risk_vec4[3] >= risk_vec4[2]:  
                    is_sporty_matched = risk_l2_memory.sporty_action(
                        risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                        sub_pattern=6,  
                    )
                    if is_sporty_matched:
                        print("[green]Sporty action turn-right found in risk L2 memory.[/green]")
                        action_from = "RISK_Level2_SPORTY"
                        return (
                            2,  # Accelerate action ID
                            action_from,
                            "Response based on DBLevel2RiskPattern search, the leve2 sub_pattern is Sporty-6, the level1 risk pattern is: " + str(risk_pattern_vec.tolist()),
                            "Driving scenario description:\n" + scenario_description +
                            "Driving Intensions(since the decision is based on previous one in db, not consider this yet) :\n" + driving_intensions +
                            "Available Actions:\n" + available_action  
                        )

            # print("[cyan]Risk level is below threshold. Checking for exact match in risk memory...[/cyan]")

            if is_exact_matched:
                print("[green]Exact match found in risk memory and risk level < 0.75[/green]")
                matched_action_id = get_action_from_mem(exact_matched_memory)

                if matched_action_id == 1:  # IDLE 的 action_id
                    print("[green]Matched action is IDLE. Using IDLE action.[/green]")
                    action_from = action_from + "_IDLE"  
                    return (
                        matched_action_id,
                        action_from,
                        "Response based on RiskDb search, the exact risk pattern is: " + str(risk_pattern_vec.tolist()),
                        "Driving scenario description:\n" + scenario_description +
                        "Driving Intensions(since the decision is based on previous one in db, not consider this yet) :\n" + driving_intensions +
                        "Available Actions:\n" + available_action 
                    )
                else:
                    print("[yellow]Matched action is not IDLE.[/yellow]. the matched action id is: ", matched_action_id)
                    is_updated, updated_available_action, removed_actions = risk_l2_memory.should_not_do_which_action(
                        risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                        current_available_actions_str=updated_available_action                 
                    )
                    if is_updated:
                        # print("[yellow]risk_level < 0.75, use FEW_SHOT (place #1), to tell LLM which actions should not do.[/yellow]")
                        action_from = "FEW_SHOT"  
                        # print("[yellow]Updated available actions based on level 2 risk pattern.[/yellow]")
                        # print("[cyan]Removed actions: ", removed_actions, "[/cyan]")
                        # print("previous available actions: ", available_action)
                        # print("[cyan]Updated available actions: ", updated_available_action, "[/cyan]")
                    else:
                        action_from = "0_SHOT"
            else: 
                is_updated, updated_available_action, removed_actions = risk_l2_memory.should_not_do_which_action(
                    risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                    current_available_actions_str=updated_available_action                 
                )
                if is_updated:
                    # print("[yellow]risk_level < 0.75, use FEW_SHOT (place #2), to tell LLM which actions should not do.[/yellow]")
                    action_from = "FEW_SHOT"  
                    # print("[yellow]Updated available actions based on level 2 risk pattern.[/yellow]")
                    # print("[cyan]Removed actions: ", removed_actions, "[/cyan]")
                    # print("previous available actions: ", available_action)
                    # print("[cyan]Updated available actions: ", updated_available_action, "[/cyan]")
                else:
                    action_from = "0_SHOT"
                
        else:   # risk_level >= 0.75
            print("[green]Risk level is above threshold.[/green]")
            
            if is_exact_matched:
                print(f"[green]Exact match found in risk memory, in situation that risk level is high: {risk_level}.[/green]")
                
                confidence = exact_matched_memory.get("confidence", 0.0)  
                if confidence == 1.0: 
                    print("[green]Confidence is 1.0. Proceeding with action_id in exact matched memory.[/green]")
                    matched_action_id = exact_matched_memory.get("action_id", -1)  
                    action_from = action_from + "_CONFIDENCE_1.0"  
                    return (
                        matched_action_id,
                        action_from,
                        "Response based on RiskDb search, the exact risk pattern is: " + str(risk_pattern_vec.tolist()),
                        "Driving scenario description:\n" + scenario_description +
                        "Driving Intensions(since the decision is based on previous one in db, not consider this yet) :\n" + driving_intensions +
                        "Available Actions:\n" + available_action  
                    )
                else:
                    print(f"[yellow]Level1 exact_matched, but Confidence is not 1.0. go on level2 matching [/yellow]")
                    # set is_exact_matched to False, to use below level2 matching
                    is_exact_matched = False
            
            if not is_exact_matched: 
                if risk_vec4[0] >= risk_vec4[1]: # and risk_vec4[0] >= 0.75:   
                    print("[yellow]Risk level is high, but no exact match found. Try using level 2 risk pattern (sub_pattern 3) to determine actions.[/yellow]")
                    
                    is_matched = risk_l2_memory.should_do_change_lane(
                        risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                        sub_pattern = 3,  
                    )
                    if is_matched:
                        print("[green]Matched action found in level 2 risk pattern.[/green]")
                        solution_count, changed_lane_action_id = change_lane_solution(
                            risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                        )
                        if solution_count == 1:
                            action_from = "RISK_Level2"
                            print("decided to change lane by level2 risk pattern, the changed lane action id is: ", changed_lane_action_id)
                            return (
                                changed_lane_action_id,
                                action_from,
                                "Response based on DBLevel2RiskPattern search, the leve2 sub_pattern is FRONT-3, the level1 risk pattern is: " + str(risk_pattern_vec.tolist()),
                                "Driving scenario description:\n" + scenario_description +
                                "Driving Intensions(since the decision is based on previous one in db, not consider this yet) :\n" + driving_intensions +
                                "Available Actions:\n" + available_action  
                            )
                        elif solution_count == 0:  
                            print("[red] level2 risk pattern indicated that should changed lane, but no solution found![/red]")
                            print("[green]Rule based decision[/green]")
                            action_from = "Rule_Based"
                            if risk_vec4[1] < 0.5:
                                print("[green]Rule based decision, to decelerate, the action id is: [/green]", 4)
                                return (
                                    4,  # Decelerate action ID
                                    action_from,
                                    "Response based on Rule, the level1 risk pattern is: " + str(risk_pattern_vec.tolist()),
                                    "Driving scenario description:\n" + scenario_description +
                                    "Driving Intensions(since the decision is based on Rules, not consider this yet) :\n" + driving_intensions +
                                    "Available Actions:\n" + available_action  
                                )
                            else:
                                print("[red] level2 risk pattern indicated that should changed lane, but no solution found! and indicated to decelerate, but rear risk is high. So use the LLM[/red]")
                                action_from = "0_SHOT"  
                        elif solution_count == 2:
                            print("[yellow] level2 risk pattern indicated that should changed lane, but there are two solutions, need LLM to decide with FEW_SHOT (place #3). [/yellow]")
                            action_from = "FEW_SHOT"
                            updated_available_action = remove_actions_from_string( available_action, [1, 3, 4] )  
                        else:
                            print("[red] level2 risk pattern indicated that should changed lane, but this is unexpected situation![/red]")
                            action_from = "0_SHOT"    
                    else:
                        print("[red]No matched action found in level 2 risk pattern.[/red]")

                        if sporty == True: 
                            is_sporty_matched = risk_l2_memory.sporty_action(
                                risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                                sub_pattern=10,  
                            )
                            if is_sporty_matched:
                                print("[green]Sporty action decelerate found in risk L2 memory.[/green]")
                                action_from = "RISK_Level2_SPORTY"
                                return (
                                    4,  # Decelerate action ID
                                    action_from,
                                    "Response based on DBLevel2RiskPattern search, the leve2 sub_pattern is Sporty-10, the level1 risk pattern is: " + str(risk_pattern_vec.tolist()),
                                    "Driving scenario description:\n" + scenario_description +
                                    "Driving Intensions(since the decision is based on previous one in db, not consider this yet) :\n" + driving_intensions +
                                    "Available Actions:\n" + available_action  
                                )

                        action_from = "0_SHOT"

                elif risk_vec4[0] < risk_vec4[1]: 
                    print("[yellow]Risk level is high, but no exact match found. Try using level 2 risk pattern (sub_pattern 4) to determine actions.[/yellow]")
                    is_matched = risk_l2_memory.should_do_change_lane(
                        risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                        sub_pattern = 4,  
                    )
                    if is_matched:
                        print("[green]Matched action found in level 2 risk pattern.[/green]")
                        solution_count, changed_lane_action_id = change_lane_solution(
                            risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                        )
                        if solution_count == 1:
                            action_from = "Rule_Based"
                            print("decided to change lane by level2 risk pattern, the changed lane action id is: ", changed_lane_action_id)
                            return (
                                changed_lane_action_id,
                                action_from,
                                "Response based on DBLevel2RiskPattern search, the leve2 sub_pattern is FRONT-3, the level1 risk pattern is: " + str(risk_pattern_vec.tolist()),
                                "Driving scenario description:\n" + scenario_description +
                                "Driving Intensions(since the decision is based on previous one in db, not consider this yet) :\n" + driving_intensions +
                                "Available Actions:\n" + available_action  
                            )
                        elif solution_count == 0:
                            print("[red] level2 risk pattern indicated that should changed lane, but no solution found![/red]")
                            print("[green]Rule based decision[/green]")
                            action_from = "Rule_Based"
                            if risk_vec4[0] < 0.33:
                                print("[green]Rule based decision, the action id is: [/green]", 3)
                                return (
                                    3,  # Accelerate action ID
                                    action_from,
                                    "Response based on Rule, the level1 risk pattern is: " + str(risk_pattern_vec.tolist()),
                                    "Driving scenario description:\n" + scenario_description +
                                    "Driving Intensions(since the decision is based on Rules, not consider this yet) :\n" + driving_intensions +
                                    "Available Actions:\n" + available_action  
                                ) 
                            else:
                                print("[red] level2 risk pattern indicated that should changed lane, but no solution found![/red]")
                                action_from = "0_SHOT"  
                        elif solution_count == 2:
                            print("[yellow] level2 risk pattern indicated that should changed lane, but there are two solutions, need LLM to decide with FEW_SHOT (place #4). [/yellow]")
                            action_from = "FEW_SHOT"
                            updated_available_action = remove_actions_from_string( available_action, [1, 3, 4] )
                        else:
                            print("[red] level2 risk pattern indicated that should changed lane, but this is unexpected situation![/red]")
                            action_from = "0_SHOT"    
                    else:
                        print("[red]No matched action found in level 2 risk pattern.[/red]")
                        action_from = "0_SHOT"
                else:   
                    # print(f"[red]both the front & behind risks are higher than 0.75[/red], the front risk: {risk_vec4[0]}, the behind risk: {risk_vec4[1]}")
                    print("[red] should not be here: the risk pattern is :[/red]", str(risk_pattern_vec.flatten().astype(int).tolist()))

    if( action_from == "0_SHOT"):
        print("[yellow]Action determined from Zero Shot[/yellow]")
        zero_shot_action, zero_shot_response, human_question = DA.zero_shot_decision(
            scenario_description=scenario_description, available_actions=available_action,
            previous_decisions=action_id_to_action_str(previous_action),
            driving_intensions=driving_intensions, 
        )
        return zero_shot_action, action_from, zero_shot_response, human_question
    elif( action_from == "FEW_SHOT"):    
        # reminding of the updated available actions
        reminding_str = "\n Finally, remind you one more time, you should only take the action within the available actions given below.\n"

        print("[yellow]Action determined from Few Shot, use Zero Shot with updatd available_action[/yellow]")
        zero_shot_action, zero_shot_response, human_question = DA.zero_shot_decision(
            scenario_description=scenario_description, available_actions=reminding_str + updated_available_action,
            previous_decisions=action_id_to_action_str(previous_action),
            driving_intensions=driving_intensions, 
        )
        return zero_shot_action, action_from, zero_shot_response, human_question
    elif( action_from == "RISK_MEM"):
        print("[red]: Should not be here. Should have been handled already[/red]")
    else:
        print("[red]Should not be here. Unknown action_from: ", action_from, "[/red]")
    
    print("[red]Should not be here, at the end or make_decision_with_level2(...) function.[/red]")
    return -1, action_from, "ERROR", "ERROR" 

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    config = yaml.load(open('configs/respond.yaml'), Loader=yaml.FullLoader)
    env_config = setup_env(config)

    respond_demo_option = config["respond_demo_option"]

    REFLECTION = config["reflection_module"]
    RISK_PATTERN_MEMORY_MODE = config["risk_pattern_memory_mode"]  
    enableRMR = config["risk_memory_refection"]
    SPORTY_TEACHING_MODE = config["sporty_teaching_mode"]  
    memory_path = config["memory_path"]    
    risk_memory_path = config["risk_mem_path"]  

    dbl2_address = config["risk_mem_level2_path"]   
    dbl2 = DBLevel2RiskPattern(dbl2_address)

    few_shot_num = config["few_shot_num"]
    result_folder = config["result_folder"]
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(result_folder + "/" + 'log.txt', 'w') as f:
        f.write("result_folder {} | lanes_count: {} \n".format(
            result_folder, env_config['highway-v0']['lanes_count']))

    agent_memory = DrivingMemory(db_path=memory_path)
    if REFLECTION:
        updated_memory = DrivingMemory(db_path=memory_path + "_updated")
        # updated_memory.combineMemory(agent_memory)

    if RISK_PATTERN_MEMORY_MODE:
        risk_memory = RiskMemory(db_path=risk_memory_path)
        print("[green]Successfully load risk memory module![/green]")

    print('cuda? ',torch.cuda.is_available())
    episode = 0

    if respond_demo_option.lower() == "option1" and RISK_PATTERN_MEMORY_MODE:
        print("[green]Respond demo option 1: use L1+L2 risk memory[/green]")
    elif respond_demo_option.lower() == "option2" and RISK_PATTERN_MEMORY_MODE:
        print("[green]Respond demo option 2: use L1 risk memory only[/green]")
    elif respond_demo_option.lower() == "option3":
        print("[green]Respond demo option 3: use zero-shot decision with risk value in scenario description[/green]")
    elif respond_demo_option.lower() == "option4" and RISK_PATTERN_MEMORY_MODE:
        print("[green]Respond demo option 4: use L1+L2 risk memory and enabled Sporty Mode[/green]")
    else:
        print("[red]Error: should not enable RISK_PATTERN_MEMORY_MODE with this option: ", respond_demo_option, "[/red]")
        sys.exit(1)

    while episode < config["episodes_num"]:
        # setup highway-env
        envType = 'highway-v0'
        env = gym.make(envType, render_mode="rgb_array")
        env.configure(env_config[envType])
        result_prefix = f"highway_{episode}"
        env = RecordVideo(env, result_folder, name_prefix=result_prefix)
        env.unwrapped.set_record_video_wrapper(env)
        seed = random.choice(test_list_seed)  
        obs, info = env.reset(seed=seed)
        env.render()

        frame = env.render()
        if hasattr(env, "video_recorder") and env.video_recorder:
            h, w, *_ = frame.shape
            # print(f"Frame size: {(h,w)} ")

        database_path = result_folder + "/" + result_prefix + ".db"
        
        sce = EnvScenario(env, envType, seed, database_path)
        
        DA = DriverAgent(sce, verbose=True)
        if REFLECTION:
            RA = ReflectionAgent(verbose=True)
        
        if RISK_PATTERN_MEMORY_MODE:
            RPRA = RiskPatternReflectionAgent( sce, verbose=True)
        
        action = "Not available"  
        docs = []
        collision_frame = -1

        next_action = 1 # just give a int value, IDLE, will be replaced by make_decision output
        try:
            already_decision_steps = 0
            ego_speed_stat = 0.0
            ego_risk_level_stat = 0.0

            decision_from = "Not available"
            response = "Not available"  
            human_question = "Not available" 

            risk_pattern_check = None          
            response_check = None
            human_question_check = None        
            last_sce_for_reflection = None      
            last_sce_descrip = "Not available"  
            last_action_id = -1  
            collision_sce_descrip = "Not available"  
            collision_vehicle = None  
            last_drive_intensions = "Not available"  
            previous_action_id_int = 1  
            
            keep_env_str = "Not available"  
            keep_risk_vec4 = [0.0, 0.0, 0.0, 0.0]  
    

            for i in range(0, config["simulation_duration"]):
                frame_img_path = os.path.join(result_folder, f"scene_episode{episode}_step{i:03d}.png")
                sce_copy = copy.deepcopy(sce)
                sce_copy.plotSce2( frame_img_path, sce.getSurroundVehicles(LOOK_SURROUND_V_NUM))

                obs = np.array(obs, dtype=float)

                # print("[cyan]Retreive similar memories...[/cyan]")
                fewshot_results = agent_memory.retriveMemory(
                    sce, i, few_shot_num) if few_shot_num > 0 else []
                fewshot_messages = []
                fewshot_answers = []
                fewshot_actions = []
                
                for fewshot_result in fewshot_results:
                    fewshot_messages.append(
                        fewshot_result["human_question"])
                    fewshot_answers.append(fewshot_result["LLM_response"])
                    fewshot_actions.append(fewshot_result["action"]) 
 
                    mode_action = max(
                        set(fewshot_actions), key=fewshot_actions.count)
                    mode_action_count = fewshot_actions.count(mode_action)
             
                if RISK_PATTERN_MEMORY_MODE:
                    print("[cyan]=============== Risk Pattern Memory Mode ===========================[/cyan]")
                    print("enable RESPOND Risk Pattern Memory")
                    print("[cyan]=====================================================================[/cyan]")

                sce_descrip, ego_speed, risk_vec4, risk_pattern_vec, road_speed_level, crash_happened = sce.describe3(i)     # 引入memory和risk_pattern的版本
                avail_action = sce.availableActionsDescription()

                last_sce_descrip = sce_descrip  

                current_risk_level = max(risk_vec4[0], risk_vec4[1])  
                risk_level_if_crash_happened_for_level2 = 0.0

                ego_speed_stat += ego_speed
                ego_risk_level_stat += current_risk_level

                road_speed_level = round(road_speed_level, 2)
                driving_intensions_with_risk_in_mind = ""
                if crash_happened:  # this will only indicate crash in front place
                    risk_level_if_crash_happened_for_level2 = 1.0
                    driving_intensions_with_risk_in_mind = (
                        "Since there are crashed vehicles in the road, you need to identify which lanes are blocked first. "
                        "and then you need to find chance to make yourself not on the blocked lane.  "
                        "if you don't have chance to change to the lane without blocked due to that lane currently in extreme high risk.  "
                        "then you need to slow down your speed as soon as possible."
                    )
                else:
                    driving_intensions_with_risk_in_mind = (
                        "Try to keep your front & behind risk value under certain level, "
                        "for example, try to keep the front risk and behind risk under 0.75, "
                        "and then make your speed close to the current average speed of all nearby vehicles: "
                        + str(road_speed_level)
                    )

                next_action = 1 # just give a int value, IDLE, will be replaced by make_decision output
                fewshot_answer = "Not available"  

                if RISK_PATTERN_MEMORY_MODE:

                    last_drive_intensions = driving_intensions_with_risk_in_mind  

                    env_type = "highway-v0"  
                    lanes_count = env_config[env_type]["lanes_count"] 
                    vehicles_density = env_config[env_type]["vehicles_density"]  
                    env_str = f"{env_type}-{lanes_count}lane-density{vehicles_density:.1f}"
                    keep_env_str = env_str  
                    # print(env_str)

                if RISK_PATTERN_MEMORY_MODE:
                    
                    risk_pattern_check = risk_pattern_vec   
                    keep_risk_vec4 = risk_vec4.tolist() 

                    if respond_demo_option.lower() == "option1": # use L1+L2 risk memory
                        # print("[green]Respond demo option 1: use L1+L2 risk memory[/green]")
                        if not crash_happened:
                            risk_level_if_crash_happened_for_level2 = current_risk_level
                        next_action, decision_from, response, human_question = make_decision_with_level2(
                            risk_memory=risk_memory,
                            risk_l2_memory=dbl2,
                            risk_vec4=risk_vec4.tolist(),
                            risk_pattern_vec=risk_pattern_vec,
                            ego_state=(sce.ego.speed, sce.ego.action['acceleration'], sce.ego.lane_index[2]),
                            scenario_description=sce_descrip,
                            available_action=avail_action,
                            driving_intensions=driving_intensions_with_risk_in_mind,
                            previous_action=previous_action_id_int,
                            risk_level=risk_level_if_crash_happened_for_level2, # option1 not use current_risk_level,
                            speed_level=road_speed_level,   
                            sporty = False,
                        )
                        
                    elif respond_demo_option.lower() == "option2":  #use Layer 1 risk pattern only
                        # print("[green]Respond demo option 2: use Layer 1 risk pattern only[/green]")
                        next_action, decision_from, response, human_question = make_decision(
                            risk_memory=risk_memory,
                            risk_pattern_vec=risk_pattern_vec,
                            ego_state=(sce.ego.speed, sce.ego.action['acceleration'], sce.ego.lane_index[2]),
                            scenario_description=sce_descrip,
                            available_action=avail_action,
                            driving_intensions=driving_intensions_with_risk_in_mind,
                            previous_action=previous_action_id_int,
                            risk_level=current_risk_level,
                            speed_level=road_speed_level   
                        )
                    elif respond_demo_option.lower() == "option3":  # zero-shot with risk value in senario discription
                        next_action, decision_from, response, human_question = make_zero_shot_decision(
                            risk_memory=None,
                            risk_pattern_vec=risk_pattern_vec,
                            ego_state=(sce.ego.speed, sce.ego.action['acceleration'], sce.ego.lane_index[2]),
                            scenario_description=sce_descrip,
                            available_action=avail_action,
                            driving_intensions=driving_intensions_with_risk_in_mind,
                            previous_action=previous_action_id_int,
                            risk_level=current_risk_level,
                            speed_level=road_speed_level   
                        )
                    elif respond_demo_option.lower() == "option4":  # sporty mode with L1+L2 risk memory
                        if not crash_happened:
                            risk_level_if_crash_happened_for_level2 = current_risk_level
                        next_action, decision_from, response, human_question = make_decision_with_level2(
                            risk_memory=risk_memory,
                            risk_l2_memory=dbl2,
                            risk_vec4=risk_vec4.tolist(),
                            risk_pattern_vec=risk_pattern_vec,
                            ego_state=(sce.ego.speed, sce.ego.action['acceleration'], sce.ego.lane_index[2]),
                            scenario_description=sce_descrip,
                            available_action=avail_action,
                            driving_intensions=driving_intensions_with_risk_in_mind,
                            previous_action=previous_action_id_int,
                            risk_level=risk_level_if_crash_happened_for_level2, # in sporty mode： it means that crash situation, stop sporty style temporarily
                            speed_level=road_speed_level, 
                            sporty = True,
                        )
                    
                    else:
                        print("[red]Error: should not enable RISK_PATTERN_MEMORY_MODE with this option: ", respond_demo_option, "[/red]")
                        break
                    
                    last_action_id = next_action  
                    response_check = response
                    human_question_check = human_question  
                else: # to comply DiLu Legacy
                    # print("[green]Now in the few shot decision mode.[/green]")
                    next_action, response, human_question, fewshot_answer = DA.few_shot_decision(
                        scenario_description=sce_descrip, available_actions=avail_action,
                        previous_decisions=action,
                        fewshot_messages=fewshot_messages,
                        driving_intensions="Drive safely and avoid collisons", # just keep DiLu's original one
                        fewshot_answers=fewshot_answers,
                    )

                if next_action == -1:
                    print("[red]Error: next_action is -1, which means no valid action was determined.[/red]")
                    break
                
                docs.append({
                    "sce_descrip": sce_descrip,
                    "human_question": human_question,
                    "response": response,
                    "action": next_action,
                    "sce": copy.deepcopy(sce)
                })

                last_sce_for_reflection = copy.deepcopy(sce)  

                if SPORTY_TEACHING_MODE:
                    next_action = get_human_feedback_and_save(
                        risk_l2_memory=dbl2,
                        risk_vec4=risk_vec4.tolist(),
                        risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),
                        next_action=next_action
                    )
                    

                obs, reward, done, info, _ = env.step(next_action) 
                previous_action_id_int = next_action 
                already_decision_steps += 1
                print( "[red]already_decision_steps: [/red]",  already_decision_steps )

                sce.promptsCommit(i, None, done, human_question,
                                  fewshot_answer, response)
                env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()

                print("--------------------")

                if RISK_PATTERN_MEMORY_MODE:

                    delta_risk_level = 0.0

                    if done:    
                        print("[red]Risk Pattern Mode: Simulation crash after running steps: [/red] ", i)

                        risk_memory.add_risk_mem (
                            risk_vec4=risk_vec4.tolist(), 
                            risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),  
                            ego_state=(sce.ego.speed, sce.ego.action['acceleration'], sce.ego.lane_index[2]),  
                            drive_intension=driving_intensions_with_risk_in_mind,  
                            action_id=next_action,  
                            outcome="COLLISION", 
                            reflection_text=None,
                            style_tag=None,
                            env=env_str  
                        )

                        collision_frame = i
                        # print("observation: ", obs)
                        # print("reward: ", reward)
                        # print("info: ", info)   
                        print("The last action is: ", next_action)
                        frame_img_path = os.path.join(result_folder, f"scene_episode{episode}_step{i:03d}_final.png")
                        sce_final_copy = copy.deepcopy(sce)
                        collision_sce_descrip, collision_vehicle = sce_final_copy.plotSce2( frame_img_path, sce.getSurroundVehicles(LOOK_SURROUND_V_NUM), isFinal=True)
                        break
                    else:
                        risk_level_after = getEgoRiskLevel( sce, i+100)  # use i+100, just in case enable print out the scene png
                        delta_risk_level = risk_level_after - current_risk_level

                        if risk_level_after < 0.75:
                            print("[green]Risk level after action is below 0.75[/green]")
                            print("The risk level after action is: ", risk_level_after)
                            print("The risk level before action is: ", current_risk_level)

                            risk_memory.add_risk_mem(
                                risk_vec4=risk_vec4.tolist(),  
                                risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),  
                                ego_state=(sce.ego.speed, sce.ego.action['acceleration'], sce.ego.lane_index[2]),  
                                drive_intension=driving_intensions_with_risk_in_mind,  
                                action_id=next_action,  
                                outcome="SAFE",  
                                reflection_text=None,
                                style_tag=None,
                                env=env_str  
                            )
                        else:
                            if delta_risk_level < 0:
                                print("[green]Risk level after action is lower than before, but still above 0.75[/green]")
                                print("The risk level after action is: ", risk_level_after)
                                print("The risk level before action is: ", current_risk_level)

                                risk_memory.add_risk_mem(
                                risk_vec4=risk_vec4.tolist(), 
                                risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),  
                                ego_state=(sce.ego.speed, sce.ego.action['acceleration'], sce.ego.lane_index[2]), 
                                drive_intension=driving_intensions_with_risk_in_mind,  
                                action_id=next_action,  
                                outcome="SAFE",  
                                reflection_text=None,
                                style_tag=None,
                                env=env_str  
                            )
                            else:
                                print("[red]Risk level after action is still above 0.75, and increasing[/red], the over all risk level is: ", risk_level_after)
                                print("The risk level after action is: ", risk_level_after)
                                print("The risk level before action is: ", current_risk_level)

                                risk_memory.add_risk_mem(
                                risk_vec4=risk_vec4.tolist(),
                                risk_pattern=risk_pattern_vec.flatten().astype(int).tolist(),  
                                ego_state=(sce.ego.speed, sce.ego.action['acceleration'], sce.ego.lane_index[2]),  
                                drive_intension=driving_intensions_with_risk_in_mind,  
                                action_id=next_action,  
                                outcome="UNSAFE",  
                                reflection_text=None,
                                style_tag=None,
                                env=env_str  
                            )
                else:  
                    if done:    
                        print("[red]Simulation crash after running steps: [/red] ", i)

                        collision_frame = i
                        # print("observation: ", obs)
                        # print("reward: ", reward)
                        # print("info: ", info)  
                        print("The last action is: ", action)
                        frame_img_path = os.path.join(result_folder, f"scene_episode{episode}_step{i:03d}_final.png")
                        sce_final_copy = copy.deepcopy(sce)
                        collision_sce_descrip, collision_vehicle = sce_final_copy.plotSce2( frame_img_path, sce.getSurroundVehicles(LOOK_SURROUND_V_NUM), isFinal=True)
                        break
                # END： if RISK_PATTERN_MEMORY_MODE:


        finally:
            if next_action == -1:
                print("[red]Error: next_action is -1, which means no valid action was determined.[/red]")

            if already_decision_steps == 0:
                already_decision_steps = 1 # to avoid division by zero

            with open(result_folder + "/" + 'log.txt', 'a') as f:
                f.write(
                    "Simulation {} | Seed {} | Steps: {} | File prefix: {} | Ego Avg Speed {} | ego Avg Risk {} | Respond Options: {} \n".format(episode, seed, already_decision_steps, result_prefix, ego_speed_stat/already_decision_steps, ego_risk_level_stat/already_decision_steps, respond_demo_option))

            print( "[red] Ego Avg Speed: [/red]", ego_speed_stat/already_decision_steps)
            print( "[red] Ego Avg Risk: [/red]", ego_risk_level_stat/already_decision_steps)

            if REFLECTION: # DiLu's reflection module
                print("[yellow]Now running reflection agent...[/yellow]")
                if collision_frame != -1: # End with collision
                    for i in range(collision_frame, -1, -1):
                        if docs[i]["action"] != 4:  # not decelerate 
                            corrected_response = RA.reflection(
                                docs[i]["human_question"], docs[i]["response"])
                            
                            choice = input("[yellow]Do you want to add this new memory item to update memory module? (Y/N): ").strip().upper()
                            if choice == 'Y':
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],  
                                    corrected_response,
                                    docs[i]["action"], 
                                    docs[i]["sce"],
                                    comments="mistake-correction"
                                )
                                print("[green] Successfully add a new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                            else:
                                print("[blue]Ignore this new memory item[/blue]")
                            break
                else:
                    print("[yellow]Do you want to add[/yellow]",len(docs)//5, "[yellow]new memory item to update memory module?[/yellow]",end="")
                    choice = input("(Y/N): ").strip().upper()
                    if choice == 'Y':
                        cnt = 0
                        for i in range(0, len(docs)):
                            if i % 5 == 1:
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    docs[i]["response"],
                                    docs[i]["action"],
                                    docs[i]["sce"],
                                    comments="no-mistake-direct"
                                )
                                cnt +=1
                        print("[green] Successfully add[/green] ",cnt," [green]new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                    else:
                        print("[blue]Ignore these new memory items[/blue]")
            # END DiLu's reflection module
            
            if enableRMR and (RISK_PATTERN_MEMORY_MODE == True and collision_frame != -1): 
                print("[yellow]Now running risk memory reflection module...[/yellow]")

                rp = risk_pattern_check.flatten().astype(int).tolist()

                if last_action_id == 0: 
                    dbl2.addLevel2RiskPattern(
                        level2_pattern=int(f"8{rp[1]}{rp[3]}{rp[4]}{rp[5]}"),
                        risk_pattern=str(rp),  
                        action_id=None,
                        dont_action_id_1=0,
                        dont_action_id_2=None,
                        dont_action_id_3=None,
                        sub_pattern=0,  # left
                    )
                if last_action_id == 2: 
                    dbl2.addLevel2RiskPattern(
                        level2_pattern=int(f"2{rp[13]}{rp[9]}{rp[10]}{rp[11]}"),
                        risk_pattern=str(rp),  
                        action_id=None,
                        dont_action_id_1=2,
                        dont_action_id_2=None,
                        dont_action_id_3=None,
                        sub_pattern=2,  # right
                    )

                rp_copy = copy.deepcopy(rp) # rp_copy is a List of integers, e.g. [0, 1 ... 15]
                if isFlatListedRiskPatternSymmetric(rp_copy):
                    print("[green]The risk pattern is symmetric, no need to update the level2 memory.[/green]")
                else:
                    rp_reverse = axialSymmetryTransformRiskPatternList(rp_copy)  
                    if last_action_id == 0:
                        dbl2.addLevel2RiskPattern(
                            level2_pattern=int(f"2{rp[1]}{rp[3]}{rp[4]}{rp[5]}"),  
                            risk_pattern=str(rp_reverse), 
                            action_id=None,
                            dont_action_id_1=2,
                            dont_action_id_2=None,
                            dont_action_id_3=None,
                            sub_pattern=2,  # right
                        )
                    if last_action_id == 2:
                        dbl2.addLevel2RiskPattern(
                            level2_pattern=int(f"8{rp[13]}{rp[9]}{rp[10]}{rp[11]}"),   
                            risk_pattern=str(rp_reverse), 
                            action_id=None,
                            dont_action_id_1=0,
                            dont_action_id_2=None,
                            dont_action_id_3=None,
                            sub_pattern=0,  # left
                        )

                if "RISK_MEM" in decision_from:
                    # print(f"[green]Decision source includes '{decision_from}'. Proceeding with risk memory reflection module.[/green]")
                    # print("<red>The according risk_pattern is:</red> ", risk_pattern_check)
                    
                    reflection_str, new_action_id = RPRA.reflection_in_risk_pattern_mode(
                        ego_car=last_sce_for_reflection.ego, 
                        previous_scenario=human_question_check, 
                        current_scenario=collision_sce_descrip,
                        action_done=last_action_id,
                        car_crash=collision_vehicle
                    )

                    if( new_action_id == -1):
                        print("[red]Error: new_action_id is -1, which means no valid action was determined in risk memory reflection.[/red]")
                    elif new_action_id >=0 and new_action_id < 5:
                        print("[green]New action ID determined in risk memory reflection: ", new_action_id, "[/green]")
                        if new_action_id == last_action_id:
                            print("[green]New action ID is the same as the last action ID, will not to update memory.[/green]")
                        else:
                            risk_memory.add_risk_mem (
                                risk_vec4=keep_risk_vec4,  
                                risk_pattern=risk_pattern_check.flatten().astype(int).tolist(),  
                                ego_state=(last_sce_for_reflection.ego.speed, last_sce_for_reflection.ego.action['acceleration'], last_sce_for_reflection.ego.lane_index[2]),  
                                drive_intension=last_drive_intensions,  
                                action_id=new_action_id, 
                                outcome="CORRECTED", 
                                reflection_text=reflection_str,  
                                style_tag=None,
                                env=keep_env_str  
                            )

                            if (new_action_id == 0 or new_action_id == 2) and (last_action_id == 1 or last_action_id == 3 or last_action_id == 4): 
                                print("[green]The new action ID is a turn action, the last action ID is not a turn action, will update the level2 risk pattern (1st place).[/green]")
                                
                                if (keep_risk_vec4[0] >= keep_risk_vec4[1]) and ( keep_risk_vec4[0] >= 0.75 ): 
                                    addL2FrontorBehind( dbl2, risk_pattern_check.flatten().astype(int).tolist(), 3) 
                                elif (keep_risk_vec4[0] < keep_risk_vec4[1]) and ( keep_risk_vec4[1] >= 0.75 ): 
                                    addL2FrontorBehind( dbl2, risk_pattern_check.flatten().astype(int).tolist(), 4) 
                                else: 
                                    print("[yellow]The risk level is not high enough to update the level2 risk pattern.[/yellow]")
                            else:
                                print(f"[green]In current action pattern, new_action_id: {new_action_id}, last_action_id: {last_action_id}, will not update the level2 risk pattern front or behind.[/green]")

                            if (isFlatListedRiskPatternSymmetric(risk_pattern_check.flatten().astype(int).tolist())):
                                print("[green]The risk pattern is symmetric, no need to update the memory.[/green]")
                            else:
                                print("[green]The risk pattern is not symmetric, updating the memory (the Symmetrical Pattern).[/green]")
                               
                                risk_memory.add_risk_mem( 
                                    risk_vec4 = axialSymmetryTransformRiskVec4(keep_risk_vec4),  
                                    risk_pattern = axialSymmetryTransformRiskPatternList(risk_pattern_check.flatten().astype(int).tolist()),  
                                    ego_state=(last_sce_for_reflection.ego.speed, last_sce_for_reflection.ego.action['acceleration'], last_sce_for_reflection.ego.lane_index[2]),  
                                    drive_intension =last_drive_intensions, 
                                    action_id = axialSymmetryTransformActionId(new_action_id),  
                                    outcome="CORRECTED",  
                                    reflection_text=reflection_str + "Notice: here is the Symmetrical Version",  
                                    style_tag=None,  
                                    env=keep_env_str, 
                                )

                    else:
                        print("[red]Error: new_action_id is out of range, should be between 0 and 4.[/red]")
                    

                else: 
                    # print(f"[yellow]Decision source from LLM '{decision_from}'. Skipping risk memory reflection module.[/yellow]")
                    # print("<red>The risk pattern is:</red> ", risk_pattern_check)
                    # print("<red>The human question is:</red> ", human_question_check)
                    # print("<red>The response is:</red> ", response_check)

                    reflection_str, new_action_id = RPRA.reflection_in_risk_pattern_mode(
                        ego_car=last_sce_for_reflection.ego, 
                        previous_scenario=human_question_check, 
                        current_scenario=collision_sce_descrip,
                        action_done=last_action_id,
                        car_crash=collision_vehicle
                    )

                    if( new_action_id == -1):
                        print("[red]Error: new_action_id is -1, which means no valid action was determined in risk memory reflection.[/red]")
                    elif new_action_id >=0 and new_action_id < 5:
                        print(f"[green]New action ID determined in risk memory reflection: {new_action_id}[/green]")
                        if new_action_id == last_action_id:
                            print("[green]New action ID is the same as the last action ID, will not to update memory.[/green]")
                            print(f"[red]need human to look into it, why last_ation_id {last_action_id} caused the collision.[/red]")
                        else:

                            risk_memory.add_risk_mem (
                                risk_vec4=keep_risk_vec4,  
                                risk_pattern=risk_pattern_check.flatten().astype(int).tolist(),  
                                ego_state=(last_sce_for_reflection.ego.speed, last_sce_for_reflection.ego.action['acceleration'], last_sce_for_reflection.ego.lane_index[2]),  
                                drive_intension=last_drive_intensions, 
                                action_id=new_action_id,  
                                outcome="CORRECTED", 
                                reflection_text=reflection_str,  
                                style_tag=None,
                                env=keep_env_str  
                            )

                            if (new_action_id == 0 or new_action_id == 2) and (last_action_id == 1 or last_action_id == 3 or last_action_id == 4): 
                                print("[green]The new action ID is a turn action, the last action ID is not a turn action, will update the level2 risk pattern (2nd place).[/green]")
                                
                                if (keep_risk_vec4[0] >= keep_risk_vec4[1]) and ( keep_risk_vec4[0] >= 0.75 ):  
                                    addL2FrontorBehind( dbl2, risk_pattern_check.flatten().astype(int).tolist(), 3)
                                elif (keep_risk_vec4[0] < keep_risk_vec4[1]) and ( keep_risk_vec4[1] >= 0.75 ):  
                                    addL2FrontorBehind( dbl2, risk_pattern_check.flatten().astype(int).tolist(), 4) 
                                else: 
                                    print("[yellow]The risk level is not high enough to update the level2 risk pattern.[/yellow]")
                            else:
                                print(f"[green]In current action pattern, new_action_id: {new_action_id}, last_action_id: {last_action_id}, will not update the level2 risk pattern front or behind.[/green]")


                            if (isFlatListedRiskPatternSymmetric(risk_pattern_check.flatten().astype(int).tolist())):
                                print("[green]The risk pattern is symmetric, no need to update the memory.[/green]")
                            else:

                                print("[green]The risk pattern is not symmetric, updating the memory (the Symmetrical Pattern).[/green]")

                                risk_memory.add_risk_mem( 
                                    risk_vec4 = axialSymmetryTransformRiskVec4(keep_risk_vec4), 
                                    risk_pattern = axialSymmetryTransformRiskPatternList(risk_pattern_check.flatten().astype(int).tolist()),  
                                    ego_state=(last_sce_for_reflection.ego.speed, last_sce_for_reflection.ego.action['acceleration'], last_sce_for_reflection.ego.lane_index[2]),  
                                    drive_intension =last_drive_intensions,  
                                    action_id = axialSymmetryTransformActionId(new_action_id),  
                                    outcome="CORRECTED",  
                                    reflection_text=reflection_str + "Notice: here is the Symmetrical Version",  
                                    style_tag=None,  
                                    env=keep_env_str,  
                                )
                    else:
                        print("[red]Error: new_action_id is out of range, should be between 0 and 4.[/red]")

            
            print("==========Simulation {} Done==========".format(episode))
            episode += 1
            env.close()


