import argparse
import math
import time
import os

import numpy as np
import pandas as pd
import yaml

from respond.agent.driverAgent import DriverAgent
from respond.highd.highD_helpers import (
    NormalizedLaneInfo,
    get4Risk,
    getLeftRightValid,
)
from run_RESPOND import setup_env

ACTIONS_DESCRIPTION = {
    0: "Turn-left - change lane to the left of the current lane",
    1: "IDLE - remain in the current lane with current speed",
    2: "Turn-right - change lane to the right of the current lane",
    3: "Acceleration - accelerate the vehicle",
    4: "Deceleration - decelerate the vehicle",
}
# dec: -1: turn left, 1: turn right, -2: decelerate, 2: accelerate, 0: idle
dilu_act_2_highd_act = np.array([-1, 0, 1, 2, -2])


def square(x):
    return x * x


def get_valid_actions_description(keep_left, keep_right):
    avaliableActionDescription = "Your available actions are: \n"
    availableActions = []
    if keep_left:
        availableActions.append(0)
    availableActions.append(1)
    if keep_right:
        availableActions.append(2)
    availableActions = availableActions + [3, 4]

    for action in availableActions:
        avaliableActionDescription += (
            ACTIONS_DESCRIPTION[action] + " Action_id: " + str(action) + "\n"
        )
    return avaliableActionDescription


# x,y,xVelocity,yVelocity,xAcceleration,yAcceleration
def get_road_description(ego_car, numLanes, egoLaneRank):
    if numLanes == 1:
        roadCondition = (
            "You are driving on a road with only one lane, you can't change lane. "
        )
    else:
        if egoLaneRank == 0:
            roadCondition = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the leftmost lane. "
        elif egoLaneRank == numLanes - 1:
            roadCondition = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the rightmost lane. "
        else:
            laneRankDict = {
                1: "second",
                2: "third",
                3: "fourth",
                4: "fifth",
                5: "sixth",
                6: "seventh",
            }
            roadCondition = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the {laneRankDict[egoLaneRank]} lane from the left. "

    speed = math.sqrt(square(ego_car.xVelocity) + square(ego_car.yVelocity))
    acce = math.sqrt(square(ego_car.xAcceleration) + square(ego_car.yAcceleration))
    lanePos = math.sqrt(square(ego_car.x) + square(ego_car.y))
    max_pos = 415  # highD dataset road width approximation
    if ego_car.xVelocity < 0:
        lanePos = max_pos - lanePos
    roadCondition += f"Your current position is `({ego_car.x:.2f}, {ego_car.y:.2f})`, speed is {speed:.2f} m/s, acceleration is {acce:.2f} m/s^2, and lane position is {lanePos:.2f} m.\n"
    return roadCondition


# precedingRisk,followingRisk,leftPrecedingRisk,leftAlongsideRisk,leftFollowingRisk,rightPrecedingRisk,rightAlongsideRisk,rightFollowingRisk
def getRiskDesc(ego_car, keep_left: bool, keep_right: bool) -> str:
    front_risk, behind_risk, left_risk, right_risk = get4Risk(
        ego_car, keep_left, keep_right
    )
    risk_description = f"\nThe driving Risk Values around you are provided as below:\n(Overall Left Risk Value, {left_risk:.2f}), (Overall Right Risk Value, {right_risk:.2f}), (Front Risk Value, {front_risk:.2f}), (Behind Risk Value, {behind_risk:.2f}). The Risk Value is a float number from 0.00 to 1.00. The Value 0.00 means no risk, and 1.00 means the highest risk. You should consider your action and do reasoning, based on the scenario descriptions together with the Left, Right, Front, Behind Risk Values provided to you."
    return risk_description


# surrounding_count,precedingId,followingId,leftPrecedingId,leftAlongsideId,leftFollowingId,rightPrecedingId,rightAlongsideId,rightFollowingId
def get_surrounding_cars_description(ego_car):
    if ego_car.surrounding_count == 0:
        SVDescription = "There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n"
    else:
        SVDescription = "There are other vehicles driving around you, and below is their basic information:\n"
        if ego_car.precedingId > 0:
            SVDescription += f"- Vehicle `{ego_car.precedingId}` is driving on the same lane as you and is ahead of you. "
        if ego_car.followingId > 0:
            SVDescription += f"- Vehicle `{ego_car.followingId}` is driving on the same lane as you and is behind of you. "
        if ego_car.leftPrecedingId > 0:
            SVDescription += f"- Vehicle `{ego_car.leftPrecedingId}` is driving on the lane to your left and is ahead of you. "
        if ego_car.leftFollowingId > 0:
            SVDescription += f"- Vehicle `{ego_car.leftFollowingId}` is driving on the lane to your left and is behind of you. "
        if ego_car.rightPrecedingId > 0:
            SVDescription += f"- Vehicle `{ego_car.rightPrecedingId}` is driving on the lane to your left and is ahead of you. "
        if ego_car.rightFollowingId > 0:
            SVDescription += f"- Vehicle `{ego_car.rightFollowingId}` is driving on the lane to your left and is behind of you. "
    return SVDescription


# lane_id, avg_x_speed
# x,y,xVelocity,yVelocity,xAcceleration,yAcceleration
# surrounding_count,precedingId,followingId,leftPrecedingId,leftAlongsideId,leftFollowingId,rightPrecedingId,rightAlongsideId,rightFollowingId
# precedingRisk,followingRisk,leftPrecedingRisk,leftAlongsideRisk,leftFollowingRisk,rightPrecedingRisk,rightAlongsideRisk,rightFollowingRisk
def prepare_prompt(ego_car, lane_info: NormalizedLaneInfo):
    keep_left, keep_right, num_lanes, norm_lane_idx, _, _ = getLeftRightValid(
        ego_car, lane_info
    )

    avaliableActionDescription = get_valid_actions_description(keep_left, keep_right)

    road_speed_level = abs(ego_car.avg_x_speed)
    driving_intensions_with_risk_in_mind = f"Try to keep your front and behind risk value under certain level, for example, try to keep the Max of front risk, behind risk under 0.75, and then make your forward speed close to the current average foward speed of all nearby vehicles: {road_speed_level:.2f}"
    # driving_intensions="Drive safely and avoid collisons"

    roadCondition = get_road_description(ego_car, num_lanes, norm_lane_idx)

    risk_description = getRiskDesc(ego_car, keep_left, keep_right)

    SVDescription = get_surrounding_cars_description(ego_car)

    scenario_description = roadCondition + SVDescription + risk_description
    return (
        scenario_description,
        driving_intensions_with_risk_in_mind,
        avaliableActionDescription,
    )


def llm_dec(data_path: str, fn: str, da: DriverAgent, output_file: str):
    prepare_df = pd.read_csv(fn, dtype={"iRecord": str})
    llm_dec_results = []
    count = 0
    same_count = 0
    for ego_car in prepare_df.itertuples():
        print(f"# {count}")
        count += 1
        iRecord = ego_car.iRecord
        meta_file_path = os.path.join(data_path, f"{iRecord}_recordingMeta.csv")
        recording_meta = pd.read_csv(meta_file_path)
        lane_info = NormalizedLaneInfo.by_mgr(recording_meta)
        scenario_description, driving_intensions, available_actions = prepare_prompt(
            ego_car, lane_info
        )
        result, response_content, human_message = da.zero_shot_decision(
            scenario_description=scenario_description,
            driving_intensions=driving_intensions,
            available_actions=available_actions,
        )
        llm_dec = dilu_act_2_highd_act[result] if (result >= 0 and result < 5) else -4
        risk_result = {}
        risk_result["result"] = result
        risk_result["llm_dec"] = llm_dec
        risk_result["response"] = response_content
        risk_result["prompt"] = human_message
        llm_dec_results.append(risk_result)
        if llm_dec == ego_car.dec:
            same_count += 1
            print(f"same decision, rate: {(same_count / count):2f}")
        else:
            print(f"different desision: {llm_dec} vs {ego_car.dec}")

    print(f"final consistent rate: {(same_count / count):2f}")
    results_df = pd.DataFrame(llm_dec_results)
    cmp_result_df = pd.concat([prepare_df, results_df], axis=1)
    cmp_result_df.to_csv(output_file, index=False)
    print(f"{output_file} is written")


def parse_args():
    parser = argparse.ArgumentParser(description="RESPONSE decision making using LLM")
    parser.add_argument(
        "--preprocessing_dir",
        type=str,
        default="../preprocessed_highD",
        help="Path to preprocessed CSV file",
        required=True,
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="dangerous_turns_ttc_4.0_cont_0.04.csv",
        help="input data filename",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="llm_dec_ttc_4.0.csv",
        help="output filename",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/respond.yaml",
        help="config filename",
    )
    return parser.parse_args()


# python -m respond.highd.llm_decision --preprocessing_dir=<path_to_preprocessed_data> --filename=dangerous_turns_ttc_4.0_cont_0.04.csv --output_file=llm_dec_ttc_4.0.csv
if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()

    data_path = args.preprocessing_dir
    fn = args.filename
    output_file = args.output_file

    sce = None

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    env_config = setup_env(config)

    llm_dec(data_path, fn, DriverAgent(sce, verbose=True), output_file)

    print(f"\nTotal elapsed time: {time.time() - start_time:.2f}s")
