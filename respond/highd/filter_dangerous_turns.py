"""
Filter Dangerous Turns from Preprocessed Data

This script reads preprocessed turn sequence data and applies dangerous turn
filtering logic to identify potentially dangerous lane changes.

Input:
- preprocessing_highD_{iRecord}.csv: All vehicle frames during turn sequences
- {iRecord}_recordingMeta.csv: Recording metadata
- {iRecord}_tracksMeta.csv: Tracks metadata

Output:
- dangerous_turns_ttc_{min_ttc}_cont_{continue_dangerous_frames}.csv
"""

import argparse
import gc
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd

import respond.highd.highD_helpers as helpers


# Configuration parameters (defaults, can be overridden by CLI)
continue_dangerous_frames = 1
min_ttc = 1.0
buffer_time = 2

# Desired output columns (same as dangerous_turn.py)
desired_columns = [
    "iRecord",
    "id",
    "dec",
    "frame_start",
    "frame_end",
    "frame_count",
    "dec_frame",
    "cont_end",
    "lane_id",
    "lane_change_frame",
    "llm_frame",
    "affectedId",
    "affected_ttc",
    "affected_dhw",
    "y_dist",
    "dist",
    "x",
    "y",
    "xVelocity",
    "yVelocity",
    "xAcceleration",
    "yAcceleration",
    "numLaneChanges",
    "precedingId",
    "followingId",
    "leftPrecedingId",
    "leftFollowingId",
    "rightPrecedingId",
    "rightFollowingId",
    "precedingRisk",
    "followingRisk",
    "leftPrecedingRisk",
    "leftFollowingRisk",
    "rightPrecedingRisk",
    "rightFollowingRisk",
    "avg_x_speed",
    "surrounding_count",
    "preceding_x",
    "preceding_y",
    "preceding_xv",
    "following_x",
    "following_y",
    "following_xv",
    "leftPreceding_x",
    "leftPreceding_y",
    "leftPreceding_xv",
    "leftFollowing_x",
    "leftFollowing_y",
    "leftFollowing_xv",
    "rightPreceding_x",
    "rightPreceding_y",
    "rightPreceding_xv",
    "rightFollowing_x",
    "rightFollowing_y",
    "rightFollowing_xv",
    "rp00",
    "rp01",
    "rp02",
    "rp10",
    "rp11",
    "rp12",
    "rp20",
    "rp21",
    "rp22",
    "rp30",
    "rp31",
    "rp32",
    "rp40",
    "rp41",
    "rp42",
]


def load_preprocessing_data(preprocessing_dir: str, iRecord: str) -> pd.DataFrame:
    """Load turn sequences for one recording from preprocessing_highD.csv"""
    csv_path = os.path.join(preprocessing_dir, f"preprocessing_highD_{iRecord}.csv")
    df = pd.read_csv(csv_path)
    return df


def load_recording_meta(data_path: str, iRecord: str) -> pd.DataFrame:
    """Read {iRecord}_recordingMeta.csv directly"""
    file_path = os.path.join(data_path, f"{iRecord}_recordingMeta.csv")
    return pd.read_csv(file_path)


def load_tracks_meta(data_path: str, iRecord: str) -> pd.DataFrame:
    """Read {iRecord}_tracksMeta.csv directly"""
    file_path = os.path.join(data_path, f"{iRecord}_tracksMeta.csv")
    return pd.read_csv(file_path)


def build_data_structures(turn_df: pd.DataFrame):
    """
    Build the same structures as original find_target_turns() needs.

    Returns:
        car_groups: {id -> DataFrame}
        frames_groups: {frame -> DataFrame}
        frame_car_grp: {(frame, id) -> DataFrame}
    """
    car_groups = {car_id: g for car_id, g in turn_df.groupby("id")}
    frames_groups = {frame_id: g for frame_id, g in turn_df.groupby("frame")}
    frame_car_grp = {(frame, id): g for (frame, id), g in turn_df.groupby(["frame", "id"])}
    return car_groups, frames_groups, frame_car_grp


def cal_frame_dist(continue_start, continue_end, lane_change_frame):
    """Calculate distance from lane change frame to continuous dangerous period."""
    if continue_start <= lane_change_frame and lane_change_frame <= continue_end:
        return (0, lane_change_frame - 1)
    if lane_change_frame > continue_end:
        return (lane_change_frame - continue_end, continue_end)
    if continue_start > lane_change_frame:
        return (continue_start - lane_change_frame, continue_start)


def find_target_turns_from_csv(
    car_id: int,
    car_ext_groups: Dict[int, pd.DataFrame],
    target_dec: int,
    tracks_car_groups: Dict[int, pd.DataFrame],
    lane_change_count: int,
    drawer: helpers.FrameDrawer,
    frames_groups: Dict[int, pd.DataFrame],
    iRecord: str,
    frame_car_grp: Dict,
    lane_info: helpers.NormalizedLaneInfo,
    max_frame_id: int,
) -> List:
    """
    Find dangerous turns for a vehicle using preprocessed CSV data.

    This is adapted from dangerous_turn.py find_target_turns() function.
    Key differences:
    - Uses CSV data instead of DataManagerHighD
    - max_frame_id is passed as parameter instead of computed from data

    Returns:
        List of dangerous turn dictionaries with all risk information
    """
    # Get current vehicle's extended data
    car_ext = car_ext_groups[car_id]

    # Mark frames matching target decision
    mask = car_ext["dec"] == target_dec

    # Detect continuous sequences
    seq_groups = mask.ne(mask.shift()).cumsum()
    car_ext.loc[:, "seq"] = seq_groups

    # Group by sequence and compute statistics
    turn_dataframe = (
        car_ext[mask]
        .groupby("seq")
        .agg(
            frame_start=("frame", "min"),
            frame_end=("frame", "max"),
            frame_count=("frame", "count"),
        )
    )

    dangerous_turns = []

    # Check if there are valid turn sequences
    if not turn_dataframe.empty:
        car_tracks = tracks_car_groups[car_id]
        turn_dataframe.reset_index(drop=True)
        max_cont_count = int(25 * continue_dangerous_frames)
        buffer_count = int(25 * buffer_time)

        # Process each turn sequence
        for turn in turn_dataframe.itertuples(index=False):
            # Get frames in this turn sequence
            start_end_frames = car_tracks[
                (car_tracks["frame"] >= turn.frame_start) &
                (car_tracks["frame"] <= turn.frame_end)
            ]

            # Find lane change frames
            lane_mask = start_end_frames["laneId"].ne(
                start_end_frames["laneId"].shift()
            )

            # Segment by lane changes
            lane_seq = start_end_frames[lane_mask]

            # Prepare end frame list
            end_frame_id_list = list(lane_seq["frame"].iloc[2:])
            end_frame_id_list.append(
                turn.frame_end + 1 if turn.frame_end < max_frame_id else turn.frame_end
            )

            # Process each lane change segment
            for first_frame_start, tgt_frame_start, tgt_frame_end in zip(
                lane_seq["frame"].iloc[:-1],
                lane_seq["frame"].iloc[1:],
                end_frame_id_list,
            ):
                ego_tracks = tracks_car_groups[car_id]
                ego_lane_change_frame = ego_tracks[ego_tracks["frame"] == tgt_frame_start - 1]
                ego_lane_id = ego_lane_change_frame["laneId"].item()
                ego_lane_change_frame = ego_tracks[ego_tracks["frame"] == tgt_frame_start]
                target_lane_id = ego_lane_change_frame["laneId"].item()
                valid_lanes = [ego_lane_id, target_lane_id]

                # Collect affected vehicle IDs in target lane
                affected_list = []
                for frame_id in range(first_frame_start, tgt_frame_end):
                    candidate_frame_group = frames_groups[frame_id]
                    candidate_frame_group = candidate_frame_group[
                        candidate_frame_group["laneId"] == target_lane_id
                    ]
                    affected_list.append(candidate_frame_group["id"])

                # Merge and deduplicate
                all_affected = pd.concat(affected_list, ignore_index=True)
                unique_ids = all_affected[all_affected > 0].unique().tolist()

                candidate_list = []

                # Check each affected vehicle for danger
                for candidate_id in unique_ids:
                    min_turn = None
                    continue_count = 0
                    candidate_tracks = tracks_car_groups[candidate_id]

                    for frame_id in range(first_frame_start, tgt_frame_end):
                        # Get affected vehicle data
                        affected_car_frame = candidate_tracks[
                            candidate_tracks["frame"] == frame_id
                        ]

                        if affected_car_frame.empty:
                            if continue_count >= max_cont_count:
                                min_turn["cont_end"] = frame_id - 1
                                candidate_list.append(min_turn)
                            continue_count = 0
                            min_turn = None
                            continue

                        # Convert to vehicle dict
                        affected_car = helpers.convert_to_vehicle_dict(
                            affected_car_frame
                        )

                        # Get ego vehicle data
                        ego_tracks = tracks_car_groups[car_id]
                        ego_frame = ego_tracks[ego_tracks["frame"] == frame_id]
                        ego_car = helpers.convert_to_vehicle_dict(ego_frame)

                        # Skip if affected car is ahead
                        if (
                            affected_car.x + affected_car.width < ego_car.x
                            and affected_car.xVelocity < 0
                        ) or (
                            affected_car.x > ego_car.x + ego_car.width
                            and affected_car.xVelocity > 0
                        ):
                            if continue_count >= max_cont_count:
                                min_turn["cont_end"] = frame_id - 1
                                candidate_list.append(min_turn)
                            continue_count = 0
                            min_turn = None
                            continue

                        # Calculate TTC and distance
                        (
                            (ttc, distance),
                            (ego_attach_x, ego_attach_y),
                            (follow_attach_x, follow_attach_y),
                        ) = helpers.vehicle_min_distance_x_ttc(ego_car, affected_car)

                        # Check TTC threshold
                        x_dhw = abs(ego_attach_x - follow_attach_x)

                        if ttc > 0 and min_ttc > ttc and affected_car.laneId in valid_lanes:
                            if continue_count > 0:
                                continue_count += 1
                                continue
                            if frame_id > tgt_frame_start + buffer_count:
                                continue

                            # Create turn dict
                            turn_dict = turn._asdict()
                            turn_dict["id"] = car_id
                            turn_dict["dec_frame"] = frame_id
                            turn_dict["lane_change_frame"] = tgt_frame_start
                            turn_dict["dec"] = target_dec
                            turn_dict["numLaneChanges"] = lane_change_count
                            turn_dict["affectedId"] = candidate_id
                            turn_dict["affected_ttc"] = ttc
                            turn_dict["affected_dhw"] = x_dhw
                            turn_dict["y_dist"] = abs(ego_attach_y - follow_attach_y)
                            turn_dict["dist"] = distance
                            min_turn = turn_dict
                            continue_count += 1
                        else:
                            if continue_count >= max_cont_count:
                                min_turn["cont_end"] = frame_id - 1
                                candidate_list.append(min_turn)
                            continue_count = 0
                            min_turn = None

                    # Handle final continuous sequence
                    if continue_count >= max_cont_count:
                        min_turn["cont_end"] = tgt_frame_end - 1
                        candidate_list.append(min_turn)

                # Select best dangerous situation
                if len(candidate_list) > 0:
                    min_turn = None
                    nearest_dist = None
                    nearest_frame = None

                    for candidate_turn in candidate_list:
                        lane_change_frame = candidate_turn["lane_change_frame"]
                        affected_id = candidate_turn["affectedId"]
                        affected_car_frame = frame_car_grp.get(
                            (lane_change_frame, affected_id)
                        )

                        if affected_car_frame is None:
                            continue

                        affected_car_lane_id = affected_car_frame["laneId"].item()
                        if affected_car_lane_id != target_lane_id:
                            continue

                        ego_car_frame = frame_car_grp[(lane_change_frame, car_id)]
                        ego_car = helpers.convert_to_vehicle_dict(ego_car_frame)
                        affected_car = helpers.convert_to_vehicle_dict(
                            affected_car_frame
                        )

                        # Validate position
                        if (
                            ego_car.xVelocity < 0
                            and affected_car.x + affected_car.width < ego_car.x
                        ) or (
                            ego_car.xVelocity > 0
                            and ego_car.x + ego_car.width < affected_car.x
                        ):
                            continue

                        continue_end = candidate_turn["cont_end"]
                        continue_start = candidate_turn["dec_frame"]
                        frame_dist, candidate_frame = cal_frame_dist(
                            continue_start, continue_end, lane_change_frame
                        )

                        if (
                            nearest_dist is None
                            or nearest_dist > frame_dist
                            or (nearest_dist == frame_dist and nearest_frame > candidate_frame)
                        ):
                            nearest_dist = frame_dist
                            nearest_frame = candidate_frame
                            min_turn = candidate_turn

                    if min_turn is None:
                        continue

                    # Determine analysis frame
                    llm_frame = min(nearest_frame, min_turn["lane_change_frame"] - 1)
                    affected_id = min_turn["affectedId"]
                    continue_end = min_turn["cont_end"]
                    continue_start = min_turn["dec_frame"]

                    # Adjust llm_frame if vehicles not available
                    while (
                        frame_car_grp.get((llm_frame, car_id)) is None
                        or frame_car_grp.get((llm_frame, affected_id)) is None
                    ):
                        llm_frame -= 1
                        if not (continue_start <= llm_frame and llm_frame <= continue_end):
                            print(f"problematic: {min_turn}")
                            break

                    # Calculate surrounding risks
                    drawer.init()
                    (
                        precedingId,
                        followingId,
                        leftPrecedingId,
                        leftFollowingId,
                        rightPrecedingId,
                        rightFollowingId,
                        precedingRisk,
                        followingRisk,
                        leftPrecedingRisk,
                        leftFollowingRisk,
                        rightPrecedingRisk,
                        rightFollowingRisk,
                        avg_x_speed,
                        surrounding_count,
                        preceding_x,
                        preceding_y,
                        preceding_xv,
                        following_x,
                        following_y,
                        following_xv,
                        leftPreceding_x,
                        leftPreceding_y,
                        leftPreceding_xv,
                        leftFollowing_x,
                        leftFollowing_y,
                        leftFollowing_xv,
                        rightPreceding_x,
                        rightPreceding_y,
                        rightPreceding_xv,
                        rightFollowing_x,
                        rightFollowing_y,
                        rightFollowing_xv,
                    ) = helpers.car_surrounding_six_risks(
                        frames_groups, car_id, llm_frame, drawer
                    )

                    # Add risk information to min_turn
                    min_turn["llm_frame"] = llm_frame
                    min_turn["precedingId"] = precedingId
                    min_turn["followingId"] = followingId
                    min_turn["leftPrecedingId"] = leftPrecedingId
                    min_turn["leftFollowingId"] = leftFollowingId
                    min_turn["rightPrecedingId"] = rightPrecedingId
                    min_turn["rightFollowingId"] = rightFollowingId
                    min_turn["precedingRisk"] = precedingRisk
                    min_turn["followingRisk"] = followingRisk
                    min_turn["leftPrecedingRisk"] = leftPrecedingRisk
                    min_turn["leftFollowingRisk"] = leftFollowingRisk
                    min_turn["rightPrecedingRisk"] = rightPrecedingRisk
                    min_turn["rightFollowingRisk"] = rightFollowingRisk
                    min_turn["avg_x_speed"] = avg_x_speed
                    min_turn["surrounding_count"] = surrounding_count
                    min_turn["preceding_x"] = preceding_x
                    min_turn["preceding_y"] = preceding_y
                    min_turn["preceding_xv"] = preceding_xv
                    min_turn["following_x"] = following_x
                    min_turn["following_y"] = following_y
                    min_turn["following_xv"] = following_xv
                    min_turn["leftPreceding_x"] = leftPreceding_x
                    min_turn["leftPreceding_y"] = leftPreceding_y
                    min_turn["leftPreceding_xv"] = leftPreceding_xv
                    min_turn["leftFollowing_x"] = leftFollowing_x
                    min_turn["leftFollowing_y"] = leftFollowing_y
                    min_turn["leftFollowing_xv"] = leftFollowing_xv
                    min_turn["rightPreceding_x"] = rightPreceding_x
                    min_turn["rightPreceding_y"] = rightPreceding_y
                    min_turn["rightPreceding_xv"] = rightPreceding_xv
                    min_turn["rightFollowing_x"] = rightFollowing_x
                    min_turn["rightFollowing_y"] = rightFollowing_y
                    min_turn["rightFollowing_xv"] = rightFollowing_xv

                    # Get ego vehicle kinematics at llm_frame
                    ego_car_frame = frame_car_grp[(llm_frame, car_id)]
                    min_turn["frame_start"] = first_frame_start
                    min_turn["frame_end"] = tgt_frame_end
                    min_turn["lane_id"] = ego_car_frame["laneId"].item()
                    min_turn["x"] = ego_car_frame["x"].item()
                    min_turn["y"] = ego_car_frame["y"].item()
                    min_turn["xVelocity"] = ego_car_frame["xVelocity"].item()
                    min_turn["yVelocity"] = ego_car_frame["yVelocity"].item()
                    min_turn["xAcceleration"] = ego_car_frame[
                        "xAcceleration"
                    ].item()
                    min_turn["yAcceleration"] = ego_car_frame[
                        "yAcceleration"
                    ].item()

                    # Calculate risk patterns
                    keep_left, keep_right, _, _, keep_left2, keep_right2 = (
                        lane_info.car_lane_left_right_valid(
                            min_turn["lane_id"], min_turn["xVelocity"]
                        )
                    )
                    risk_pattern_vec = helpers.get_risk_pattern(
                        frames_groups[llm_frame],
                        min_turn,
                        leftFollowingRisk,
                        leftPrecedingRisk,
                        followingRisk,
                        precedingRisk,
                        rightFollowingRisk,
                        rightPrecedingRisk,
                        keep_left,
                        keep_right,
                        keep_left2,
                        keep_right2,
                    )

                    # Add risk pattern matrix
                    min_turn["rp00"] = risk_pattern_vec[0, 0]
                    min_turn["rp01"] = risk_pattern_vec[0, 1]
                    min_turn["rp02"] = risk_pattern_vec[0, 2]
                    min_turn["rp10"] = risk_pattern_vec[1, 0]
                    min_turn["rp11"] = risk_pattern_vec[1, 1]
                    min_turn["rp12"] = risk_pattern_vec[1, 2]
                    min_turn["rp20"] = risk_pattern_vec[2, 0]
                    min_turn["rp21"] = risk_pattern_vec[2, 1]
                    min_turn["rp22"] = risk_pattern_vec[2, 2]
                    min_turn["rp30"] = risk_pattern_vec[3, 0]
                    min_turn["rp31"] = risk_pattern_vec[3, 1]
                    min_turn["rp32"] = risk_pattern_vec[3, 2]
                    min_turn["rp40"] = risk_pattern_vec[4, 0]
                    min_turn["rp41"] = risk_pattern_vec[4, 1]
                    min_turn["rp42"] = risk_pattern_vec[4, 2]

                    dangerous_turns.append(min_turn)
                    min_turn = None

    return dangerous_turns


def filter_dangerous_turns_recording(
    iRecord: str,
    gpu_num: int,
    min_ttc_val: float,
    continue_dangerous_val: float,
    buffer_time_val: float,
    preprocessing_dir: str,
):
    """
    Process one recording to find dangerous turns.

    Args:
        data_path: Path to highD dataset
        iRecord: Recording identifier
        gpu_num: GPU number for drawer
        min_ttc_val: TTC threshold
        continue_dangerous_val: Continuous dangerous time threshold
        buffer_time_val: Buffer time after lane change
        preprocessing_csv: Path to preprocessing_highD.csv

    Returns:
        numpy array of dangerous turn results
    """
    global min_ttc, continue_dangerous_frames, buffer_time
    min_ttc = min_ttc_val
    continue_dangerous_frames = continue_dangerous_val
    buffer_time = buffer_time_val

    # Load preprocessed data
    turn_df = load_preprocessing_data(preprocessing_dir, iRecord)

    if turn_df.empty:
        return np.array([])

    # Load metadata files directly
    recording_meta = load_recording_meta(preprocessing_dir, iRecord)
    tracks_meta = load_tracks_meta(preprocessing_dir, iRecord)

    # Get lane info
    lane_info = helpers.NormalizedLaneInfo.by_mgr(recording_meta)

    # Get max frame ID from tracks_meta
    max_frame_id = tracks_meta["finalFrame"].max()

    # Build data structures with ALL vehicles
    car_groups, frames_groups, frame_car_grp = build_data_structures(turn_df)

    # Get lane-changing vehicles
    lane_change_list = tracks_meta[tracks_meta["numLaneChanges"] > 0]
    lane_change_list = lane_change_list[["id", "numLaneChanges"]]

    # Initialize drawer
    percentile = 0.997
    drawer = helpers.FrameDrawer(gpu_num, percentile)

    # Process each lane-changing vehicle
    result = []
    for car_meta in lane_change_list.itertuples(index=False):
        car_id = car_meta.id
        lane_change_count = car_meta.numLaneChanges

        # Check left turns
        turn_dataframe = find_target_turns_from_csv(
            car_id,
            car_groups,
            -1,
            car_groups,
            lane_change_count,
            drawer,
            frames_groups,
            iRecord,
            frame_car_grp,
            lane_info,
            max_frame_id,
        )
        if len(turn_dataframe) > 0:
            result.extend(turn_dataframe)

        # Check right turns
        turn_dataframe = find_target_turns_from_csv(
            car_id,
            car_groups,
            1,
            car_groups,
            lane_change_count,
            drawer,
            frames_groups,
            iRecord,
            frame_car_grp,
            lane_info,
            max_frame_id,
        )
        if len(turn_dataframe) > 0:
            result.extend(turn_dataframe)

    # Convert to DataFrame
    result_df = pd.DataFrame(result)
    if result_df.empty:
        return result_df.to_numpy()

    result_df.loc[:, "iRecord"] = iRecord

    # Reorder columns
    result_df = result_df[desired_columns]

    # Clean up
    del (
        frames_groups,
        car_groups,
        lane_change_list,
        turn_df,
    )
    gc.collect()

    return result_df.to_numpy()


def filter_task_mp(args):
    """Multiprocessing wrapper for filter_dangerous_turns_recording."""
    (
        data_path,
        iRecord,
        gpu_num,
        min_ttc,
        continue_dangerous,
        buffer_time,
        preprocessing_csv,
    ) = args
    result = filter_dangerous_turns_recording(
        data_path, iRecord, gpu_num, min_ttc, continue_dangerous, buffer_time, preprocessing_csv
    )
    gc.collect()
    return result


def parallel_filter_dangerous_turns(
    record_list: List[str],
    gpu_count: int,
    gpu_offset: int,
    max_workers,
    min_ttc: float,
    continue_dangerous: float,
    buffer_time: float,
    preprocessing_dir: str,
):
    """
    Filter dangerous turns from multiple recordings in parallel.

    Args:
        data_path: Path to highD dataset
        record_list: List of recording identifiers
        gpu_count: Number of GPUs
        gpu_offset: GPU ID offset
        max_workers: Number of parallel processes
        min_ttc: TTC threshold
        continue_dangerous: Continuous dangerous time
        buffer_time: Buffer time after lane change
        preprocessing_csv: Path to preprocessed data

    Returns:
        Total number of dangerous turns found
    """
    # Auto-detect CPU count
    max_workers = max_workers or os.cpu_count()
    gpu_nums = [gpu_offset + n % gpu_count for n in range(len(record_list))]

    start_time = time.perf_counter()
    task_args = [
        (r, g, min_ttc, continue_dangerous_frames, buffer_time, preprocessing_dir)
        for r, g in zip(record_list, gpu_nums)
    ]
    total = 0

    with ProcessPoolExecutor(max_workers) as executor:
        futures = [executor.submit(filter_task_mp, args) for args in task_args]

        # Collect results
        list_numpy_array = []
        for future in as_completed(futures):
            numpy_array = future.result()
            count = len(numpy_array)
            if count > 0:
                list_numpy_array.append(numpy_array)
            total += count

        # Convert to DataFrame and save
        list_dataframes = [
            pd.DataFrame(arr, columns=desired_columns) for arr in list_numpy_array
        ]

        if len(list_dataframes) == 0:
            print("warning: data frame is empty")
        else:
            all_df = pd.concat(list_dataframes, ignore_index=True)
            all_df.sort_values(by=["iRecord", "id", "frame_start"], inplace=True)
            csv_fn = f"dangerous_turns_ttc_{min_ttc}_cont_{continue_dangerous}.csv"
            all_df.to_csv(csv_fn, index=False)
            print(f"{csv_fn} is written")

    duration = time.perf_counter() - start_time
    print(
        f"Processing complete | Total: {total:,} | Time: {duration:.2f}s | "
        f"Speed: {total / duration:,.0f} items/s"
    )
    return total


def parse_args():
    parser = argparse.ArgumentParser(description="Filter dangerous turns from preprocessed data")
    parser.add_argument(
        "--preprocessing_dir",
        type=str,
        default="../preprocessed_highD",
        help="Path to preprocessed CSV file",
        required=True,
    )
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--gpu_offset", type=int, default=0, help="GPU offset")
    parser.add_argument(
        "--max_workers", type=int, default=None, help="Max worker processes (default: CPU count)"
    )
    parser.add_argument("--ttc", type=float, default=4.0, help="Minimum TTC threshold")
    parser.add_argument("--cont", type=float, default=0.04, help="Continuous dangerous time")
    parser.add_argument("--buffer", type=float, default=2.0, help="Buffer time after lane change")
    return parser.parse_args()

# python -m respond.highd.filter_dangerous_turns --preprocessing_dir=<path_to_preprocessed_data>
if __name__ == "__main__":
    args = parse_args()

    # Process recordings 01-60
    record_list = [f"{i:02d}" for i in range(1, 61)]

    start_time = time.time()

    print("=" * 60)
    print("Filtering Dangerous Turns from Preprocessed Data")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Preprocessing Directory: {args.preprocessing_dir}")
    print(f"Recordings: {len(record_list)} (01-60)")
    print(f"TTC threshold: {args.ttc}")
    print(f"Continuous dangerous: {args.cont}")
    print(f"Buffer time: {args.buffer}")
    print(f"Max workers: {args.max_workers or os.cpu_count()}")
    print("=" * 60)
    print()

    parallel_filter_dangerous_turns(
        record_list=record_list,
        gpu_count=args.gpu_count,
        gpu_offset=args.gpu_offset,
        max_workers=args.max_workers,
        min_ttc=args.ttc,
        continue_dangerous=args.cont,
        buffer_time=args.buffer,
        preprocessing_dir=args.preprocessing_dir,
    )

    print(f"\nTotal elapsed time: {time.time() - start_time:.2f}s")
