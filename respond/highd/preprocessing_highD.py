"""
Preprocessing Script for highD Dataset - Extract All Turn Sequences

This script extracts all frames within turn sequences (left and right lane changes)
from the highD dataset

Output: preprocessing_highD.csv
"""

import argparse
import gc
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import numpy as np
import pandas as pd

import respond.highd.highD_helpers as helpers
from respond.highd.highD_helpers import sign

# Configuration parameters
acc_threashold = 0.01
turning_threashold = 1.0

# Original tracks fields (25 fields from highD dataset)
tracks_fields = [
    "frame", "id", "x", "y", "width", "height",
    "xVelocity", "yVelocity", "xAcceleration", "yAcceleration",
    "frontSightDistance", "backSightDistance", "dhw", "thw", "ttc",
    "precedingXVelocity", "precedingId", "followingId",
    "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
    "rightPrecedingId", "rightAlongsideId", "rightFollowingId",
    "laneId"
]


# Reuse utility functions from dangerous_turn.py
def sign_item(x):
    x = x.item()
    return sign(x)


v_sign = np.vectorize(sign)


def detect_item_acc(cur: pd.DataFrame, threashold):
    return (
        sign_item(cur["xVelocity"])
        * sign_item(cur["xAcceleration"])
        * sign_item(cur["xAcceleration"].abs() > threashold)
    )


def detect_acc(cur: pd.DataFrame, threashold):
    return (
        v_sign(cur["xVelocity"])
        * v_sign(cur["xAcceleration"])
        * v_sign(cur["xAcceleration"].abs() > threashold)
    )


def cal_human_decision(data_mgr: helpers.DataManagerHighD, iRecord: str):
    """
    Classify driving decisions for all vehicles in a recording.

    Returns:
        DataFrame with all tracks fields + dec column (26 total):
        [id, frame, dec, x, y, width, height,
         xVelocity, yVelocity, xAcceleration, yAcceleration,
         frontSightDistance, backSightDistance, dhw, thw, ttc,
         precedingXVelocity, precedingId, followingId,
         leftPrecedingId, leftAlongsideId, leftFollowingId,
         rightPrecedingId, rightAlongsideId, rightFollowingId,
         laneId]
    """
    recording_meta = data_mgr.recording_meta(iRecord)
    lane_info = helpers.NormalizedLaneInfo.by_mgr(recording_meta)
    lane_up_map = lane_info.lane_up_map
    lane_down_map = lane_info.lane_down_map

    tracks = data_mgr.tracks(iRecord)
    tracks_meta = data_mgr.tracks_meta(iRecord)

    # Use all tracks fields
    tracks_use = tracks
    lane_change_list = tracks_meta[tracks_meta["numLaneChanges"] > 0].reset_index(
        drop=True
    )
    lane_change_list = lane_change_list[
        ["id", "width", "height", "initialFrame", "finalFrame", "numLaneChanges"]
    ]
    changed_ids = lane_change_list["id"]
    not_change = tracks_use[~tracks_use["id"].isin(changed_ids)].copy()
    not_change = not_change.rename(columns={"y": "y_up"})
    not_change["y_down"] = not_change["y_up"] + not_change["height"]
    not_change["laneUp"] = not_change["laneId"].map(lane_up_map)
    not_change["laneDown"] = not_change["laneId"].map(lane_down_map)
    not_change["turning"] = 0
    not_change["accelerating"] = detect_acc(not_change, acc_threashold)
    not_change["turnDec"] = 0
    not_change["dec"] = 2 * not_change["accelerating"]
    whole = [not_change]
    for val in lane_change_list.itertuples():
        cur = tracks_use[tracks_use["id"] == val.id].copy()
        cur["y_down"] = cur["y"] + val.height
        cur = cur.rename(columns={"y": "y_up"})
        cur["laneUp"] = cur["laneId"].map(lane_up_map)
        cur["laneDown"] = cur["laneId"].map(lane_down_map)
        # -1: turn left, 1: turn right, 0: no turning
        cur["turning"] = v_sign(cur["xVelocity"]) * v_sign(cur["yVelocity"])
        # -1: decelerating, 1: accelerating, 0: idle
        cur["accelerating"] = detect_acc(cur, acc_threashold)
        cur["turnDec"] = 0
        change_point = cur[cur["laneId"].diff().ne(0)]
        change_point = change_point["frame"]
        points = []
        for _, point in change_point.items():
            points.append(point)
        points.append(val.finalFrame)
        change_seg = []
        for i in range(0, len(points) - 2):
            change_seg.append((points[i], points[i + 1], points[i + 2]))
        for min_id, cut, max_id in change_seg:
            decision = cur[cur["frame"] == cut]["turning"].item()
            candidates = cur[
                (cur["turning"].diff().ne(0))
                & (cur["frame"] >= min_id)
                & (cur["frame"] < cut)
            ]
            candidates = candidates["frame"].max()
            start = min_id if np.isnan(candidates) else candidates
            candidates = cur[
                (cur["turning"].diff().ne(0))
                & (cur["frame"] >= cut)
                & (cur["frame"] <= max_id)
            ]
            candidates = candidates["frame"].min()
            end = max_id if np.isnan(candidates) else candidates - 1
            cur.loc[(cur["frame"] >= start) & (cur["frame"] <= end), "turnDec"] = (
                decision
            )
        # -1: turn left, 1: turn right, -2: decelerate, 2: accelerate, 0: idle
        cur["dec"] = np.where(
            cur["turnDec"] != 0, cur["turnDec"], 2 * cur["accelerating"]
        )
        whole.append(cur)

    all_frames = pd.concat(whole, ignore_index=True)
    # Rename y_up back to y
    all_frames = all_frames.rename(columns={"y_up": "y"})
    # Keep all tracks fields + dec column
    # Remove temporary columns
    # temp_cols = ["y_down", "laneUp", "laneDown", "turning", "accelerating", "turnDec"]
    # all_frames = all_frames.drop(columns=[col for col in temp_cols if col in all_frames.columns])
    # Reorder columns: id, frame, dec, then all other tracks fields
    output_columns = ["id", "frame", "dec"] + [f for f in tracks_fields if f not in ["id", "frame"]]
    all_frames = all_frames[output_columns]
    del (
        whole,
        not_change,
        changed_ids,
        lane_change_list,
        tracks_use,
    )
    return all_frames


def extract_turn_frames_for_recording(data_path: str, output_directory: str, iRecord: str) -> int:
    """
    Extract all frames within turn sequences for a single recording.
    Includes ALL vehicles in the scene during turn sequences (not just turning vehicles).

    Args:
        data_path: Path to highD dataset
        iRecord: Recording identifier (e.g., "01", "02")

    Returns:
        DataFrame with columns: [frame, id, dec, ...tracks_fields] (26 columns)
        Returns empty DataFrame if no lane-changing vehicles exist.
    """
    data_mgr = helpers.DataManagerHighD(data_path)

    # Get tracks_meta to identify lane-changing vehicles
    tracks_meta = data_mgr.tracks_meta(iRecord)
    lane_change_list = tracks_meta[tracks_meta["numLaneChanges"] > 0]

    if len(lane_change_list) == 0:
        # No lane-changing vehicles in this recording
        del data_mgr, tracks_meta
        gc.collect()
        return pd.DataFrame()

    # Compute human decisions - now returns all tracks fields + dec
    merged = cal_human_decision(data_mgr, iRecord)
    merged.sort_values(by=["id", "frame"], inplace=True)

    frame_ranges = []

    # Process each lane-changing vehicle to collect frame ranges
    for vehicle_id in lane_change_list["id"].unique():
        vehicle_tracks = merged[merged["id"] == vehicle_id].copy()

        if len(vehicle_tracks) == 0:
            continue

        # Sort by frame to ensure correct sequence detection
        vehicle_tracks.sort_values("frame", inplace=True)

        # Filter turning frames (|dec| == 1)
        turning_mask = vehicle_tracks["dec"].abs() == 1

        if not turning_mask.any():
            continue

        # Compute sequence IDs using the same method as dangerous_turn.py
        seq_groups = turning_mask.ne(turning_mask.shift()).cumsum()

        # Add seq column to vehicle tracks (for internal grouping only, not in output)
        vehicle_tracks.loc[:, "seq"] = seq_groups

        # For each unique sequence, save the frame range
        for seq_id in vehicle_tracks.loc[turning_mask, "seq"].unique():
            seq_frames = vehicle_tracks[vehicle_tracks["seq"] == seq_id]

            if len(seq_frames) == 0:
                continue

            frame_start = seq_frames["frame"].min()
            frame_end = seq_frames["frame"].max()
            frame_ranges.append((frame_start, frame_end))

    if len(frame_ranges) == 0:
        del data_mgr, tracks, tracks_meta, car_ext_df, merged
        gc.collect()
        return pd.DataFrame()

    # Merge overlapping frame ranges using segment concatenation algorithm
    frame_ranges.sort(key=lambda x: x[0])  # Sort by start frame
    merged_ranges = []
    for start, end in frame_ranges:
        if not merged_ranges:
            merged_ranges.append([start, end])
        else:
            last_start, last_end = merged_ranges[-1]
            # Merge if overlapping or adjacent
            if start <= last_end:
                merged_ranges[-1][1] = max(last_end, end)
            else:
                merged_ranges.append([start, end])

    # Extract data for each merged frame range
    result_list = []
    for frame_start, frame_end in merged_ranges:
        # Extract ALL vehicles in this frame range
        all_frames_in_seq = merged[
            (merged["frame"] >= frame_start) &
            (merged["frame"] <= frame_end)
        ].copy()
        result_list.append(all_frames_in_seq)

    # Concatenate all results
    result_df = pd.concat(result_list, ignore_index=True)

    # tracks_fields already includes frame and id at the beginning
    output_columns = ["frame", "id", "dec"] + [f for f in tracks_fields if f not in ["frame", "id"]]
    result_df = result_df[output_columns]
    # Sort by frame, id
    result_df.sort_values(by=["frame", "id"], inplace=True)

    # Save to CSV
    os.makedirs(output_directory, exist_ok=True)
    output_filename = f"{output_directory}/preprocessing_highD_{iRecord}.csv"
    result_df.to_csv(output_filename, index=False)
    print(f"{output_filename} is written")

    count = result_df.shape[0]

    # Clean up
    del data_mgr, tracks_meta, merged, result_df
    gc.collect()

    return count


def extract_task_mp(args):
    """Multiprocessing wrapper for extract_turn_frames_for_recording."""
    data_path, output_directory, iRecord = args
    result = extract_turn_frames_for_recording(data_path, output_directory, iRecord)
    return result


def parallel_extract_turn_frames(
    data_path: str,
    record_list: List[str],
    output_directory: str,
    max_workers: int = None
) -> int:
    """
    Extract turn frames from multiple recordings in parallel.

    Args:
        data_path: Path to highD dataset
        record_list: List of recording identifiers (e.g., ["01", "02", ...])
        max_workers: Number of parallel processes (default: CPU count)

    Returns:
        Total number of rows extracted
    """
    # Auto-detect CPU count if not specified
    max_workers = max_workers or os.cpu_count()

    start_time = time.perf_counter()
    task_args = [(data_path, output_directory, r) for r in record_list]
    total = 0

    print(f"Processing {len(record_list)} recordings with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(extract_task_mp, args) for args in task_args]

        for future in as_completed(futures):
            count = future.result()
            total += count
            # Print progress
            completed = sum(1 for f in futures if f.done())
            print(f"Progress: {completed}/{len(task_args)} recordings | Rows so far: {total:,}")

    # Combine all DataFrames
    if total == 0:
        print("warning: no turn sequences found in any recording")
        return 0

    print(f"Total rows: {total:,}")

    duration = time.perf_counter() - start_time
    print(f"\nProcessing completed in {duration:.2f}s")
    print(f"Speed: {total / duration:,.0f} rows/sec")

    return total


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract all turn sequences from highD dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="highD-data",
        help="Path to highD dataset",
        required=True
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="preprocessed_highD",
        help="Output directory for processed data (default: preprocessed_highD)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of parallel processes (default: CPU count)"
    )
    return parser.parse_args()

# python -m respond.highd.preprocessing_highD --data_path=<path_to_highD_data>
if __name__ == "__main__":
    args = parse_args()

    # Process all 60 recordings
    record_list = [f"{i:02d}" for i in range(1, 61)]

    print("=" * 60)
    print("highD Turn Sequence Extraction")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Recordings: {len(record_list)} (01-60)")
    print(f"Max workers: {args.max_workers or os.cpu_count()}")
    print("=" * 60)
    print()

    start_time = time.time()

    parallel_extract_turn_frames(
        data_path=args.data_path,
        record_list=record_list,
        output_directory=args.output_directory,
        max_workers=args.max_workers
    )

    print(f"\nTotal elapsed time: {time.time() - start_time:.2f}s")
