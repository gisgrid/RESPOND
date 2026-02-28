import argparse
import os
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from respond.highd.highD_helpers import DataManagerHighD, FrameDrawer, draw_segment_animation,cal_draw_start


def parallel_draw(
    data_path,
    selected_fn,
    output_path,
    gpu_count=1,
    gpu_offset: int = 0,
    max_workers=None,
):
    cmp = pd.read_csv(selected_fn, dtype={"iRecord": str})
    unique_records = cmp["iRecord"].unique().tolist()
    del cmp
    max_workers = max_workers or os.cpu_count()
    gpu_nums = [gpu_offset + (n % gpu_count) for n in range(0, len(unique_records))]
    start_time = time.perf_counter()
    task_args = [
        (data_path, selected_fn, r, output_path, g)
        for (r, g) in zip(unique_records, gpu_nums)
    ]
    total = 0
    with ProcessPoolExecutor(max_workers) as executor:
        futures = [executor.submit(draw_frame, *args) for args in task_args]

        for future in as_completed(futures):
            count = future.result()
            total += count

    duration = time.perf_counter() - start_time
    print(
        f"处理完成 | 总数: {total:,} | 用时: {duration:.2f}s | "
        f"速度: {total / duration:,.0f} items/s"
    )
    return total


def draw_frame(data_path, selected_fn: str, iRecord: str, output_path, gpu_num: int):
    cmp = pd.read_csv(selected_fn, dtype={"iRecord": str})
    groups = {iRecord: g for iRecord, g in cmp.groupby("iRecord")}
    tgt = groups[iRecord]

    data_mgr = DataManagerHighD(data_path)
    percentile = 0.997
    drawer = FrameDrawer(gpu_num, percentile)
    drawer.init()
    slow_down_ratio = 5

    start_list = []
    for val in tgt.itertuples(index=False):
        iRecord = val.iRecord
        vehicle_id = val.id
        llm_frame = val.llm_frame
        dec_frame = val.dec_frame
        lane_change_frame = val.lane_change_frame
        segment_start = val.frame_start
        start_change = cal_draw_start(dec_frame,segment_start,lane_change_frame)
        end_change = val.frame_end
        affected_id = val.affectedId
        dangerous_class = 3
        img_id = f"{iRecord}_{vehicle_id}_{start_change}_{end_change}_{affected_id}"
        if img_id in start_list:
            continue
        start_list.append(img_id)
        draw_segment_animation(
            data_mgr,
            output_path,
            drawer,
            slow_down_ratio,
            iRecord,
            start_change,
            end_change,
            dangerous_class,
            vehicle_id,
            affected_id,
            dec_frame,
            llm_frame,
        )

    drawer.clear_cache()
    del drawer, tgt, groups, cmp, data_mgr
    gc.collect()
    return len(start_list)


def parse_args():
    parser = argparse.ArgumentParser(description="draw highD car frame animations")
    parser.add_argument(
        "--data_path", type=str, default="02.highD-data", help="highD data path",required=True
    )
    parser.add_argument("--gpu_count", type=int, default=1, help="GPU count")
    parser.add_argument("--gpu_offset", type=int, default=0, help="GPU start index offset")
    parser.add_argument(
        "--max_workers", type=int, default=None, help="max worker processes (default: number of CPU cores)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="dangerous_turns_ttc_1.0_cont_1.0.csv",
        help="input data file name",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="dangerous",
        help="output data path",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_path = args.data_path
    selected_fn = args.filename
    output_path = args.output_path
    start_time = time.time()

    parallel_draw(data_path, selected_fn, output_path, args.gpu_count, args.gpu_offset)

    print(f"\nTotal elapsed time: {time.time() - start_time:.2f}s")
