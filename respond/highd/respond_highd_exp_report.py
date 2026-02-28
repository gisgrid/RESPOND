import argparse
import time
import os

import pandas as pd

from respond.highd.highD_helpers import (
    NormalizedLaneInfo,
    get4Risk,
    getLeftRightValid,
    cal_draw_start,
)


def calc_idle_risk(
    keep_left, keep_right, front_risk, behind_risk, left_risk, right_risk
):
    sum = front_risk + behind_risk
    count = 2
    if keep_left:
        sum += left_risk
        count += 1
    if keep_right:
        sum += right_risk
        count += 1
    return sum / count


def calc_dec_risk(
    decision, keep_left, keep_right, front_risk, behind_risk, left_risk, right_risk
):
    if decision == -1 or decision == 1:
        dec_risk = left_risk if decision == -1 else right_risk
    elif decision == -2 or decision == 2:
        dec_risk = behind_risk if decision == -2 else front_risk
    else:
        dec_risk = calc_idle_risk(
            keep_left, keep_right, front_risk, behind_risk, left_risk, right_risk
        )
    return dec_risk


def report(fn: str, data_path: str, output_file: str):
    result_df = pd.read_csv(fn, dtype={"iRecord": str})
    safer_count = 0
    total_count = len(result_df)
    ext_list = []
    row_index = 0
    for row in result_df.itertuples():
        affected_id = row.affectedId
        ego_id = row.id
        llm_dec = row.llm_dec
        human_dec = row.dec

        iRecord = row.iRecord
        meta_file_path = os.path.join(data_path, f"{iRecord}_recordingMeta.csv")
        recording_meta = pd.read_csv(meta_file_path)
        lane_info = NormalizedLaneInfo.by_mgr(recording_meta)
        keep_left, keep_right, _, _, _, _ = getLeftRightValid(row, lane_info)
        front_risk, behind_risk, left_risk, right_risk = get4Risk(
            row, keep_left, keep_right
        )
        risk_human = calc_dec_risk(
            human_dec,
            keep_left,
            keep_right,
            front_risk,
            behind_risk,
            left_risk,
            right_risk,
        )
        risk_llm = calc_dec_risk(
            llm_dec,
            keep_left,
            keep_right,
            front_risk,
            behind_risk,
            left_risk,
            right_risk,
        )

        # Three-state comparison: better, equal, not_better
        comparison = "not_better"
        if risk_llm < risk_human:
            safer_count += 1
            comparison = "better"
        elif risk_llm == risk_human:
            comparison = "equal"
        else:
            comparison = "not_better"

        ext_dict = {}
        row_index += 1
        ext_dict["No"] = row_index
        ext_dict["comparison"] = comparison
        ext_dict["left_risk"] = left_risk
        ext_dict["right_risk"] = right_risk
        ext_dict["front_risk"] = front_risk
        ext_dict["behind_risk"] = behind_risk
        ext_dict["llm_dec_risk"] = risk_llm
        ext_dict["human_dec_risk"] = risk_human

        start_change = cal_draw_start(row.dec_frame,row.frame_start,row.lane_change_frame)
        end_change = row.frame_end

        ext_dict["Pic"] = f"llm_{iRecord}_{row.llm_frame}_{ego_id}_{affected_id}.png"
        ext_dict["Mp4"] = f"{iRecord}_{start_change}_{end_change}_{ego_id}_{affected_id}.mp4"
        ext_list.append(ext_dict)

    safer_rate = safer_count / total_count
    print(
        f"safer_rate:{safer_rate},safer_count:{safer_count},total_count:{total_count}"
    )
    ext = pd.DataFrame(ext_list)
    result_ext_df = pd.concat([result_df, ext], axis=1)
    # No,Pic,Mp4,iRecord,id,affectedId,dec,dec_frame,llm_frame,llm_dec,frame_start,frame_end,lane_change_frame,advoid,is_compatible,is_safer,front_risk,behind_risk,left_risk,right_risk
    desired_columns = [
        "No",
        "Pic",
        "Mp4",
        "iRecord",
        "id",
        "affectedId",
        "dec_frame",
        "dec",
        "human_dec_risk",
        "llm_frame",
        "llm_dec",
        "llm_dec_risk",
        "frame_start",
        "frame_end",
        "lane_change_frame",
        "lane_id",
        "comparison",
        "front_risk",
        "behind_risk",
        "left_risk",
        "right_risk",
        "leftPrecedingRisk",
        "leftFollowingRisk",
        "rightPrecedingRisk",
        "rightFollowingRisk",
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
    report_df = result_ext_df[desired_columns]
    report_df.to_csv(output_file, index=False)
    print(f"{output_file} is written")


def parse_args():
    parser = argparse.ArgumentParser(description="RESPONSE highD experiment report generation")
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
        default="llm_dec_ttc_4.0.csv",
        help="RESPONSE result data filename",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="RESPOND_highD_experiment_report_ttc_4.0.csv",
        help="output report filename",
    )
    return parser.parse_args()

# python -m respond.highd.respond_highd_exp_report --preprocessing_dir=<path_to_preprocessed_data> --filename=llm_dec_ttc_4.0.csv --output_file=RESPOND_highD_experiment_report_ttc_4.0.csv
if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()

    data_path = args.preprocessing_dir
    fn = args.filename
    output_file = args.output_file
    print(f"processing {fn}")

    report(fn, data_path, output_file)

    print(f"\nTotal elapsed time: {time.time() - start_time:.2f}s")
