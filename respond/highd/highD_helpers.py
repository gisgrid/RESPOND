import gc
import math
import os
import sys
from collections import namedtuple
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from numpy import ndarray as D
from PIL import Image
from PIL.ImageFile import ImageFile
from torch import Tensor as T

import respond.risk_pattern.risk as risk
import respond.risk_pattern.risk_torch as risk_torch


def sign(x):
    if x < 0.0:
        return -1
    elif x > 0.0:
        return 1
    else:
        return 0


def cal_following_distance(
    ego_x: float,
    ego_width: float,
    ego_xv: float,
    following_x: float,
    following_width: float,
):
    if ego_xv < 0:
        # 向左行驶时，x是左上角坐标，因此前车车尾位置需要加上车的长度
        following_distance = following_x - (ego_width + ego_x)
    else:
        # 向右行驶时，后车车头位置需要加上车的长度
        following_distance = ego_x - (following_x + following_width)
    return following_distance


def cal_min_distance_x_ttc(
    ego_x: float,
    ego_y: float,
    ego_width: float,
    ego_height: float,
    ego_x_v: float,
    ego_y_v: float,
    follow_x: float,
    follow_y: float,
    follow_width: float,
    follow_height: float,
    follow_x_v: float,
    follow_y_v: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    计算两辆车之间的最小距离和预计碰撞时间（TTC - Time To Collision）

    参数:
        ego_x, ego_y: 主车（ego车辆）的位置坐标（左上角）
        ego_width, ego_height: 主车的宽度和高度
        ego_x_v, ego_y_v: 主车在x和Y轴的速度
        follow_x, follow_y: 跟随车（follow车辆）的位置坐标（左上角）
        follow_width, follow_height: 跟随车的宽度和高度
        follow_x_v, follow_y_v: 跟随车在x和Y轴的速度

    返回:
        一个包含三个元组的元组:
        - (ttc, min_dhw): 预计碰撞时间和最小距离
        - (ego_attach_x, ego_attach_y): 主车离跟随车辆的最近点坐标
        - (follow_attach_x, follow_attach_y): 跟随车离主车的最近点坐标
    """

    # === X轴计算 ===
    # 确定两辆车在X轴上的最近点

    if follow_x > ego_x + ego_width:
        # 跟随车在主车右侧，距离为两车之间的空隙
        # 主车最近点为其右边缘，跟随车最近点为其左边缘
        ego_attach_x = ego_x + ego_width
        follow_attach_x = follow_x
    elif follow_x + follow_width < ego_x:
        # 跟随车在主车左侧，距离为两车之间的空隙
        # 主车最近点为其左边缘，跟随车最近点为其右边缘
        ego_attach_x = ego_x
        follow_attach_x = follow_x + follow_width
    else:
        # 两辆车在X轴上有重叠
        # 取重叠区域的中间位置作为两车的最近点
        x_list = [ego_x, ego_x + ego_width, follow_x, follow_x + follow_width]
        x_list.sort()
        ego_attach_x = (x_list[1] + x_list[2]) / 2
        follow_attach_x = ego_attach_x

    is_y_overlap = False
    # === Y轴计算 ===
    # 确定两辆车在Y轴上的最近点

    if follow_y > ego_y + ego_height:
        # 跟随车在主车下方，距离为两车之间的空隙
        # 主车最近点为其下边缘，跟随车最近点为其上边缘
        ego_attach_y = ego_y + ego_height
        follow_attach_y = follow_y
    elif follow_y + follow_height < ego_y:
        # 跟随车在主车上方，距离为两车之间的空隙
        # 主车最近点为其上边缘，跟随车最近点为其下边缘
        ego_attach_y = ego_y
        follow_attach_y = follow_y + follow_height
    else:
        # 两辆车在Y轴上有重叠
        # 取重叠区域的中间位置作为两车的最近点
        y_list = [ego_y, ego_y + ego_height, follow_y, follow_y + follow_height]
        y_list.sort()
        ego_attach_y = (y_list[1] + y_list[2]) / 2
        follow_attach_y = ego_attach_y
        is_y_overlap = True

    # === 计算最小距离 ===
    # 计算两车最近点之间的X轴和Y轴距离
    min_x = ego_attach_x - follow_attach_x
    min_y = ego_attach_y - follow_attach_y
    # 使用欧几里得距离公式计算两车最近点之间的直线距离
    min_dhw = math.sqrt(min_x * min_x + min_y * min_y)

    # === 计算预计碰撞时间（TTC） ===
    # 首先计算X轴的相对速度
    rel_v_x = follow_x_v - ego_x_v
    if not is_y_overlap or rel_v_x == 0.0:
        # 如果X轴相对速度为0，则不会在X轴发生碰撞
        ttc = +0.0
    else:
        # 计算X轴上的预计碰撞时间：距离/相对速度
        # 使用max确保TTC不为负数（负数表示正在远离）
        ttc = abs(max(min_x / rel_v_x, 0.0))

    # === 返回结果 ===
    return (
        (ttc, min_dhw),  # 预计碰撞时间和最小距离
        (ego_attach_x, ego_attach_y),  # 主车上的最近点坐标
        (follow_attach_x, follow_attach_y),  # 跟随车上的最近点坐标
    )


def convert_to_vehicle_dict(car_frame):
    no_index = car_frame.iloc[0]
    car_dict = no_index.to_dict()
    CarTuple = namedtuple("CarTuple", car_dict.keys())
    vehicle = CarTuple(**car_dict)
    return vehicle


def vehicle_min_distance_x_ttc(ego, follow):
    """
    (ttc, distance),
    (ego_attach_x, ego_attach_y),
    (follow_attach_x, follow_attach_y),
    """
    return cal_min_distance_x_ttc(
        ego.x,
        ego.y,
        ego.width,
        ego.height,
        ego.xVelocity,
        ego.yVelocity,
        follow.x,
        follow.y,
        follow.width,
        follow.height,
        follow.xVelocity,
        follow.yVelocity,
    )


class NormalizedLaneInfo:
    def __init__(
        self,
        upper_lane_info: List[Tuple[int, float, float]],
        lower_lane_info: List[Tuple[int, float, float]],
        lane_up_map,
        lane_down_map,
    ):
        upper = []
        for lane_id, _, _ in upper_lane_info:
            upper.append(lane_id)
        lower = []
        for lane_id, _, _ in lower_lane_info:
            lower.append(lane_id)
        self.lower = lower
        self.upper = upper
        self.lane_up_map = lane_up_map
        self.lane_down_map = lane_down_map
        self.upper_lane_info = upper_lane_info
        self.lower_lane_info = lower_lane_info
        self.upper_lane_range = (upper_lane_info[0][0], upper_lane_info[-1][0])
        self.lower_lane_range = (lower_lane_info[0][0], lower_lane_info[-1][0])

    @staticmethod
    def by_mgr(recording_meta: pd.DataFrame):
        lane_marks: str = (
            recording_meta["upperLaneMarkings"].item()
            + ";"
            + recording_meta["lowerLaneMarkings"].item()
        )
        lane_marks_lst = parse_lane_marks(lane_marks)
        upper_lane_count = len(recording_meta["upperLaneMarkings"].item().split(";"))
        upper_lane_info = lane_marks_lst[0:upper_lane_count]
        lower_lane_info = lane_marks_lst[upper_lane_count:]
        lane_meta = pd.DataFrame(
            lane_marks_lst, columns=["laneId", "laneUp", "laneDown"]
        )
        lane_up_map = lane_meta.set_index("laneId")["laneUp"].to_dict()
        lane_down_map = lane_meta.set_index("laneId")["laneDown"].to_dict()
        return NormalizedLaneInfo(
            upper_lane_info, lower_lane_info, lane_up_map, lane_down_map
        )

    def get_lane_up(self, lane_id: int):
        return self.lane_up_map[lane_id]

    def get_lane_down(self, lane_id: int):
        return self.lane_down_map[lane_id]

    def get_lane_up_down(self, lane_id: int):
        return (self.lane_up_map[lane_id], self.lane_down_map[lane_id])

    def num_side_lanes(self, lane_id: int) -> Tuple[int, int]:
        idx = 0
        for global_id in self.lower:
            if lane_id == global_id:
                return len(self.lower) - 1, idx - 1
            idx += 1

        idx = 0
        for global_id in self.upper:
            if lane_id == global_id:
                return len(self.upper) - 1, idx - 1
            idx += 1

        return 0, -1

    def car_lane_left_right_valid(self, lane_id: int, xVelocity: float):
        num_lanes, norm_lane_idx = self.num_side_lanes(lane_id)
        if xVelocity < 0:
            # HighD车道的顺序是假设车辆从左向右行驶的(车道编号从左到右排列),如果速度是负数,意味着从右向左行驶,需要车道编号从左到右的排列需要反过来
            norm_lane_idx = num_lanes - norm_lane_idx - 1

        # 生成格式化后的可用动作描述字符串
        keep_left = norm_lane_idx > 0
        keep_right = norm_lane_idx < num_lanes - 1
        keep_left2 = norm_lane_idx > 1
        keep_right2 = norm_lane_idx < num_lanes - 2
        return (
            keep_left,
            keep_right,
            num_lanes,
            norm_lane_idx,
            keep_left2,
            keep_right2,
        )

    def in_upper_range(self, lane_id: int):
        return (
            self.upper_lane_range[0] <= lane_id and lane_id <= self.upper_lane_range[1]
        )

    def in_lower_range(self, lane_id: int):
        return (
            self.lower_lane_range[0] <= lane_id and lane_id <= self.lower_lane_range[1]
        )


def getLeftRightValid(ego_car, lane_info: NormalizedLaneInfo):
    return lane_info.car_lane_left_right_valid(ego_car.lane_id, ego_car.xVelocity)


def get4Risk(ego_car, keep_left: bool, keep_right: bool):
    front_risk = ego_car.precedingRisk
    behind_risk = ego_car.followingRisk
    if keep_left:
        left_risk = max(
            ego_car.leftPrecedingRisk,
            ego_car.leftFollowingRisk,
        )
    else:
        left_risk = 1.0
    if keep_right:
        right_risk = max(
            ego_car.rightPrecedingRisk,
            ego_car.rightFollowingRisk,
        )
    else:
        right_risk = 1.0
    return (front_risk, behind_risk, left_risk, right_risk)


class DataManagerHighD:
    """
    lazy loading and caching data
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.tracks_dict = {}
        self.tracks_meta_dict = {}
        self.recording_meta_dict = {}
        self.bg_img_dict = {}
        self.frame_rate_dict = {}
        self.lanes_dict = {}
        self.lazy_load_dict = {}
        self.lazy_load_dict["tracks"] = (self.tracks_dict, "tracks")
        self.lazy_load_dict["tracks_meta"] = (self.tracks_meta_dict, "tracksMeta")
        self.lazy_load_dict["recording_meta"] = (
            self.recording_meta_dict,
            "recordingMeta",
        )

    def lanes_y(self, iRecord: str):
        rec_meta = self.recording_meta(iRecord)
        data = self.lanes_dict.get(iRecord, None)
        if data is None:
            lane_y_list = rec_meta["upperLaneMarkings"].iloc[0].split(";")
            lane_y_list: list[str] = lane_y_list + rec_meta["lowerLaneMarkings"].iloc[
                0
            ].split(";")
            data = [float(y) for y in lane_y_list]
            self.lanes_dict[iRecord] = data
        return data

    def frame_rate(self, iRecord: str):
        rec_meta = self.recording_meta(iRecord)
        data = self.frame_rate_dict.get(iRecord, None)
        if data is None:
            data = rec_meta["frameRate"][0]
            self.frame_rate_dict[iRecord] = data
        return data

    def bg_img(self, iRecord: str):
        data = self.bg_img_dict.get(iRecord, None)
        if data is None:
            data = plt.imread(f"{self.data_path}/{iRecord}_highway.jpg")
            self.bg_img_dict[iRecord] = data
        return data

    def lazy_load(self, iRecord: str, name: str) -> pd.DataFrame:
        (cache, nm) = self.lazy_load_dict[name]
        data = cache.get(iRecord, None)
        if data is None:
            fileName = f"{self.data_path}/{iRecord}_{nm}.csv"
            data = pd.read_csv(fileName)
            cache[iRecord] = data
        return data

    def tracks(self, iRecord: str):
        return self.lazy_load(iRecord, "tracks")

    def tracks_meta(self, iRecord: str):
        return self.lazy_load(iRecord, "tracks_meta")

    def recording_meta(self, iRecord: str):
        return self.lazy_load(iRecord, "recording_meta")


# torch.where(xVelocity > 0, torch.tensor(0.01), torch.tensor(180.01))
def heading(xVelocity):
    if xVelocity > 0:
        return 0.01
    else:
        return 180.01


v_heading = np.vectorize(heading)


def generate_range_size(length, block_size):
    n, r = divmod(length, block_size)
    for i in range(0, n + min(r, 1)):
        is_align = i < n
        size = is_align * block_size + (1 - is_align) * r
        start_ind = i * block_size
        end_ind = start_ind + size
        yield i, size, start_ind, end_ind


def save_fig_array(fig: plt.Figure):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    buffer.seek(0)  # 将缓冲区指针重置到开头
    image = Image.open(buffer)  # 使用 PIL 打开图像
    # 缓冲区不能重用，因为只有当image被释放后才能再次使用，同时也不要close buffer，等image使用完后才释放
    return (image, buffer)


def frame_fn(iRecord, frame_id, out=None):
    out_fn = f"{iRecord}_{frame_id}.png"
    if out is not None:
        out_fn = os.path.join(out, out_fn)
    return out_fn


class FrameDrawer:
    def __init__(self, gpu_num=0, percentile=0.995):
        self.is_init = False
        self.gpu_num = gpu_num
        self.percentile = percentile

    def init(self):
        if self.is_init:
            return
        # 初始化网格
        self.res = 0.2
        self.grid_x = np.arange(0, 1001, self.res)
        self.grid_y = np.arange(0, 101, self.res)
        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)
        # 调整网格坐标以匹配背景图片尺寸，并应用坐标变换
        self.ratio = 0.10106 * 4
        self.grid_x_scaled = self.grid_x / self.ratio
        self.grid_y_scaled = self.grid_y / self.ratio
        # 设置图形大小和分辨率
        fig_size = (16, 12)  # 将图形大小增加一倍，单位为英寸
        dpi = 300  # 设置较高的分辨率
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=fig_size, dpi=dpi)
        # 设置高斯分布参数
        self.Sr = 54
        self.L = 5
        self.par1 = 2 * 0.0064
        self.mcexp = 0.07  # updated
        self.cexp = 1.0  # updated
        self.kexp1 = 1 * 0.5
        self.kexp2 = 5 * 0.5
        self.tla = 4.0  # look ahead time in highway
        self.steering_angle = 0.1
        self.delta_fut_h = (np.pi / 180) * self.steering_angle / self.Sr
        self.delta = risk.Gaussian_3d_torus_delta(self.delta_fut_h)
        self.R = risk.Gaussian_3d_torus_R(self.L, self.delta)
        self.mexp1 = risk.Gaussian_3d_torus_mexp(self.kexp1, self.mcexp, self.delta)
        self.mexp2 = risk.Gaussian_3d_torus_mexp(self.kexp2, self.mcexp, self.delta)
        self.selected_cols = ["x", "shift_y", "speed", "heading"]

        use_gpu = torch.cuda.is_available()
        print("use_gpu: ", use_gpu)
        if use_gpu:
            self.risk_count = self._risk_count_torch
            self.calculate_risk = self.calculate_risk_torch
            self.block_size_grid = 256 * 1024
            self.block_size_vehicle = 64
            self.result_shape = self.X.shape
            self.X = self.X.reshape(-1)
            self.Y = self.Y.reshape(-1)
            device = f"cuda:{self.gpu_num}"
            torch.cuda.empty_cache()
            self.X = torch.from_numpy(self.X).to(device=device)
            self.Y = torch.from_numpy(self.Y).to(device=device)
            self.delta = torch.tensor(self.delta).to(device=device)
            self.R = torch.tensor(self.R).to(device=device)
            self.device = device
        else:
            self.risk_count = self._risk_count_cpu
            self.calculate_risk = self.calculate_risk_cpu

        self.is_init = True

    def clear_cache(self):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            torch.cuda.empty_cache()

    def calculate_risk_torch(self, tracks_by_frame: pd.DataFrame):
        acc = torch.zeros_like(self.X).to(device=self.device)
        frame_data = tracks_by_frame.values
        frame_data = torch.from_numpy(frame_data).to(device=self.device)
        with torch.no_grad():
            xv = frame_data[:, 0]
            yv = frame_data[:, 1]
            speed = frame_data[:, 2]
            heading = frame_data[:, 3]
            for _, _, start_ind, end_ind in generate_range_size(
                len(self.X), self.block_size_grid
            ):
                x_ = self.X[start_ind:end_ind]
                y_ = self.Y[start_ind:end_ind]
                for _, _, veh_start_ind, veh_end_ind in generate_range_size(
                    len(frame_data), self.block_size_vehicle
                ):
                    xv_ = xv[veh_start_ind:veh_end_ind]
                    yv_ = yv[veh_start_ind:veh_end_ind]
                    speed_ = speed[veh_start_ind:veh_end_ind]
                    heading_ = heading[veh_start_ind:veh_end_ind]
                    phiv = risk_torch.Gaussian_3d_torus_phiv(heading_)
                    xc, yc = risk_torch.Gaussian_3d_torus_xcyc(
                        xv_, yv_, phiv, self.delta, self.R
                    )
                    arc_len = risk_torch.Gaussian_3d_torus_arclen(
                        x_, y_, xv_, yv_, self.delta, xc, yc, self.R
                    )
                    dla = risk_torch.Gaussian_3d_torus_dla(self.tla, speed_)
                    a = risk_torch.Gaussian_3d_torus_a(arc_len, self.par1, dla)
                    sigma1 = risk_torch.Gaussian_3d_torus_sigma(
                        arc_len, self.mexp1, self.cexp
                    )
                    sigma2 = risk_torch.Gaussian_3d_torus_sigma(
                        arc_len, self.mexp2, self.cexp
                    )
                    Z_cur = risk_torch.Gaussian_3d_torus_z(
                        x_, y_, xc, yc, self.R, a, sigma1, sigma2
                    )
                    acc[start_ind:end_ind] = Z_cur.sum(dim=0)
            result = acc.cpu().numpy()
        torch.cuda.empty_cache()
        return result.reshape(self.result_shape)

    def _risk_count_torch(self, tracks_by_frame: pd.DataFrame):
        acc = torch.zeros_like(self.X).to(device=self.device)
        frame_data = tracks_by_frame.values
        frame_data = torch.from_numpy(frame_data).to(device=self.device)
        with torch.no_grad():
            xv = frame_data[:, 0]
            yv = frame_data[:, 1]
            speed = frame_data[:, 2]
            heading = frame_data[:, 3]
            for _, _, start_ind, end_ind in generate_range_size(
                len(self.X), self.block_size_grid
            ):
                x_ = self.X[start_ind:end_ind]
                y_ = self.Y[start_ind:end_ind]
                for _, _, veh_start_ind, veh_end_ind in generate_range_size(
                    len(frame_data), self.block_size_vehicle
                ):
                    xv_ = xv[veh_start_ind:veh_end_ind]
                    yv_ = yv[veh_start_ind:veh_end_ind]
                    speed_ = speed[veh_start_ind:veh_end_ind]
                    heading_ = heading[veh_start_ind:veh_end_ind]
                    phiv = risk_torch.Gaussian_3d_torus_phiv(heading_)
                    xc, yc = risk_torch.Gaussian_3d_torus_xcyc(
                        xv_, yv_, phiv, self.delta, self.R
                    )
                    arc_len = risk_torch.Gaussian_3d_torus_arclen(
                        x_, y_, xv_, yv_, self.delta, xc, yc, self.R
                    )
                    dla = risk_torch.Gaussian_3d_torus_dla(self.tla, speed_)
                    a = risk_torch.Gaussian_3d_torus_a(arc_len, self.par1, dla)
                    sigma1 = risk_torch.Gaussian_3d_torus_sigma(
                        arc_len, self.mexp1, self.cexp
                    )
                    sigma2 = risk_torch.Gaussian_3d_torus_sigma(
                        arc_len, self.mexp2, self.cexp
                    )
                    Z_cur = risk_torch.Gaussian_3d_torus_z(
                        x_, y_, xc, yc, self.R, a, sigma1, sigma2
                    )
                    acc[start_ind:end_ind] = Z_cur.sum(dim=0)
            q99 = torch.quantile(acc, self.percentile).item()
            result = acc.cpu().numpy()
        torch.cuda.empty_cache()
        return result.reshape(self.result_shape), q99

    def _risk_count_cpu(self, tracks_by_frame: pd.DataFrame):
        # 初始化风险值矩阵
        frame_risk = np.zeros_like(self.X)
        for val in tracks_by_frame.itertuples():
            x, y, speed, heading = val.x, val.shift_y, val.speed, val.heading
            # 计算高斯分布参数
            phiv = risk.Gaussian_3d_torus_phiv(heading)
            xc, yc = risk.Gaussian_3d_torus_xcyc(x, y, phiv, self.delta, self.R)
            arc_len = risk.Gaussian_3d_torus_arclen(
                self.X, self.Y, x, y, self.delta, xc, yc, self.R
            )
            dla = risk.Gaussian_3d_torus_dla(self.tla, speed)
            a = risk.Gaussian_3d_torus_a(arc_len, self.par1, dla)
            sigma1 = risk.Gaussian_3d_torus_sigma(arc_len, self.mexp1, self.cexp)
            sigma2 = risk.Gaussian_3d_torus_sigma(arc_len, self.mexp2, self.cexp)
            Z_cur = risk.Gaussian_3d_torus_z(
                self.X, self.Y, xc, yc, self.R, a, sigma1, sigma2
            )
            # 累加当前车辆的风险值矩阵到总的风险值矩阵中
            frame_risk += Z_cur
        return frame_risk, np.quantile(frame_risk, self.percentile)

    def calculate_risk_cpu(self, tracks_by_frame: pd.DataFrame):
        # 初始化风险值矩阵
        frame_risk = np.zeros_like(self.X)
        for val in tracks_by_frame.itertuples():
            x, y, speed, heading = val.x, val.shift_y, val.speed, val.heading
            # 计算高斯分布参数
            phiv = risk.Gaussian_3d_torus_phiv(heading)
            xc, yc = risk.Gaussian_3d_torus_xcyc(x, y, phiv, self.delta, self.R)
            arc_len = risk.Gaussian_3d_torus_arclen(
                self.X, self.Y, x, y, self.delta, xc, yc, self.R
            )
            dla = risk.Gaussian_3d_torus_dla(self.tla, speed)
            a = risk.Gaussian_3d_torus_a(arc_len, self.par1, dla)
            sigma = risk.Gaussian_3d_torus_sigma(arc_len, self.mexp, self.cexp)
            Z_cur = risk.Gaussian_3d_torus_z(
                self.X, self.Y, xc, yc, self.R, a, sigma, sigma
            )
            # 累加当前车辆的风险值矩阵到总的风险值矩阵中
            frame_risk += Z_cur
        return frame_risk

    def single_car_risk(self, car_info: Dict[str, Any]):
        if car_info is None:
            return None
        filtered_info = {k: car_info[k] for k in self.selected_cols}
        tracks_by_frame = pd.DataFrame([filtered_info], columns=self.selected_cols)
        frame_risk = self.calculate_risk(tracks_by_frame)
        return frame_risk

    def shift_pos_center(
        self, ego: Dict[str, Any], others: List[Optional[Dict[str, Any]]]
    ) -> Tuple[Dict[str, Any], List[Optional[Dict[str, Any]]]]:
        center_x = self.x_range / 2
        center_y = self.y_range / 2
        move_x = center_x - ego["x"]
        move_y = center_y - ego["shift_y"]
        ego_shift = ego.copy()
        ego_shift["x"] = center_x
        ego_shift["shift_y"] = center_y
        others_shift = []
        for vehicle in others:
            if vehicle is None:  # 处理可能的None值
                others_shift.append(None)
                continue
            shifted = vehicle.copy()
            shifted["x"] += move_x
            shifted["shift_y"] += move_y
            if (
                0 <= shifted["x"]
                and shifted["x"] <= self.x_range
                and 0 <= shifted["shift_y"]
                and shifted["shift_y"] <= self.y_range
            ):
                others_shift.append(shifted)
            else:
                others_shift.append(None)
        return ego_shift, others_shift

    def debug_draw(
        self,
        data_mgr: DataManagerHighD,
        iRecord: str,
        frame_id: int,
        car_id: int,
        car_arr: np.ndarray,
    ):
        self.init()
        out_fn = f"debug_risk/{iRecord}_{frame_id}_{car_id}.png"
        lane_y_list = data_mgr.lanes_y(iRecord)
        img = data_mgr.bg_img(iRecord)

        frame_risk = car_arr
        # 读取背景图片并获取尺寸
        img_height, img_width = img.shape[:2]
        # self.ax.cla()
        # 绘制背景图片和风险值矩阵: 数据坐标系的y轴方向被翻转
        self.ax.imshow(img, extent=[0, img_width, img_height, 0])
        # 画车道线
        for lane_y in lane_y_list:
            self.ax.axhline(
                y=lane_y / self.ratio, color="black", linewidth=2, linestyle="--"
            )
        self.ax.contourf(
            self.grid_x_scaled,
            self.grid_y_scaled,
            frame_risk,
            levels=200,
            cmap="jet",
            alpha=0.7,
        )
        # 绘制车辆位置
        # self.ax.scatter(x_array / self.ratio, y_array / self.ratio, c='white', s=20)
        self.ax.axis("off")
        self.ax.set_xlim(0, img_width)
        # 翻转 y 轴
        self.ax.set_ylim(img_height, 0)
        self.fig.savefig(out_fn, bbox_inches="tight", pad_inches=0)
        self.ax.clear()
        print(f"finish risk plot draw:{out_fn}")

    def draw_car_box_id(
        self,
        vehicle,
        fix_scale_x,
        fix_scale_y,
        clip_rect,
        box_edge_color="black",
        box_face_color=None,
        text_color="gold",
    ):
        width = vehicle.width / self.ratio
        height = vehicle.height / self.ratio
        x = vehicle.x / self.ratio
        y = vehicle.y / self.ratio
        x *= fix_scale_x
        width *= fix_scale_x
        y *= fix_scale_y
        height *= fix_scale_y

        # 创建矩形对象 (x,y):矩形左下角的坐标
        rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=1,
            edgecolor=box_edge_color,
            facecolor=box_face_color,
        )
        self.ax.add_patch(rect)

        # 在矩形框内显示车辆ID
        t = self.ax.text(
            x + width / 2,  # 矩形中心x坐标
            y + height / 2,  # 矩形中心y坐标
            str(int(vehicle.id)).strip(),  # 显示车辆ID
            fontsize=5,
            color=text_color,  # 文字颜色
            ha="center",  # 水平居中
            va="center",  # 垂直居中
            weight="bold",  # 加粗显示
        )

        if self.get_width(t) > self.get_width(rect):
            if vehicle.xVelocity > 0:
                # 向右行驶: 右对齐
                pos_x = x + width
                pos_y = y + height / 2
                t.set_horizontalalignment("right")
            else:
                # 向左行驶: 左对齐
                pos_x = x
                pos_y = y + height / 2
                t.set_horizontalalignment("left")
            t.set_position((pos_x, pos_y))

        if t is not None and clip_rect is not None:
            t.set_clip_path(clip_rect)
            t.set_clip_on(True)

            # 检查文字是否仍然超出边界
            text_bbox = t.get_window_extent()
            bbox_data = self.ax.transData.inverted().transform(text_bbox)
            # 如果仍然超出，进一步调整
            if (
                bbox_data[0, 0] < 0
                or bbox_data[1, 0] > clip_rect.get_width()
                or bbox_data[0, 1] < 0
                or bbox_data[1, 1] > clip_rect.get_height()
            ):
                t.remove()

    def get_width(self, obj):
        bbox = obj.get_tightbbox()
        if bbox is None:
            return obj.get_window_extent().width
        return bbox.width

    def draw_frame_risk(
        self,
        data_mgr: DataManagerHighD,
        iRecord: str,
        frame_id,
        bg_shape,
        out,
        overwrite=False,
    ):
        out_fn = frame_fn(iRecord, frame_id, out)
        if not overwrite and os.path.exists(out_fn):
            # 检查文件大小，如果为零则继续计算
            if os.path.getsize(out_fn) > 0:
                return
        self.init()
        tracks = data_mgr.tracks(iRecord)
        lane_y_list = data_mgr.lanes_y(iRecord)
        img = data_mgr.bg_img(iRecord)

        # 读取背景图片并获取尺寸
        img_height, img_width = img.shape[:2]
        # 遍历当前帧的所有车辆数据，计算其风险值，并累加到风险值矩阵中
        _tracks_by_frame: pd.DataFrame = tracks[tracks["frame"] == frame_id]
        tracks_by_frame = _tracks_by_frame[
            ["x", "y", "width", "height", "xVelocity", "yVelocity"]
        ].copy()
        tracks_by_frame["x"] = tracks_by_frame["x"].where(
            tracks_by_frame["xVelocity"] < 0,
            tracks_by_frame["x"] + tracks_by_frame["width"],
        )
        tracks_by_frame["shift_y"] = (
            tracks_by_frame["y"] + tracks_by_frame["height"] / 2
        )
        tracks_by_frame["heading"] = v_heading(tracks_by_frame["xVelocity"].values) * (
            np.pi / 180
        )
        tracks_by_frame["speed"] = np.sqrt(
            np.square(tracks_by_frame["xVelocity"])
            + np.square(tracks_by_frame["yVelocity"])
        )
        tracks_by_frame = tracks_by_frame[["x", "shift_y", "speed", "heading"]]

        frame_risk = self.calculate_risk(tracks_by_frame)
        x_array = tracks_by_frame["x"].values
        y_array = tracks_by_frame["shift_y"].values
        bg_img_height, bg_img_width = bg_shape
        fix_scale_x = img_width / bg_img_width
        fix_scale_y = img_height / bg_img_height

        # self.ax.cla()
        # 绘制背景图片和风险值矩阵: 数据坐标系的y轴方向被翻转
        self.ax.imshow(img, extent=[0, img_width, img_height, 0])
        # 创建剪裁矩形
        clip_rect = Rectangle(
            (0, 0), img_width, img_height, transform=self.ax.transData
        )
        # 设置轴的剪裁区域
        self.ax.set_clip_path(clip_rect)
        self.ax.set_clip_on(True)

        # 画车道线
        for lane_y in lane_y_list:
            self.ax.axhline(
                y=lane_y / self.ratio, color="black", linewidth=2, linestyle="--"
            )
        self.ax.contourf(
            self.grid_x_scaled,
            self.grid_y_scaled,
            frame_risk,
            levels=200,
            cmap="jet",
            alpha=0.7,
        )
        # 绘制所有车辆的矩形框和ID
        for vehicle in _tracks_by_frame.itertuples(index=False):
            self.draw_car_box_id(vehicle, fix_scale_x, fix_scale_y, clip_rect)

        del tracks_by_frame, _tracks_by_frame
        # 绘制车辆位置
        self.ax.scatter(x_array / self.ratio, y_array / self.ratio, c="black", s=20)
        self.ax.axis("off")
        self.ax.set_xlim(0, img_width)
        # 翻转 y 轴
        self.ax.set_ylim(img_height, 0)
        self.fig.savefig(out_fn, bbox_inches="tight", pad_inches=0)
        self.ax.clear()
        print(f"finish risk plot draw:{out_fn}")
        del frame_risk, x_array, y_array
        gc.collect()

    def focus_frame(
        self,
        tracks_range: pd.DataFrame,
        img,
        bg_shape,
        frame_id,
        vehicle_id,
        affected_id,
        dangerous_class,
        bottom_title,
        top_title,
        iRecord,
        redraw_all_cars=False,
    ):
        tracks_by_frame = tracks_range[tracks_range["frame"] == frame_id]
        bg_img_height, bg_img_width = bg_shape
        img_height, img_width = img.shape[:2]
        fix_scale_x = img_width / bg_img_width
        fix_scale_y = img_height / bg_img_height
        self.ax.imshow(img, extent=[0, img_width, img_height, 0])
        enlarge_y = img_height * 0.1
        x_min, x_max = (0, img_width)
        y_min, y_max = (-enlarge_y, img_height + enlarge_y)
        # 创建精确匹配的剪裁矩形
        clip_rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, transform=self.ax.transData
        )
        # 应用剪裁
        self.ax.set_clip_path(clip_rect)
        self.ax.set_clip_on(True)

        # 绘制所有车辆的矩形框和ID
        if redraw_all_cars:
            for vehicle in tracks_by_frame.itertuples(index=False):
                self.draw_car_box_id(vehicle, fix_scale_x, fix_scale_y, clip_rect)

        bottom_y_ratio = -0.0
        top_y_ratio = 1.0
        top_title = f"record: {iRecord}, frame: {frame_id}, {top_title}"

        prev_id = self.draw_vehicle(
            vehicle_id,
            tracks_by_frame,
            fix_scale_x,
            fix_scale_y,
            "red",
            (0.5, top_y_ratio),
            top_title,
            clip_rect,
            None,
        )
        if dangerous_class == 2:
            affected_id = prev_id

        is_bottom_drawn = False
        if affected_id > 0:
            ego_target = tracks_by_frame[tracks_by_frame["id"] == vehicle_id]
            if len(ego_target) > 0:
                ego_vehicle = convert_to_vehicle_dict(ego_target)
                affected_target = tracks_by_frame[tracks_by_frame["id"] == affected_id]
                if len(affected_target) > 0:
                    follow_vehicle = convert_to_vehicle_dict(affected_target)
                    (
                        (ttc, min_dhw),
                        (ego_attach_x, ego_attach_y),
                        (follow_attach_x, follow_attach_y),
                    ) = vehicle_min_distance_x_ttc(ego_vehicle, follow_vehicle)
                    self.draw_vehicle(
                        affected_id,
                        tracks_by_frame,
                        fix_scale_x,
                        fix_scale_y,
                        "blue",
                        (0.5, bottom_y_ratio),
                        bottom_title,
                        clip_rect,
                        (ttc, min_dhw),
                    )
                    self.draw_connection_line(
                        ego_attach_x * fix_scale_x / self.ratio,
                        ego_attach_y * fix_scale_y / self.ratio,
                        follow_attach_x * fix_scale_x / self.ratio,
                        follow_attach_y * fix_scale_y / self.ratio,
                        color="red",
                        linewidth=1,
                        clip_rect=clip_rect,
                    )
                    is_bottom_drawn = True

        if not is_bottom_drawn:
            # 底部占位，保持所有图片大小一致
            self.ax.text(
                0.5,
                bottom_y_ratio,
                "affected car not presented",
                fontsize=8,
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )

        self.ax.axis("off")
        self.ax.set_xlim(0, img_width)
        self.ax.set_ylim(img_height + enlarge_y, -enlarge_y)
        result = save_fig_array(self.fig)
        self.ax.clear()
        return result

    def draw_vehicle(
        self,
        vehicle_id,
        tracks_by_frame: pd.DataFrame,
        fix_scale_x,
        fix_scale_y,
        facecolor,
        text_pos,
        text_title,
        clip_rect,
        ttc_dhw,
    ):
        txt_x, txt_y = text_pos
        target = tracks_by_frame[tracks_by_frame["id"] == vehicle_id]
        if len(target) == 0:
            self.ax.text(
                txt_x,
                txt_y,
                "ego car not presented",
                fontsize=8,
                color=facecolor,
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            return 0
        vehicle = convert_to_vehicle_dict(target)
        self.draw_car_box_id(vehicle, fix_scale_x, fix_scale_y, clip_rect, facecolor)

        # m/s -> km/h
        v_u = 3.6
        xv = vehicle.xVelocity * v_u
        yv = vehicle.yVelocity * v_u
        xa = vehicle.xAcceleration
        ya = vehicle.yAcceleration

        if ttc_dhw is not None:
            ttc, dhw = ttc_dhw
            txt = f"{text_title}:{vehicle_id},xv:{xv:0.2f} km/h,yv:{yv:0.2f} km/h,xa:{xa:0.2f} m/s^2,ya:{ya:0.2f} m/s^2,ttc:{ttc:0.4f},distance:{dhw:0.2f}"
        else:
            txt = f"{text_title}:{vehicle_id},xv:{xv:0.2f} km/h,yv:{yv:0.2f} km/h,xa:{xa:0.2f} m/s^2,ya:{ya:0.2f} m/s^2"

        # 在图片上方添加文字
        self.ax.text(
            txt_x,
            txt_y,
            txt,
            fontsize=8,
            color=facecolor,
            ha="center",
            va="center",
            transform=self.ax.transAxes,
        )
        return int(vehicle.precedingId)

    def draw_connection_line(
        self, start_x, start_y, end_x, end_y, color="red", linewidth=1, clip_rect=None
    ):
        """
        绘制连接两点的线

        参数:
            start_x, start_y: 起点坐标
            end_x, end_y: 终点坐标
            color: 线的颜色，默认为红色
            linewidth: 线宽，默认为2
            clip_rect: 剪裁区域，如果提供则应用剪裁

        返回:
            line: matplotlib 线对象
        """
        line = self.ax.plot(
            [start_x, end_x], [start_y, end_y], color=color, linewidth=linewidth
        )[0]

        # 应用剪裁
        if clip_rect is not None:
            line.set_clip_path(clip_rect)

        return line


# preceding是主车，following是对preceding产生威胁的车
def risk_of_two(
    preceding_arr: np.ndarray,
    preceding_threshold,
    following_arr: np.ndarray,
    following_threshold,
) -> float:
    if following_arr is None or preceding_arr is None:
        return 0
    total = np.sum(preceding_arr > preceding_threshold).item()
    if total == 0:
        ratio = 0
    else:
        intersection = np.sum(
            (preceding_arr > preceding_threshold) & (following_arr > following_threshold)
        ).item()
        ratio = intersection / total
    return ratio


# car_groups: some specific (iRecord,frame_id)'s tracks group by car id
# dataframe contains lables: x, shift_y, speed, heading
def risk_by_rel(
    preceding_id: int,
    following_id: int,
    is_same_lane: bool,
    car_groups: Dict[int, pd.DataFrame],
    drawer: FrameDrawer,
    risk_arr_cache: Dict[int, Any],
    q99_cache: Dict[int, float],
    pair_cache: Dict[Tuple[int, int], Any],
    ratio=0.2,
    force_shift=False,
    overlap_vehicle_box=False,
) -> float:
    """
    car_groups: some specific (iRecord,frame_id)'s tracks group by car id

    dataframe contains lables: x, shift_y, speed, heading
    """

    preceding_id = int(preceding_id)
    following_id = int(following_id)
    if preceding_id == 0 or following_id == 0:
        return 0
    r = pair_cache.get((preceding_id, following_id), None)
    if r is not None:
        return r
    preceding_y = None
    following_y = None
    preceding_arr = risk_arr_cache.get(preceding_id, None)
    if preceding_arr is None:
        preceding_frame = car_groups.get(preceding_id)[drawer.selected_cols]
        preceding_y = preceding_frame["shift_y"].iloc[0]
        preceding_arr, preceding_threshold = drawer.risk_count(preceding_frame)
        risk_arr_cache[preceding_id] = preceding_arr
        q99_cache[preceding_id] = preceding_threshold
    following_arr = risk_arr_cache.get(following_id, None)

    following_frame = car_groups.get(following_id)[drawer.selected_cols]
    following_y = following_frame["shift_y"].iloc[0]
    if following_arr is None:
        following_arr, following_threshold = drawer.risk_count(following_frame)
        risk_arr_cache[following_id] = following_arr
        q99_cache[following_id] = following_threshold

    following_frame = car_groups.get(following_id)
    preceding_threshold = q99_cache[preceding_id]
    following_threshold = q99_cache[following_id]
    if overlap_vehicle_box:
        following_x = int(following_frame["x"].item() / ratio)
        following_x_end = int(
            (following_frame["x"].item() + following_frame["width"].item()) / ratio
        )
        following_org_y = int(following_frame["y"].item() / ratio)
        following_org_y_end = int(
            (following_frame["y"].item() + following_frame["height"].item()) / ratio
        )
        following_center_y = int(following_y / ratio)
        following_arr[following_org_y:following_org_y_end, following_x:following_x_end] = (
            following_arr[following_center_y, following_x] + 1
        )

    if force_shift or not is_same_lane:
        # 不同车道的情况下，把两辆车移动到相同的y轴上进行比较
        if preceding_y is None:
            preceding_frame = car_groups.get(preceding_id)
            preceding_y = preceding_frame["shift_y"].iloc[0]
        if following_y is None:
            following_frame = car_groups.get(following_id)
            following_y = following_frame["shift_y"].iloc[0]
        # preceding_y + shift == following_y, preceding_y_after_roll == shift / ratio, ratio == FrameDrawer.res (0.2)
        shifted_amount = int((following_y - preceding_y) / ratio)
        preceding_arr = np.roll(preceding_arr, shift=shifted_amount, axis=0)
    r = risk_of_two(preceding_arr, preceding_threshold, following_arr, following_threshold)
    pair_cache[(preceding_id, following_id)] = r
    return r

def calRiskForStationarySituation(preceding_car,following_car):
        if following_car is None or preceding_car is None:
            return 0.0

        preceding_speed_x = preceding_car.xVelocity
        following_speed_x = following_car.xVelocity

        if not(abs(preceding_speed_x) < 5):
            return None

        (
            _,  # 碰撞时间(TTC)和距离
            (ego_attach_x, _),  # 自车的接触点坐标
            (follow_attach_x, _),  # 受影响车辆的接触点坐标
        ) = vehicle_min_distance_x_ttc(preceding_car, following_car)

        rel_v_x = following_speed_x - preceding_speed_x
        if (rel_v_x <= 0.0):
            return 0.0
        distance = ego_attach_x - follow_attach_x

        if distance <= RISK_WARNING_DISTANCE:
            return 1.0

        x_ttc = distance / rel_v_x
        if x_ttc <= 3.0:
            risk_value = 1.0
        elif x_ttc >= 6.0:
            risk_value = 0.0
        else:
            risk_value = (6.0 - x_ttc) / 3.0

        return risk_value

def adjust_preceding_following(car_groups,ego_car_obj,preceding_id,alongside_id, following_id):
    if alongside_id > 0:
        alongside_car_obj = convert_to_vehicle_dict(car_groups[alongside_id])
        if ego_car_obj.xVelocity > 0 and ego_car_obj.x + ego_car_obj.width < alongside_car_obj.x + alongside_car_obj.width or \
            ego_car_obj.xVelocity < 0 and ego_car_obj.x > alongside_car_obj.x:
            # replace preceding id by alongside id
            preceding_id = alongside_id
        else:
            following_id = alongside_id
    following_car_obj = convert_to_vehicle_dict(car_groups[following_id]) if following_id > 0 else None
    preceding_car_obj = convert_to_vehicle_dict(car_groups[preceding_id]) if preceding_id > 0 else None
    return (preceding_id,following_id),(preceding_car_obj, following_car_obj)

def six_risks(
    car_groups: Dict[int, pd.DataFrame], ego_car: int, drawer: FrameDrawer, row
):
    risk_arr_cache = {}
    q99_cache = {}
    pair_cache = {}

    ego_car_obj = convert_to_vehicle_dict(car_groups[ego_car])
    precedingCar = convert_to_vehicle_dict(car_groups[row.precedingId]) if row.precedingId > 0 else None
    followingCar = convert_to_vehicle_dict(car_groups[row.followingId]) if row.followingId > 0 else None

    precedingRisk = calRiskForStationarySituation(precedingCar, ego_car_obj)
    precedingRisk = precedingRisk if precedingRisk is not None else risk_by_rel(
        ego_car,
        row.precedingId,
        True,
        car_groups,
        drawer,
        risk_arr_cache,
        q99_cache,
        pair_cache,
    )
    followingRisk = calRiskForStationarySituation(ego_car_obj, followingCar)
    followingRisk = followingRisk if followingRisk is not None else risk_by_rel(
        row.followingId,
        ego_car,
        True,
        car_groups,
        drawer,
        risk_arr_cache,
        q99_cache,
        pair_cache,
    )

    (leftPrecedingId,leftFollowingId),(leftPrecedingCar,leftFollowingCar) = adjust_preceding_following(
        car_groups, ego_car_obj, row.leftPrecedingId, row.leftAlongsideId, row.leftFollowingId)

    leftPrecedingRisk = calRiskForStationarySituation(leftPrecedingCar, ego_car_obj)
    leftPrecedingRisk = leftPrecedingRisk if leftPrecedingRisk is not None else risk_by_rel(
        ego_car,
        leftPrecedingId,
        False,
        car_groups,
        drawer,
        risk_arr_cache,
        q99_cache,
        pair_cache,
    )
    leftFollowingRisk = calRiskForStationarySituation(ego_car_obj, leftFollowingCar)
    leftFollowingRisk = leftFollowingRisk if leftFollowingRisk is not None else risk_by_rel(
        leftFollowingId,
        ego_car,
        False,
        car_groups,
        drawer,
        risk_arr_cache,
        q99_cache,
        pair_cache,
    )

    (rightPrecedingId,rightFollowingId),(rightPrecedingCar,rightFollowingCar) = adjust_preceding_following(
        car_groups, ego_car_obj, row.rightPrecedingId, row.rightAlongsideId, row.rightFollowingId)
    rightPrecedingRisk = calRiskForStationarySituation(rightPrecedingCar, ego_car_obj)
    rightPrecedingRisk = rightPrecedingRisk if rightPrecedingRisk is not None else risk_by_rel(
        ego_car,
        rightPrecedingId,
        False,
        car_groups,
        drawer,
        risk_arr_cache,
        q99_cache,
        pair_cache,
    )
    rightFollowingRisk = calRiskForStationarySituation(ego_car_obj, rightFollowingCar)
    rightFollowingRisk = rightFollowingRisk if rightFollowingRisk is not None else risk_by_rel(
        rightFollowingId,
        ego_car,
        False,
        car_groups,
        drawer,
        risk_arr_cache,
        q99_cache,
        pair_cache,
    )

    return (
        row.precedingId,row.followingId,leftPrecedingId,leftFollowingId,rightPrecedingId,rightFollowingId
    ),(precedingCar,followingCar,leftPrecedingCar,leftFollowingCar,rightPrecedingCar,rightFollowingCar
    ),(
        precedingRisk,
        followingRisk,
        leftPrecedingRisk,
        leftFollowingRisk,
        rightPrecedingRisk,
        rightFollowingRisk,
    )


def car_surrounding_six_risks(frames_groups, car_id, frame_id, drawer: FrameDrawer):
    car_frames = frames_groups[frame_id].copy()
    car_frames["shift_y"] = car_frames["y"] + car_frames["height"] / 2
    car_frames["heading"] = v_heading(car_frames["xVelocity"].values) * (np.pi / 180)
    car_frames["speed"] = np.sqrt(
        np.square(car_frames["xVelocity"]) + np.square(car_frames["yVelocity"])
    )
    car_groups = {car_id: group for car_id, group in car_frames.groupby("id")}
    car_df = car_groups[car_id]
    info_int = car_df[
        [
            "frame",
            "laneId",
            "precedingId",
            "followingId",
            "leftPrecedingId",
            "leftAlongsideId",
            "leftFollowingId",
            "rightPrecedingId",
            "rightAlongsideId",
            "rightFollowingId",
        ]
    ]
    row = info_int.to_dict("records")[0]
    row["id"] = car_id
    EightTuple = namedtuple("EightTuple", row.keys())
    eight = EightTuple(**row)
    id_lst,car_lst,risk_lst = six_risks(car_groups, car_id, drawer, eight)
    avg_x_speed = 0.0
    surrounding_count = 0
    x_y_xv_lst = []
    for surrounding_car in car_lst:
        if surrounding_car is not None:
            avg_x_speed += surrounding_car.xVelocity
            surrounding_count += 1
            x_y_xv_lst.append(surrounding_car.x)
            x_y_xv_lst.append(surrounding_car.y)
            x_y_xv_lst.append(surrounding_car.xVelocity)
        else:
            x_y_xv_lst.append(None)
            x_y_xv_lst.append(None)
            x_y_xv_lst.append(None)
    if surrounding_count > 0:
        avg_x_speed /= surrounding_count

    return id_lst + risk_lst + (avg_x_speed, surrounding_count)+ tuple(x_y_xv_lst)


def draw_one_seg_frames(
    data_mgr: DataManagerHighD,
    out,
    drawer: FrameDrawer,
    iRecord,
    start_change,
    end_change,
    bg_shape,
):
    for frame_id in range(start_change, end_change + 1):
        drawer.draw_frame_risk(
            data_mgr, iRecord, frame_id, bg_shape, out, overwrite=False
        )


def draw_segment_animation(
    data_mgr: DataManagerHighD,
    out,
    drawer: FrameDrawer,
    slow_down_ratio,
    iRecord,
    start_change,
    end_change,
    dangerous_class,
    vehicle_id,
    follow_id,
    dec_frame,
    llm_frame,
):
    if dangerous_class == 1:
        bottom_title = "follow id"
        top_title = "bad lane changes - id"
    elif dangerous_class == 2:
        bottom_title = "preceding id"
        top_title = "too closed - id"
    else:
        bottom_title = "affected car id"
        top_title = "ego car id"

    bg_shape = data_mgr.bg_img(iRecord).shape[:2]
    draw_one_seg_frames(
        data_mgr, out, drawer, iRecord, start_change, end_change, bg_shape
    )

    tracks = data_mgr.tracks(iRecord)
    tracks_range = tracks[
        (start_change <= tracks["frame"]) & (tracks["frame"] <= end_change)
    ]

    img = data_mgr.bg_img(iRecord)
    ff, bio = drawer.focus_frame(
        tracks_range,
        img,
        bg_shape,
        dec_frame,
        vehicle_id,
        follow_id,
        dangerous_class,
        bottom_title,
        top_title,
        iRecord,
        redraw_all_cars=True,
    )
    save_image_file(
        ff,
        f"dec_{iRecord}_{dec_frame}_{vehicle_id}_{follow_id}.png",
        out,
    )
    bio.close()
    img = data_mgr.bg_img(iRecord)
    ff, bio = drawer.focus_frame(
        tracks_range,
        img,
        bg_shape,
        llm_frame,
        vehicle_id,
        follow_id,
        dangerous_class,
        bottom_title,
        top_title,
        iRecord,
        redraw_all_cars=True,
    )
    save_image_file(
        ff,
        f"llm_{iRecord}_{llm_frame}_{vehicle_id}_{follow_id}.png",
        out,
    )
    bio.close()

    max_width = 0
    max_height = 0
    images: list[Image.Image] = []
    mem_bufs: list[BytesIO] = []
    prev = None
    prev_ff = None
    for frame_id in range(start_change, end_change + 1):
        img_fn = frame_fn(iRecord, frame_id, out)
        img = plt.imread(img_fn)
        ff, bio = drawer.focus_frame(
            tracks_range,
            img,
            bg_shape,
            frame_id,
            vehicle_id,
            follow_id,
            dangerous_class,
            bottom_title,
            top_title,
            iRecord,
        )
        if prev is not None and prev_ff is not None and prev != ff.size:
            save_image_file(
                ff,
                f"debug_focus_{iRecord}_{frame_id}_{vehicle_id}_{follow_id}.png",
                out,
            )
            save_image_file(
                prev_ff,
                f"debug_focus_{iRecord}_{frame_id - 1}_{vehicle_id}_{follow_id}.png",
                out,
            )

        prev = ff.size
        prev_ff = ff
        # 确定最大尺寸并调整为16的倍数
        max_width = max(ff.size[0], max_width)
        max_height = max(ff.size[1], max_height)
        images.append(ff)
        mem_bufs.append(bio)
    output_path = f"{iRecord}_{start_change}_{end_change}_{vehicle_id}_{follow_id}.mp4"
    if out is not None:
        output_path = os.path.join(out, output_path)
    frame_rate = data_mgr.frame_rate(iRecord)

    try:
        imageio.mimsave(output_path, images, fps=frame_rate / slow_down_ratio)
        print(f"动图已保存至 {output_path}")
    except Exception as e:
        print(f"保存动图{output_path}时出错：{e}")
        # 调整为16的倍数
        target_width = ((max_width + 15) // 16) * 16
        target_height = ((max_height + 15) // 16) * 16
        temp_images: list[Image.Image] = images
        images = []
        # 创建统一尺寸的图像
        for img in temp_images:
            if img.size != (target_width, target_height):
                # 创建新的目标尺寸图像
                resized_img = Image.new(img.mode, (target_width, target_height))
                # 计算居中位置
                x_offset = (target_width - img.size[0]) // 2
                y_offset = (target_height - img.size[1]) // 2
                # 将原始图像粘贴到中心位置
                resized_img.paste(img, (x_offset, y_offset))
                images.append(resized_img)
            else:
                images.append(img)
        del temp_images
        try:
            imageio.mimsave(output_path, images, fps=frame_rate / slow_down_ratio)
            print(f"动图已保存至 {output_path}")
        except Exception as e:
            i = 0
            for frame_id in range(start_change, end_change + 1):
                img = images[i]
                print(f"{iRecord}_{frame_id}:{img.size}")
                i += 1
            sys.stdout.flush()

    del images
    for bio in mem_bufs:
        bio.close()
    del mem_bufs
    gc.collect()


def save_image_file(ff: ImageFile, focus_fn: str, out: Union[str, None]):
    if out is not None:
        focus_fn = os.path.join(out, focus_fn)
    ff.save(focus_fn)
    print(f"{focus_fn} is written")


# 假设按照起点排序
# segments: List[Tuple[Any,Any]]
def merge_segments(segments):
    if len(segments) == 0:
        return []
    merged = []
    current_segment = segments[0]

    for segment in segments[1:]:
        start, end = segment
        current_start, current_end = current_segment
        # 如果当前线段与前一个线段重叠或相邻
        if start <= current_end:
            # 合并线段
            current_segment = [min(current_start, start), max(current_end, end)]
        else:
            # 如果不重叠，将当前线段添加到结果中
            merged.append(current_segment)
            current_segment = segment
    # 添加最后一个线段
    merged.append(current_segment)
    return merged


def parse_lane_marks(whole_marks_string: str) -> List[Tuple[int, float, float]]:
    lane_mark: str = whole_marks_string
    prev = 0
    lane_id = 0
    lane_marks = []
    for mark_label in lane_mark.split(";"):
        cur = float(mark_label)
        lane_id += 1
        if prev < cur - 4.5:
            prev = cur - 4.0
        lane_marks.append((lane_id, prev, cur))
        prev = cur
    return lane_marks


# 常量声明
RISK_WARNING_DISTANCE = 15  # 风险警示距离
SAFE_DISTANCE_THRESHOLD = 30  # 安全距离阈值


def calculate_ego_pos_risk(v_ego, x_ego, front_vehicle, behind_vehicle):
    """
    计算 ego 车的风险值 risk_ego。

    参数:
    - v_ego:  ego 车速度
    - x_ego: ego 车 x 坐标
    - front_vehicle: 前车对象:车速, x坐标（可能为 None)
    - behind_vehicle: 后车对象:车速, x坐标（可能为 None)

    返回:
    - risk_ego: ego 车的风险值，范围 [0, 1]
    """
    # print("in calculate_ego_pos_risk")
    # 获取 ego 车的速度和位置
    # print(f"ego_vehicle x: {x_ego}, speed: {v_ego}")

    # 初始化风险值
    risk_ego_f = 0.0  # 前车风险
    risk_ego_b = 0.0  # 后车风险

    # 处理前车风险
    if front_vehicle is not None and len(front_vehicle) > 0:
        v_f = front_vehicle["speed"].item()  # 前车速度
        x_f = front_vehicle["x"].item()  # 前车 x 坐标

        # 计算相对速度和相对距离
        v_rf = v_ego - v_f
        s_rf = 1 * v_rf  # 相对速度对应的 1 秒钟距离
        distance_to_front = (x_f - x_ego) * sign(v_ego)  # ego 和前车的距离

        # 这里判断一下几样东西。
        # 1. 如果distance_to_front <= 0，用print打印异常。返回risk_ego=1.0
        if distance_to_front <= 0:
            return 1.0  # 返回风险值为 1.0

        # 2. 如果v_rf < 0，说明前车比ego还快，取v_rf=0
        if v_rf < 0:
            v_rf = 0
            s_rf = 0

        # 3. 如果distance_to_front >= SAFE_DISTANCE_THRESHOLD，说明前车距离ego车很远，取risk_ego_f=0
        if distance_to_front >= SAFE_DISTANCE_THRESHOLD:
            risk_ego_f = 0
        elif (
            s_rf > 2 * distance_to_front or distance_to_front < RISK_WARNING_DISTANCE
        ):  # 风险警示距离
            risk_ego_f = 1
        else:
            # just keep it simple. the number 0.34 refer to edge definition
            risk_ego_f = 0.34

    # 处理后车风险
    if behind_vehicle is not None and len(behind_vehicle) > 0:
        v_b = behind_vehicle["speed"].item()  # 后车速度
        x_b = behind_vehicle["x"].item()  # 后车 x 坐标

        # 计算相对速度和相对距离
        v_rb = v_b - v_ego
        s_rb = 1 * abs(v_rb)  # 相对速度对应的 1 秒钟距离
        distance_to_behind = (x_ego - x_b) * sign(v_ego)  # ego 和后车的距离

        # 1. 如果 distance_to_behind <= 0，用 print 打印异常，返回 risk_ego = 1.0
        if distance_to_behind <= 0:
            return 1.0  # 返回风险值为 1.0

        # 2. 如果 v_rb < 0，说明后车比 ego 车慢，取 v_rb = 0
        if v_rb < 0:
            v_rb = 0
            s_rb = 0

        # 3. 如果 distance_to_behind >= SAFE_DISTANCE_THRESHOLD，说明后车距离 ego 车很远，取 risk_ego_b = 0
        if distance_to_behind >= SAFE_DISTANCE_THRESHOLD:
            risk_ego_b = 0
        elif (
            s_rb > 2 * distance_to_behind or distance_to_behind < RISK_WARNING_DISTANCE
        ):  # 风险警示距离
            risk_ego_b = 1
        else:
            risk_ego_b = 0.34

    # 计算总风险值
    risk_ego = risk_ego_f + risk_ego_b
    if risk_ego > 1:
        risk_ego = 1.0

    # 如果前后都没有车，风险值为 0
    if front_vehicle is None and behind_vehicle is None:
        risk_ego = 0

    return risk_ego


def compute_front_behind(
    precedingId, followingId, alongSideId, track_frame: pd.DataFrame, ego_x, ego_speed
):
    front = track_frame[track_frame["id"] == precedingId]
    behind = track_frame[track_frame["id"] == followingId]
    if alongSideId > 0:
        alongside = track_frame[track_frame["id"] == alongSideId]
        alongside_x = alongside["x"].item()+ alongside["width"].item() if alongside["xVelocity"].item() > 0 else alongside["x"].item()
        if (
            ego_speed > 0
            and alongside_x > ego_x
            or ego_speed < 0
            and alongside_x <= ego_x
        ):
            # alongside 车辆在 ego 前
            front = alongside
        else:
            # alongside 车辆在 ego 后
            behind = alongside
    return front, behind


def get_risk_pattern(
    track_frame: pd.DataFrame,
    min_turn: Dict,
    left_behind_risk,
    left_front_risk,
    behind_risk,
    front_risk,
    right_behind_risk,
    right_front_risk,
    keep_left,
    keep_right,
    keep_left2,
    keep_right2,
):
    track_frame["speed"] = np.sqrt(
        np.square(track_frame["xVelocity"]) + np.square(track_frame["yVelocity"])
    )
    ego = track_frame[track_frame["id"] == min_turn["id"]]
    ego_x = ego["x"].item() + ego["width"].item() if ego["xVelocity"].item() > 0 else ego["x"].item()
    ego_speed = ego["xVelocity"].item()
    front_vehicle = track_frame[track_frame["id"] == min_turn["precedingId"]]
    behind_vehicle = track_frame[track_frame["id"] == min_turn["followingId"]]
    left_front_vehicle, left_behind_vehicle = compute_front_behind(
        min_turn["leftPrecedingId"],
        min_turn["leftFollowingId"],
        0,
        track_frame,
        ego_x,
        ego_speed,
    )

    right_front_vehicle, right_behind_vehicle = compute_front_behind(
        min_turn["rightPrecedingId"],
        min_turn["rightFollowingId"],
        0,
        track_frame,
        ego_x,
        ego_speed,
    )

    ego_pos_risk = calculate_ego_pos_risk(
        ego_speed, ego_x, front_vehicle, behind_vehicle
    )
    ego_posL_risk = calculate_ego_pos_risk(
        ego_speed, ego_x, left_front_vehicle, left_behind_vehicle
    )
    ego_posR_risk = calculate_ego_pos_risk(
        ego_speed, ego_x, right_front_vehicle, right_behind_vehicle
    )
    ego_posLL_risk = 0.0
    ego_posRR_risk = 0.0

    risk_pattern = np.array(
        [
            [0.67, round(ego_posLL_risk, 2), 0.67],
            [
                round(left_behind_risk, 2),
                round(ego_posL_risk, 2),
                round(left_front_risk, 2),
            ],
            [round(behind_risk, 2), round(ego_pos_risk, 2), round(front_risk, 2)],
            [
                round(right_behind_risk, 2),
                round(ego_posR_risk, 2),
                round(right_front_risk, 2),
            ],
            [0.67, round(ego_posRR_risk, 2), 0.67],
        ],
        dtype=np.float32,
    )

    # 判断当前 ego 的左边的左边车道是否可以行驶
    if not keep_left2:
        risk_pattern[0, 0] = 1.00
        risk_pattern[0, 1] = 1.00
        risk_pattern[0, 2] = 1.00

    # 判断当前 ego 的左边车道是否可以行驶
    if not keep_left:
        risk_pattern[1, 0] = 1.00
        risk_pattern[1, 1] = 1.00
        risk_pattern[1, 2] = 1.00

    # 判断当前 ego 的右边的右边车道是否可以行驶
    if not keep_right2:
        risk_pattern[4, 0] = 1.00
        risk_pattern[4, 1] = 1.00
        risk_pattern[4, 2] = 1.00

    # 判断当前 ego 的右边车道是否可以行驶
    if not keep_right:
        risk_pattern[3, 0] = 1.00
        risk_pattern[3, 1] = 1.00
        risk_pattern[3, 2] = 1.00

    edges = np.array(
        [0.34, 0.67, 1.00], dtype=np.float32
    )
    quantised = np.digitize(risk_pattern, edges).astype(np.float32)  # 0,1,2,3
    # risk_pattern_vec = quantised.flatten()          # shape (15,)
    return quantised


def cal_draw_start(dec_frame, segment_start, lane_change_frame):
    start_change = min(dec_frame, lane_change_frame)
    if segment_start <= start_change - 25:
        start_change -= 25
    elif start_change > segment_start:
        start_change = segment_start
    return start_change
