"""
This file is adapted from DiLu's `envScenario.py` (PJLab-ADG/DiLu, Apache-2.0),
with substantial modifications for RESPOND.

Modifications Copyright 2025 Dan Chen.
"""
from typing import List, Tuple, Optional, Union, Dict
from datetime import datetime
import math
import numpy as np
import os
import matplotlib.pyplot as plt  # Import for heat map visualization

from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.road.lane import (
    StraightLane, CircularLane, SineLane, PolyLane, PolyLaneFixedWidth
)

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.controller import MDPVehicle,ControlledVehicle
from highway_env.vehicle.behavior import IDMVehicle

from dilu.scenario.DBBridge import DBBridge
from respond.scenario.envPlotter import ScePlotter
from respond.scenario.utils import vehicleKeyInfo, vehicleInfo, vehicleKeyInfoMore, risk_of_two, printCarInfo
from respond.risk_pattern.risk_calculator import riskValueCal, getSurroundingRiskPattern
import respond.risk_pattern.risk as risk


LOOK_SURROUND_V_NUM = 15


ACTIONS_ALL = {
    0: 'LANE_LEFT',    
    1: 'IDLE',         
    2: 'LANE_RIGHT',   
    3: 'FASTER',       
    4: 'SLOWER'        
}


ACTIONS_DESCRIPTION = {
    0: 'Turn-left - change lane to the left of the current lane',
    1: 'IDLE - remain in the current lane with current speed',
    2: 'Turn-right - change lane to the right of the current lane',
    3: 'Acceleration - accelerate the vehicle',
    4: 'Deceleration - decelerate the vehicle'
}


class EnvScenario:
    def __init__(
            self, env: AbstractEnv, envType: str,
            seed: int, database: str = None
    ) -> None:

        self.env = env
        self.envType = envType

        self.ego: MDPVehicle = env.vehicle
        

        self.theta1 = math.atan(3/17.5)
        self.theta2 = math.atan(2/2.5)
        self.radius1 = np.linalg.norm([3, 17.5])
        self.radius2 = np.linalg.norm([2, 2.5])

        self.road: Road = env.road
        self.network: RoadNetwork = self.road.network

        self.plotter = ScePlotter()
        

        if database:
            self.database = database
        else:
            self.database = datetime.strftime(
                datetime.now(), '%Y-%m-%d_%H-%M-%S'
            ) + '.db'

 
        if os.path.exists(self.database):
            os.remove(self.database)


        self.dbBridge = DBBridge(self.database, env)


        self.dbBridge.createTable()
        self.dbBridge.insertSimINFO(envType, seed)
        self.dbBridge.insertNetwork()

    def getSurroundVehicles(self, vehicles_count: int) -> List[IDMVehicle]:

        return self.road.close_vehicles_to(
            self.ego, self.env.PERCEPTION_DISTANCE,
            count=vehicles_count-1, see_behind=True,
            sort='sorted'
        )

    def plotSce(self, fileName: str) -> None:


        SVs = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)

        self.plotter.plotSce(self.network, SVs, self.ego, fileName)

    def plotSce2(self, fileName: str, SVs: List[IDMVehicle], isFinal: bool = False) -> Tuple[str, Optional[IDMVehicle]]:

        if isFinal:

            return self.plotter.plotSce2(self.network, SVs, self.ego, fileName, isFinal)
        else:
            self.plotter.plotSce2(self.network, SVs, self.ego, fileName)

        return "not final, not using", None

    def getUnitVector(self, radian: float) -> Tuple[float]:

        return (
            math.cos(radian), math.sin(radian)
        )

    def isInJunction(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> float:

        if self.envType == 'intersection-v1':
            x, y = vehicle.position

            if -20 <= x <= 20 and -20 <= y <= 20:
                return True
            else:
                return False
        else:
            return False

    def getLanePosition(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> float:

        currentLaneIdx = vehicle.lane_index
        currentLane = self.network.get_lane(currentLaneIdx)
        if not isinstance(currentLane, StraightLane):
            raise ValueError(
                "The vehicle is in a junction, can't get lane position"
            )
        else:
            currentLane = self.network.get_lane(vehicle.lane_index)
            return np.linalg.norm(vehicle.position - currentLane.start)

    def availableActionsDescription(self) -> str:

        avaliableActionDescription = 'Your available actions are: \n'
        availableActions = self.env.get_available_actions()
        for action in availableActions:
            avaliableActionDescription += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(
                action) + '\n'
        # if 1 in availableActions:
        #     avaliableActionDescription += 'You should check IDLE action as FIRST priority. '
        # if 0 in availableActions or 2 in availableActions:
        #     avaliableActionDescription += 'For change lane action, CAREFULLY CHECK the safety of vehicles on target lane. '
        # if 3 in availableActions:
        #     avaliableActionDescription += 'Consider acceleration action carefully. '
        # if 4 in availableActions:
        #     avaliableActionDescription += 'The deceleration action is LAST priority. '
        # avaliableActionDescription += '\n'
        return avaliableActionDescription

    def processNormalLane(self, lidx: LaneIndex) -> str:

        sideLanes = self.network.all_side_lanes(lidx)
        numLanes = len(sideLanes)
        

        if numLanes == 1:
            description = "You are driving on a road with only one lane, you can't change lane. "
        else:

            egoLaneRank = lidx[2]
            if egoLaneRank == 0:
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the leftmost lane. "
            elif egoLaneRank == numLanes - 1:
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the rightmost lane. "
            else:

                laneRankDict = {
                    1: 'second',
                    2: 'third',
                    3: 'fourth',
                    4: 'fifth',
                    5: 'sixth',
                    6: 'seventh'
                }
                description = f"You are driving on a road with {numLanes} lanes, and you are currently driving in the {laneRankDict[egoLaneRank]} lane from the left. "


        description += f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, speed is {self.ego.speed:.2f} m/s, acceleration is {self.ego.action['acceleration']:.2f} m/s^2, and lane position is {self.getLanePosition(self.ego):.2f} m.\n"
        return description

    def isAheadOfEgo(self, sv: IDMVehicle) -> bool:
        isAhead = sv.position[0] > self.ego.position[0]
        return isAhead


    def getSVRelativeState(self, sv: IDMVehicle) -> str:


        relativePosition = sv.position - self.ego.position
        egoUnitVector = self.getUnitVector(self.ego.heading)
 
        cosineValue = sum(
            [x*y for x, y in zip(relativePosition, egoUnitVector)]
        )
        if cosineValue >= 0:
            return 'is ahead of you'
        else:
            return 'is behind of you'

    def getVehDis(self, veh: IDMVehicle):

        posA = self.ego.position
        posB = veh.position
        distance = np.linalg.norm(posA - posB)
        return distance

    def getClosestSV(self, SVs: List[IDMVehicle]):

        if SVs:
            closestIdex = -1
            closestDis = 99999999
            for i, sv in enumerate(SVs):
                dis = self.getVehDis(sv)
                if dis < closestDis:
                    closestDis = dis
                    closestIdex = i
            return SVs[closestIdex]
        else:
            return None

    def processSingleLaneSVs(self, SingleLaneSVs: List[IDMVehicle]):


        if SingleLaneSVs:
            aheadSVs = []
            behindSVs = []

            for sv in SingleLaneSVs:
                RSStr = self.getSVRelativeState(sv)
                if RSStr == 'is ahead of you':
                    aheadSVs.append(sv)
                else:
                    behindSVs.append(sv)

            aheadClosestOne = self.getClosestSV(aheadSVs)
            behindClosestOne = self.getClosestSV(behindSVs)
            return aheadClosestOne, behindClosestOne
        else:
            return None, None

    def processSVsNormalLane(
            self, SVs: List[IDMVehicle], currentLaneIndex: LaneIndex
    ):

        classifiedSVs: Dict[str, List[IDMVehicle]] = {
            'current lane': [],
            'left lane': [],
            'right lane': [],
            'target lane': []
        }
        

        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        

        for sv in SVs:
            lidx = sv.lane_index
            if lidx in sideLanes:
                if lidx == currentLaneIndex:
                    classifiedSVs['current lane'].append(sv)
                else:
                    laneRelative = lidx[2] - currentLaneIndex[2]
                    if laneRelative == 1:
                        classifiedSVs['right lane'].append(sv)
                    elif laneRelative == -1:
                        classifiedSVs['left lane'].append(sv)
                    else:
                        continue
            elif lidx == nextLane:
                classifiedSVs['target lane'].append(sv)
            else:
                continue


        validVehicles: List[IDMVehicle] = []
        existVehicles: Dict[str, bool] = {}

        for k, v in classifiedSVs.items():
            if v:
                existVehicles[k] = True
            else:
                existVehicles[k] = False
            ahead, behind = self.processSingleLaneSVs(v)
            if ahead:
                validVehicles.append(ahead)
            if behind:
                validVehicles.append(behind)

        return validVehicles, existVehicles

    def describeSVNormalLane(self, currentLaneIndex: LaneIndex) -> str:

        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        
        # print(f"Side lanes: {sideLanes}")

        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )
        # print(f"Next lane: {nextLane}")


        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)

        validVehicles, existVehicles = self.processSVsNormalLane(
            surroundVehicles, currentLaneIndex
        )

        # print(f"Valid vehicles: {validVehicles}")
        # print(f"Exist vehicles: {existVehicles}")

        
        if not surroundVehicles:
            SVDescription = "There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n"
            return SVDescription
        else:
            SVDescription = ''

            for sv in surroundVehicles:
                lidx = sv.lane_index
                if lidx in sideLanes:

                    if lidx == currentLaneIndex:

                        if sv in validVehicles:
                            SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the same lane as you and {self.getSVRelativeState(sv)}. "
                        else:
                            continue
                    else:
                        laneRelative = lidx[2] - currentLaneIndex[2]
                        if laneRelative == 1:

                            if sv in validVehicles:
                                SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the lane to your right and {self.getSVRelativeState(sv)}. "
                            else:
                                continue
                        elif laneRelative == -1:

                            if sv in validVehicles:
                                SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on the lane to your left and {self.getSVRelativeState(sv)}. "
                            else:
                                continue
                        else:

                            continue
                elif lidx == nextLane:

                    if sv in validVehicles:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on your target lane and {self.getSVRelativeState(sv)}. "
                    else:
                        continue
                else:
                    continue
                if self.envType == 'intersection-v1':
                    SVDescription += f"The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, acceleration is {sv.action['acceleration']:.2f} m/s^2.\n"
                else:
                    SVDescription += f"The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, acceleration is {sv.action['acceleration']:.2f} m/s^2, and lane position is {self.getLanePosition(sv):.2f} m.\n"
            

            if SVDescription:
                descriptionPrefix = "There are other vehicles driving around you, and below is their basic information:\n"
                return descriptionPrefix + SVDescription
            else:
                SVDescription = 'There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n'
                return SVDescription

    def isInDangerousArea(self, sv: IDMVehicle) -> bool:

        relativeVector = sv.position - self.ego.position
        distance = np.linalg.norm(relativeVector)
        egoUnitVector = self.getUnitVector(self.ego.heading)
        relativeUnitVector = relativeVector / distance

        alpha = np.arccos(
            np.clip(np.dot(egoUnitVector, relativeUnitVector), -1, 1)
        )

        if alpha <= self.theta1:
            if distance <= self.radius1:
                return True
            else:
                return False
        elif self.theta1 < alpha <= self.theta2:
            if distance <= self.radius2:
                return True
            else:
                return False
        else:
            return False

    def describeSVJunctionLane(self, currentLaneIndex: LaneIndex) -> str:

        nextLane = self.network.next_lane(
            currentLaneIndex, self.ego.route, self.ego.position
        )

        surroundVehicles = self.getSurroundVehicles(6)
        
        if not surroundVehicles:
            SVDescription = "There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n"
            return SVDescription
        else:
            SVDescription = ''
            for sv in surroundVehicles:
                lidx = sv.lane_index
                if self.isInJunction(sv):
                    collisionPoint = self.getCollisionPoint(sv)
                    if collisionPoint:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is also in the junction and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. The potential collision point is `({collisionPoint[0]:.2f}, {collisionPoint[1]:.2f})`.\n"
                    else:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is also in the junction and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. You two are no potential collision.\n"
                elif lidx == nextLane:
                    collisionPoint = self.getCollisionPoint(sv)
                    if collisionPoint:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on your target lane and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. The potential collision point is `({collisionPoint[0]:.2f}, {collisionPoint[1]:.2f})`.\n"
                    else:
                        SVDescription += f"- Vehicle `{id(sv) % 1000}` is driving on your target lane and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. You two are no potential collision.\n"
                if self.isInDangerousArea(sv):
                    print(f"Vehicle {id(sv) % 1000} is in dangerous area.")
                    SVDescription += f"- Vehicle `{id(sv) % 1000}` is also in the junction and {self.getSVRelativeState(sv)}. The position of it is `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, speed is {sv.speed:.2f} m/s, and acceleration is {sv.action['acceleration']:.2f} m/s^2. This car is within your field of vision, and you need to pay attention to its status when making decisions.\n"
                else:
                    continue
            if SVDescription:
                descriptionPrefix = "There are other vehicles driving around you, and below is their basic information:\n"
                return descriptionPrefix + SVDescription
            else:
                'There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n'
                return SVDescription

    def visualizeEgoRiskField(self, frame_name: str) -> None:
        """Visualize the ego car's risk field as a heat map."""
        # Extract ego car's information
        ego_info = vehicleKeyInfo(self.ego)
        x = ego_info['x']
        # y = ego_info['shift_y']
        y = self.ego.position[1] 

        print("in visualizeEgoRiskField")
        print(f"carId: {ego_info['carId']}")
        print(f"x: {ego_info['x']}")
        print(f"y: {y}")
        print(f"shift_y: {ego_info['shift_y']}")

        speed = ego_info['speed']
        heading_angle = ego_info['heading']

        # Call the field_distribution function from risk.py
        risk_field = risk.field_distributionNew(
            x=x, y=y, speed=speed, heading_angle=heading_angle,
            turning_angle=0.1, vehicle_length=5, common_grid=None
        )
        print("risk_field: ")
        # print(risk_field)
        print(f"Risk field type: {type(risk_field)}, shape: {getattr(risk_field, 'shape', 'N/A')}")

        # Visualize the risk field as a heat map
        # plt.figure(figsize=(8, 6))
        plt.figure(figsize=(1.5, 6))
        plt.imshow(risk_field, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Risk Intensity')
        plt.title('Ego Car Risk Field Heat Map')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        

        # Create the result folder if it does not exist
        result_folder = "results"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Save the heat map as a picture
        file_path = os.path.join(result_folder, f"risk_field_{frame_name}.png")
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Heat map saved to {file_path}")

        plt.show()
    
    def visualizeEgoRiskField2(self, frame_name: str) -> None:
        """Visualize the ego car's risk field as a heat map."""
        # Extract ego car's information
        ego_info = vehicleKeyInfo(self.ego)
        x = ego_info['x']
        # y = ego_info['shift_y']
        y = self.ego.position[1] 

        print("in visualizeEgoRiskField")
        print(f"carId: {ego_info['carId']}")
        print(f"x: {ego_info['x']}")
        print(f"y: {y}")
        print(f"shift_y: {ego_info['shift_y']}")

        speed = ego_info['speed']
        heading_angle = ego_info['heading']

        # Call the field_distribution function from risk.py
        X_common, Y_common, risk_field = risk.field_distributionNew(
            x=x, y=y, speed=speed, heading_angle=heading_angle,
            turning_angle=0.1, vehicle_length=5, common_grid=None
        )
        print("risk_field2: ")
        # print(risk_field)
        print(f"Risk field type: {type(risk_field)}, shape: {getattr(risk_field, 'shape', 'N/A')}")

        # Visualize the risk field as a heat map
        # plt.figure(figsize=(8, 6))
        plt.figure(figsize=(1.5, 6))
        plt.imshow(risk_field, cmap='hot', interpolation='nearest', extent=[X_common.min(), X_common.max(), Y_common.min(), Y_common.max()])
        plt.colorbar(label='Risk Intensity')
        plt.title('Ego Car Risk Field Heat Map')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        

        # Create the result folder if it does not exist
        result_folder = "results"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Save the heat map as a picture
        file_path = os.path.join(result_folder, f"risk_field_{frame_name}.png")
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Heat map saved to {file_path}")

        plt.show()
    
    def visualizeEgoRiskField3(self, frame_name: str) -> None:
        """Visualize the ego car's risk field as a heat map and as a contour shape."""
        # Extract ego car's information
        ego_info = vehicleKeyInfo(self.ego)
        x = ego_info['x']
        y = self.ego.position[1]

        print("in visualizeEgoRiskField2")
        print(f"carId: {ego_info['carId']}")
        print(f"x: {ego_info['x']}")
        print(f"y: {y}")
        print(f"shift_y: {ego_info['shift_y']}")

        speed = ego_info['speed']
        heading_angle = ego_info['heading']

        # Call the field_distributionNew function from risk.py
        X_common, Y_common, risk_field = risk.field_distributionNew(
            x=x, y=y, speed=speed, heading_angle=heading_angle,
            turning_angle=0.1, vehicle_length=5, common_grid=None
        )
        print("risk_field2: ")
        print(f"Risk field type: {type(risk_field)}, shape: {getattr(risk_field, 'shape', 'N/A')}")

        # Visualize the risk field as a heat map
        plt.figure(figsize=(1.5, 6))
        plt.imshow(risk_field, cmap='hot', interpolation='nearest', extent=[X_common.min(), X_common.max(), Y_common.min(), Y_common.max()])
        # plt.colorbar(label='Risk Intensity')
        plt.title('Ego Car Risk Field Heat Map')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')

        # Create the result folder if it does not exist
        result_folder = "results"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Save the heat map as a picture
        heatmap_file_path = os.path.join(result_folder, f"risk_field_{frame_name}.png")
        plt.savefig(heatmap_file_path, bbox_inches='tight')
        print(f"Heat map saved to {heatmap_file_path}")


        max_value = np.max(risk_field)
        min_value = np.min(risk_field)
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = {p: np.percentile(risk_field, p) for p in percentiles}


        sample_count = risk_field.size
        print(f"Risk field sample count: {sample_count}")

        print(f"Risk field max value: {max_value}")
        print(f"Risk field min value: {min_value}")
        print("Risk field percentiles:")
        for p, value in percentile_values.items():
            print(f"  {p}%: {value}")


        for p, value in percentile_values.items():
            proportion = np.sum(risk_field <= value) / risk_field.size
            print(f"  Proportion of values <= {p}% ({value}): {proportion:.2%}")
        
        threshold = 0.01
        print(f"Contour threshold: {threshold}")

        threshold = 0.01
        proportion = np.sum(risk_field <= threshold) / risk_field.size
        percentile = (proportion * 100) 


        print(f"Threshold {threshold} is at approximately the {percentile:.2f}% percentile of the data.")


        samples_in_contour = np.sum(risk_field >= threshold)
        total_samples = risk_field.size
        proportion_in_contour = samples_in_contour / total_samples


        print(f"Number of samples within the contour (>= {threshold}): {samples_in_contour}")
        print(f"Proportion of samples within the contour: {proportion_in_contour:.2%}")
        


        # Generate a contour plot for the shape of the risk field
        plt.figure(figsize=(1.5, 6))
        # plt.contour(X_common, Y_common, risk_field, levels=[0.01], colors='blue')  # Draw contours for non-zero values
        #threshold = np.percentile(risk_field[risk_field > 0], 95)  # 20th percentile threshold

        plt.contour(X_common, Y_common, risk_field, levels=[threshold], colors='blue')
        plt.axis('scaled')  
       
        plt.title('Ego Car Risk Field Contour')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')

        # Save the contour plot as a picture
        contour_file_path = os.path.join(result_folder, f"risk_field_contour_{frame_name}.png")
        plt.savefig(contour_file_path, bbox_inches='tight')
        print(f"Contour plot saved to {contour_file_path}")

        plt.show()
    
    # Risk Field & Risk Value merge into prompts
    def describe1(self, decisionFrame: int) -> Tuple[str, float]:
        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)
        self.dbBridge.insertVehicle(decisionFrame, surroundVehicles)


        lst = [vehicleKeyInfo(self.ego)]
        ego_speed = self.ego.speed
        

        for veh in surroundVehicles:
            lst.append(vehicleKeyInfo(veh))
        
        print("all cars info in describe1: ", lst)
        print("=============================")


        # print("=== Ego Car Information Start ===")
        ego_info = lst[0]
        
        # print(f"carId: {ego_info['carId']}")
        # print(f"x: {ego_info['x']}")
        # print(f"shift_y: {ego_info['shift_y']}")
        # print(f"speed: {ego_info['speed']}")
        # print(f"heading: {ego_info['heading']}")
        # print("=== Ego Car Information End ===")

        risk_description = self.calSurroundingValidVehiclesRisk(decisionFrame)

        # self.visualizeEgoRiskField2(decisionFrame)
        #self.visualizeEgoRiskField3(decisionFrame)
        

        # carId，x，shift_y，speed，heading。
        # risk.field_distribution
        #def field_distribution(x=0, y=0, speed=10, heading_angle=0, turning_angle=0.1, vehicle_length=5, common_grid=None):
        

        currentLaneIndex: LaneIndex = self.ego.lane_index
        if self.isInJunction(self.ego):
            roadCondition = "You are driving in an intersection, you can't change lane. "
            roadCondition += f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, speed is {self.ego.speed:.2f} m/s, and acceleration is {self.ego.action['acceleration']:.2f} m/s^2.\n"
            SVDescription = self.describeSVJunctionLane(currentLaneIndex)
        else:
            roadCondition = self.processNormalLane(currentLaneIndex)
            SVDescription = self.describeSVNormalLane(currentLaneIndex)

        # print("roadCondition: ", roadCondition)
        # print("SVDescription: ", SVDescription)

        return roadCondition + SVDescription + risk_description, ego_speed

    
    def promptsCommit(
        self, decisionFrame: int, vectorID: str, done: bool,
        description: str, fewshots: str, thoughtsAndAction: str
    ):
        self.dbBridge.insertPrompts(
            decisionFrame, vectorID, done, description,
            fewshots, thoughtsAndAction
        )

    def visualizeEgoRiskFieldX2(self, frame_name: str, carId1: int, x1: float = 0, y1: float = 0, speed1: float = 10, heading_angle1: float = 0) -> None:
        """Visualize the ego car and a next car's risk field as a heat map and as a contour shape."""

        ego_info = vehicleKeyInfo(self.ego)
        x = ego_info['x']
        y = self.ego.position[1]

        print("in visualizeEgoRiskFieldX2")
        print(f"carId: {ego_info['carId']}")
        print(f"x: {ego_info['x']}")
        print(f"y: {y}")
        print(f"shift_y: {ego_info['shift_y']}")

        # print(f"Updated speed: {ego_info['speed']} m/s")
        speed = ego_info['speed']
        heading_angle = ego_info['heading']

        print("the car near ego: ")
        print(f"carId: {carId1}")
        print(f"x: {x1}")
        print(f"y: {y1}")
        print(f"speed: {speed1}")

        # Call the field_distributionNew function from risk.py, return Ego car's risk field
        X_common, Y_common, risk_field = risk.field_distributionNew(
            x=x, y=y, speed=speed, heading_angle=heading_angle,
            turning_angle=0.1, vehicle_length=5, common_grid=None
        )

        # Call the field_distributionNew function from risk.py, return a car which near ego's risk field
        X_common1, Y_common1, risk_field1 = risk.field_distributionNew(
            x=x1, y=y1, speed=speed1, heading_angle=heading_angle1,
            turning_angle=0.1, vehicle_length=5, common_grid=None
        )

        print("risk_field ego: ")
        print(f"Risk field type: {type(risk_field)}, shape: {getattr(risk_field, 'shape', 'N/A')}")

        print("risk_field of the car near ego: ")
        print(f"Risk field type: {type(risk_field1)}, shape: {getattr(risk_field1, 'shape', 'N/A')}")

                        
        threshold = 0.01
        print(f"Contour threshold: {threshold}")

        
        # Generate a contour plot for the shape of the risk field
        plt.figure(figsize=(1.5, 6))
        
        plt.contour(X_common, Y_common, risk_field, levels=[threshold], colors='green')
        plt.contour(X_common1, Y_common1, risk_field1, levels=[threshold], colors='blue')

        plt.axis('scaled')  
  
        plt.title('Ego Car & a near car Risk Field Contour')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        result_folder = "results"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)


        # Save the contour plot as a picture
        contour_file_path = os.path.join(result_folder, f"risk_field_contourX2_{frame_name}.png")
        plt.savefig(contour_file_path, bbox_inches='tight')
        print(f"Contour plot saved to {contour_file_path}")

        plt.show()

    def getFrontV(self) -> Optional[Dict[str, Union[int, float]]]:

        currentLaneIndex = self.ego.lane_index

        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)

        closest_vehicle = None
        closest_distance = float('inf')

        for sv in surroundVehicles:
            if sv.lane_index == currentLaneIndex:  
                distance = self.getVehDis(sv)  
                relative_position = sv.position - self.ego.position
                if relative_position[0] > 0 and distance < closest_distance:  
                    closest_vehicle = sv
                    closest_distance = distance


        if closest_vehicle:
            return {
                'carId': id(closest_vehicle) % 1000,
                'x': closest_vehicle.position[0],
                'shift_y': closest_vehicle.position[1],
                'speed': closest_vehicle.speed,
                'heading': closest_vehicle.heading,
                'laneId': closest_vehicle.lane_index[2],
                'dis': closest_distance
            }
        else:

            return None

    def getBehindV(self) -> Optional[Dict[str, Union[int, float]]]:

        currentLaneIndex = self.ego.lane_index

        ego_x = self.ego.position[0]

        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)


        closest_vehicle = None
        closest_distance = float('inf')


        for sv in surroundVehicles:
            if sv.lane_index == currentLaneIndex:  
                sv_x = sv.position[0]  
                if sv_x < ego_x:  
                    distance = abs(ego_x - sv_x)  
                    if distance < closest_distance:  
                        closest_vehicle = sv
                        closest_distance = distance


        if closest_vehicle:
            return {
                'carId': id(closest_vehicle) % 1000,
                'x': closest_vehicle.position[0],
                'shift_y': closest_vehicle.position[1],
                'speed': closest_vehicle.speed,
                'heading': closest_vehicle.heading,
                'laneId': closest_vehicle.lane_index[2],
                'dis': closest_distance
            }
        else:

            return None

    def getLeftFrontV(self) -> Optional[Dict[str, Union[int, float]]]:


        currentLaneIndex = self.ego.lane_index

        ego_x = self.ego.position[0]

        leftLaneId = currentLaneIndex[2] - 1


        if leftLaneId < 0:
            # print("No left lane exists.")
            return None


        leftLaneIndex = (currentLaneIndex[0], currentLaneIndex[1], leftLaneId)


        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)


        closest_vehicle = None
        closest_distance = float('inf')


        for sv in surroundVehicles:
            if sv.lane_index == leftLaneIndex:  
                sv_x = sv.position[0]  
                if sv_x > ego_x:  
                    distance = abs(ego_x - sv_x)  
                    if distance < closest_distance:  
                        closest_vehicle = sv
                        closest_distance = distance


        if closest_vehicle:
            return {
                'carId': id(closest_vehicle) % 1000,
                'x': closest_vehicle.position[0],
                'shift_y': closest_vehicle.position[1],
                'speed': closest_vehicle.speed,
                'heading': closest_vehicle.heading,
                'laneId': closest_vehicle.lane_index[2],
                'dis': closest_distance
            }
        else:
            return None

    def getRightFrontV(self) -> Optional[Dict[str, Union[int, float]]]:

        currentLaneIndex = self.ego.lane_index

        ego_x = self.ego.position[0]

        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        numLanes = len(sideLanes)


        rightLaneId = currentLaneIndex[2] + 1


        if rightLaneId >= numLanes:
            # print("No right lane exists.")
            return None


        rightLaneIndex = (currentLaneIndex[0], currentLaneIndex[1], rightLaneId)


        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)


        closest_vehicle = None
        closest_distance = float('inf')


        for sv in surroundVehicles:
            if sv.lane_index == rightLaneIndex:  
                sv_x = sv.position[0]
                if sv_x > ego_x:  
                    distance = abs(ego_x - sv_x) 
                    if distance < closest_distance:  
                        closest_vehicle = sv
                        closest_distance = distance


        if closest_vehicle:
            return {
                'carId': id(closest_vehicle) % 1000,
                'x': closest_vehicle.position[0],
                'shift_y': closest_vehicle.position[1],
                'speed': closest_vehicle.speed,
                'heading': closest_vehicle.heading,
                'laneId': closest_vehicle.lane_index[2],
                'dis': closest_distance
            }
        else:
            return None

    def getRightBehindV(self) -> Optional[Dict[str, Union[int, float]]]:

        currentLaneIndex = self.ego.lane_index
 
        ego_x = self.ego.position[0]

        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        numLanes = len(sideLanes)


        rightLaneId = currentLaneIndex[2] + 1


        if rightLaneId >= numLanes:
            # print("No right lane exists.")
            return None


        rightLaneIndex = (currentLaneIndex[0], currentLaneIndex[1], rightLaneId)


        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)


        closest_vehicle = None
        closest_distance = float('inf')


        for sv in surroundVehicles:
            if sv.lane_index == rightLaneIndex: 
                sv_x = sv.position[0]  
                if sv_x <= ego_x:  
                    distance = abs(ego_x - sv_x)  
                    if distance < closest_distance:  
                        closest_vehicle = sv
                        closest_distance = distance


        if closest_vehicle:
            return {
                'carId': id(closest_vehicle) % 1000,
                'x': closest_vehicle.position[0],
                'shift_y': closest_vehicle.position[1],
                'speed': closest_vehicle.speed,
                'heading': closest_vehicle.heading,
                'laneId': closest_vehicle.lane_index[2],
                'dis': closest_distance
            }
        else:
            return None

    def getLeftBehindV(self) -> Optional[Dict[str, Union[int, float]]]:

        currentLaneIndex = self.ego.lane_index

        ego_x = self.ego.position[0]


        leftLaneId = currentLaneIndex[2] - 1


        if leftLaneId < 0:
            # print("No left lane exists.")
            return None


        leftLaneIndex = (currentLaneIndex[0], currentLaneIndex[1], leftLaneId)


        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)


        closest_vehicle = None
        closest_distance = float('inf')


        for sv in surroundVehicles:
            if sv.lane_index == leftLaneIndex:
                sv_x = sv.position[0]  
                if sv_x <= ego_x:  
                    distance = abs(ego_x - sv_x) 
                    if distance < closest_distance:  
                        closest_vehicle = sv
                        closest_distance = distance


        if closest_vehicle:
            return {
                'carId': id(closest_vehicle) % 1000,
                'x': closest_vehicle.position[0],
                'shift_y': closest_vehicle.position[1],
                'speed': closest_vehicle.speed,
                'heading': closest_vehicle.heading,
                'laneId': closest_vehicle.lane_index[2],
                'dis': closest_distance
            }
        else:
            return None


    def printFrontVehicleInfo(self) -> None:

        front_vehicle = self.getFrontV()
        
        if front_vehicle:
            print("Front vehicle information:")
            print(f"  Car ID: {front_vehicle['carId']}")
            print(f"  X Position: {front_vehicle['x']:.2f}")
            print(f"  Y Position (shift_y): {front_vehicle['shift_y']:.2f}")
            print(f"  Speed: {front_vehicle['speed']:.2f} m/s")
            print(f"  Heading: {front_vehicle['heading']:.2f} radians")
            print(f"  Lane ID: {front_vehicle['laneId']}")
            print(f"  Distance to Ego: {front_vehicle['dis']:.2f} meters")
        else:
            print("No front vehicle found in the same lane.")
        
    def printSurroundingValidVehiclesInfo(self) -> None:

        front_vehicle = self.getFrontV()
        behind_vehicle = self.getBehindV()
        left_front_vehicle = self.getLeftFrontV()
        left_behind_vehicle = self.getLeftBehindV()
        right_front_vehicle = self.getRightFrontV()
        right_behind_vehicle = self.getRightBehindV()


        def print_vehicle_info(position: str, vehicle: Optional[Dict[str, Union[int, float]]]) -> None:
            if vehicle:
                print(f"{position} vehicle information:")
                print(f"  Car ID: {vehicle['carId']}")
                print(f"  X Position: {vehicle['x']:.2f}")
                print(f"  Y Position (shift_y): {vehicle['shift_y']:.2f}")
                print(f"  Speed: {vehicle['speed']:.2f} m/s")
                print(f"  Heading: {vehicle['heading']:.2f} radians")
                print(f"  Lane ID: {vehicle['laneId']}")
                print(f"  Distance to Ego: {vehicle['dis']:.2f} meters")
            else:
                print(f"No {position.lower()} vehicle found.")


        print_vehicle_info("Front", front_vehicle)
        print_vehicle_info("Behind", behind_vehicle)
        print_vehicle_info("Left Front", left_front_vehicle)
        print_vehicle_info("Left Behind", left_behind_vehicle)
        print_vehicle_info("Right Front", right_front_vehicle)
        print_vehicle_info("Right Behind", right_behind_vehicle)


    def calSurroundingValidVehiclesRisk(self, decisionFrame: int) -> str:

        front_vehicle = self.getFrontV()
        behind_vehicle = self.getBehindV()
        left_front_vehicle = self.getLeftFrontV()
        left_behind_vehicle = self.getLeftBehindV()
        right_front_vehicle = self.getRightFrontV()
        right_behind_vehicle = self.getRightBehindV()


        lst = [vehicleKeyInfo(self.ego)]
        ego_info = lst[0]


        def calculate_risk(position: str, vehicle: Optional[Dict[str, Union[int, float]]]) -> Optional[Tuple[float, float, float]]:
            if vehicle:
                # print(f"Calculating risk between Ego and {position} vehicle:")
                # print(f"  Ego: {ego_info}")
                # print(f"  {position} Vehicle: {vehicle}")


                if position == "Front":
                    ego_area, other_area, lap_area = riskValueCal(
                        decisionFrame,
                        ego_info['carId'],
                        ego_info['x'],
                        ego_info['shift_y'],
                        ego_info['speed'],
                        ego_info['heading'],
                        vehicle['carId'],
                        vehicle['x'],
                        vehicle['shift_y'],
                        vehicle['speed'],
                        vehicle['heading'],
                        position
                    )
                elif position == "Behind":
                    ego_area, other_area, lap_area = riskValueCal(
                        decisionFrame,
                        ego_info['carId'],
                        ego_info['x'],
                        ego_info['shift_y'],
                        ego_info['speed'],
                        ego_info['heading'],
                        vehicle['carId'],
                        vehicle['x'],
                        vehicle['shift_y'],
                        vehicle['speed'],
                        vehicle['heading'],
                        position
                    )
                elif position == "Left Front":
                    ego_area, other_area, lap_area = riskValueCal(
                        decisionFrame,
                        ego_info['carId'],
                        ego_info['x'],
                        (ego_info['shift_y']-4.0), 
                        ego_info['speed'],
                        ego_info['heading'],
                        vehicle['carId'],
                        vehicle['x'],
                        vehicle['shift_y'],
                        vehicle['speed'],
                        vehicle['heading'],
                        position
                    )
                elif position == "Left Behind":
                    ego_area, other_area, lap_area = riskValueCal(
                        decisionFrame,
                        ego_info['carId'],
                        ego_info['x'],
                        (ego_info['shift_y']-4.0), 
                        ego_info['speed'],
                        ego_info['heading'],
                        vehicle['carId'],
                        vehicle['x'],
                        vehicle['shift_y'],
                        vehicle['speed'],
                        vehicle['heading'],
                        position
                    )
                elif position == "Right Front":
                    ego_area, other_area, lap_area = riskValueCal(
                        decisionFrame,
                        ego_info['carId'],
                        ego_info['x'],
                        (ego_info['shift_y']+4.0), 
                        ego_info['speed'],
                        ego_info['heading'],
                        vehicle['carId'],
                        vehicle['x'],
                        vehicle['shift_y'],
                        vehicle['speed'],
                        vehicle['heading'],
                        position
                    )
                elif position == "Right Behind":
                    ego_area, other_area, lap_area = riskValueCal(
                        decisionFrame,
                        ego_info['carId'],
                        ego_info['x'],
                        (ego_info['shift_y']+4.0), 
                        ego_info['speed'],
                        ego_info['heading'],
                        vehicle['carId'],
                        vehicle['x'],
                        vehicle['shift_y'],
                        vehicle['speed'],
                        vehicle['heading'],
                        position
                    )
                else:
                    print(f"Unknown position: {position}. Skipping risk calculation.")
                    return None, None, None


                return ego_area, other_area, lap_area
            else:
                #print(f"No {position.lower()} vehicle found. Skipping risk calculation.")
                return None, None, None



        risks = {
            "Front": calculate_risk("Front", front_vehicle),
            "Behind": calculate_risk("Behind", behind_vehicle),
            "Left Front": calculate_risk("Left Front", left_front_vehicle),
            "Left Behind": calculate_risk("Left Behind", left_behind_vehicle),
            "Right Front": calculate_risk("Right Front", right_front_vehicle),
            "Right Behind": calculate_risk("Right Behind", right_behind_vehicle),
        }


        front_risk = 0
        behind_risk = 0
        left_front_risk = 0
        left_behind_risk = 0
        right_front_risk = 0
        right_behind_risk = 0


        for position, risk in risks.items():

            if risk and all(value is not None for value in risk):
                ego_area, other_area, lap_area = risk
                if position == "Front":
                    # print("Calculating risk for Front vehicle...")
                    if other_area > 0:
                        front_risk = lap_area / other_area
                    else:
                        front_risk = 1
                        print("has vehicle in front, but its risk field abnormal. Setting risk to 1.")
                elif position == "Behind":
                    # print("Calculating risk for Behind vehicle...")
                    if ego_area > 0:
                        behind_risk = lap_area / ego_area
                    else:
                        behind_risk = 1
                        print("has vehicle in behind, but ego risk field abnormal. Setting risk to 1.") 
                elif position == "Left Front":
                    # print("Calculating risk for Left Front vehicle...")
                    if other_area > 0:
                        left_front_risk = lap_area / other_area
                    else:
                        left_front_risk = 1
                        print("has vehicle in left front, but its risk field abnormal. Setting risk to 1.")
                elif position == "Left Behind":
                    # print("Calculating risk for Left Behind vehicle...")
                    if ego_area > 0:
                        left_behind_risk = lap_area / ego_area
                    else:
                        left_behind_risk = 1
                        print("has vehicle in left behind, but ego risk field abnormal. Setting risk to 1.")    
                elif position == "Right Front":
                    # print("Calculating risk for Right Front vehicle...")
                    if other_area > 0:
                        right_front_risk = lap_area / other_area
                    else:
                        right_front_risk = 1
                        print("has vehicle in right front, but its risk field abnormal. Setting risk to 1.")
                elif position == "Right Behind":
                    # print("Calculating risk for Right Behind vehicle...")
                    if ego_area > 0:
                        right_behind_risk = lap_area / ego_area
                    else:
                        right_behind_risk = 1
                        print("has vehicle in right behind, but ego risk field abnormal. Setting risk to 1.")
                else:
                    print(f"Unknown position: {position}. Skipping risk calculation.")
                    return None, None, None  


            else:
                # print(f"No {position.lower()} vehicle found.")
                just_do_sth = 1

        left_risk = left_front_risk + left_behind_risk
        if left_risk > 1:
            left_risk = 1.0
        right_risk = right_front_risk + right_behind_risk
        if right_risk > 1:
            right_risk = 1.0


        if self.ego.lane_index[2] == 0: 
            left_risk = 1
        if self.ego.lane_index[2] == len(self.network.all_side_lanes(self.ego.lane_index)) - 1:  
            right_risk = 1
        

        summary = "\nRisk Value Summary:\n"

        if front_vehicle is not None:
            front_distance = abs(front_vehicle['x'] - ego_info['x'])
            front_relative_speed = front_vehicle['speed'] - ego_info['speed']
            summary += (
                f"Front:\n"
                f"  Vehicle Present: Yes\n"
                f"  Distance: {front_distance:.2f} meters\n"
                f"  Relative Speed to ego: {front_relative_speed:.2f} m/s\n"
                f"  Risk Field Value (Other Area): {risks['Front'][1]:.2f}\n"
                f"  Risk Value: {front_risk:.2f}\n"
            )
        else:
            summary += (
                f"Front:\n"
                f"  Vehicle Present: No\n"
                f"  Risk Value: {front_risk:.2f}\n"
            )


        if behind_vehicle is not None:
            behind_distance = abs(behind_vehicle['x'] - ego_info['x'])
            behind_relative_speed = behind_vehicle['speed'] - ego_info['speed']
            summary += (
                f"Behind:\n"
                f"  Vehicle Present: Yes\n"
                f"  Distance: {behind_distance:.2f} meters\n"
                f"  Relative Speed to ego: {behind_relative_speed:.2f} m/s\n"
                f"  Risk Field Value (Ego Area): {risks['Behind'][0]:.2f}\n"
                f"  Risk Value: {behind_risk:.2f}\n"
            )
        else:
            summary += (
                f"Behind:\n"
                f"  Vehicle Present: No\n"
                f"  Risk Value: {behind_risk:.2f}\n"
            )

        summary += "Left:\n"


        if self.ego.lane_index[2] == 0: 
            summary += (
                f"  You are in the leftmost lane.\n"
                f"  Overall Left Risk Value: {left_risk:.2f}\n"
            )
        else:
            if left_front_vehicle is not None:
                left_front_distance = abs(left_front_vehicle['x'] - ego_info['x'])
                left_front_relative_speed = left_front_vehicle['speed'] - ego_info['speed']
                summary += (
                    f"  Left Front:\n"
                    f"    Vehicle Present: Yes\n"
                    f"    Distance: {left_front_distance:.2f} meters\n"
                    f"    Relative Speed to ego: {left_front_relative_speed:.2f} m/s\n"
                    f"    Risk Field Value (Other Area): {risks['Left Front'][1]:.2f}\n"
                )
            else:
                summary += "  Left Front:\n    Vehicle Present: No\n"

            if left_behind_vehicle is not None:
                left_behind_distance = abs(left_behind_vehicle['x'] - ego_info['x'])
                left_behind_relative_speed = left_behind_vehicle['speed'] - ego_info['speed']
                summary += (
                    f"  Left Behind:\n"
                    f"    Vehicle Present: Yes\n"
                    f"    Distance: {left_behind_distance:.2f} meters\n"
                    f"    Relative Speed to ego: {left_behind_relative_speed:.2f} m/s\n"
                    f"    Risk Field Value (Ego Area): {risks['Left Behind'][0]:.2f}\n"
                )
            else:
                summary += "  Left Behind:\n    Vehicle Present: No\n"

            summary += f"  Overall Left Risk Value: {left_risk:.2f}\n"


        summary += "Right:\n"


        if self.ego.lane_index[2] == len(self.network.all_side_lanes(self.ego.lane_index)) - 1:  
            summary += (
                f"  You are in the rightmost lane.\n"
                f"  Overall Right Risk Value: {right_risk:.2f}\n"
            )
        else:
            if right_front_vehicle is not None:
                right_front_distance = abs(right_front_vehicle['x'] - ego_info['x'])
                right_front_relative_speed = right_front_vehicle['speed'] - ego_info['speed']
                summary += (
                    f"  Right Front:\n"
                    f"    Vehicle Present: Yes\n"
                    f"    Distance: {right_front_distance:.2f} meters\n"
                    f"    Relative Speed to ego: {right_front_relative_speed:.2f} m/s\n"
                    f"    Risk Field Value (Other Area): {risks['Right Front'][1]:.2f}\n"
                )
            else:
                summary += "  Right Front:\n    Vehicle Present: No\n"

            if right_behind_vehicle is not None:
                right_behind_distance = abs(right_behind_vehicle['x'] - ego_info['x'])
                right_behind_relative_speed = right_behind_vehicle['speed'] - ego_info['speed']
                summary += (
                    f"  Right Behind:\n"
                    f"    Vehicle Present: Yes\n"
                    f"    Distance: {right_behind_distance:.2f} meters\n"
                    f"    Relative Speed to ego: {right_behind_relative_speed:.2f} m/s\n"
                    f"    Risk Field Value (Ego Area): {risks['Right Behind'][0]:.2f}\n"
                )
            else:
                summary += "  Right Behind:\n    Vehicle Present: No\n"

            summary += f"  Overall Right Risk Value: {right_risk:.2f}\n"


        print(summary)


        prompt = "\nThe driving Risk Values around you are provided as below:\n"
        prompt += f"(Overall Left Risk Value, {left_risk:.2f}), (Overall Right Risk Value, {right_risk:.2f}), (Front Risk Value, {front_risk:.2f}), (Behind Risk Value, {behind_risk:.2f}). "
        prompt += "The Risk Value is a float number from 0.00 to 1.00. The Value 0.00 means no risk, and 1.00 means the highest risk. "
        prompt += "You should consider your action and do reasoning, based on the scenario descriptions together with the Left, Right, Front, Behind Risk Values provided to you."

        return prompt
    
    def getSixOritationVehicleInfo(self) -> List[Dict[str,any]]:

        currentLaneIndex = self.ego.lane_index

        leftLaneId = currentLaneIndex[2] - 1

        leftLaneIndex = (currentLaneIndex[0], currentLaneIndex[1], leftLaneId)

        rightLaneId = currentLaneIndex[2] + 1

        rightLaneIndex = (currentLaneIndex[0], currentLaneIndex[1], rightLaneId)

        ego_x = self.ego.position[0]

        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)

        front_vehicle = None
        front_distance = float('inf')
        behind_vehicle = None
        behind_distance = float('inf')
        left_front_vehicle = None
        left_front_distance = float('inf')
        left_behide_vehicle = None
        left_behide_distance = float('inf')
        right_front_vehicle = None
        right_front_distance = float('inf')
        right_behide_vehicle = None
        right_behide_distance = float('inf')


        for sv in surroundVehicles:
            isAhead = self.isAheadOfEgo(sv)
            sv_x = sv.position[0] 
            x_dis = abs(ego_x - sv_x)
            if sv.lane_index == currentLaneIndex: 
                if isAhead and x_dis < front_distance:  
                    front_vehicle = sv
                    front_distance = x_dis
                elif not isAhead and x_dis < behind_distance:
                    behind_vehicle = sv
                    behind_distance = x_dis
            elif sv.lane_index == leftLaneIndex:  
                if isAhead and x_dis < left_front_distance:  
                    left_front_vehicle = sv
                    left_front_distance = x_dis
                elif not isAhead and x_dis < left_behide_distance:  
                    left_behide_vehicle = sv
                    left_behide_distance = x_dis
            elif sv.lane_index == rightLaneIndex:
                if isAhead and x_dis < right_front_distance:  
                    right_front_vehicle = sv
                    right_front_distance = x_dis
                elif not isAhead and x_dis < right_behide_distance: 
                    right_behide_vehicle = sv
                    right_behide_distance = x_dis
        lst = []

        lst.append(vehicleInfo(front_vehicle,front_distance))
        lst.append(vehicleInfo(behind_vehicle,behind_distance))
        lst.append(vehicleInfo(left_front_vehicle,left_front_distance))
        lst.append(vehicleInfo(left_behide_vehicle,left_behide_distance))
        lst.append(vehicleInfo(right_front_vehicle,right_front_distance))
        lst.append(vehicleInfo(right_behide_vehicle,right_behide_distance))
        return lst
    
    def plot_car_distribution_contour(self, car_distribution, threshold):

        rows, cols = car_distribution.shape


        x = np.arange(cols)  
        y = np.arange(rows)  
        X, Y = np.meshgrid(x, y)


        plt.figure(figsize=(8, 6))
        contour = plt.contour(X, Y, car_distribution, levels=[threshold], colors='blue')
        plt.clabel(contour, inline=True, fontsize=8, fmt=f'Threshold: {threshold}')
        plt.title('Ego Distribution Contour')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.gca().invert_yaxis()  
        plt.grid(True)

        result_folder = "results"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)



        contour_file_path = os.path.join(result_folder, f"lyj-101-1001.png")
        plt.savefig(contour_file_path, bbox_inches='tight')
        print(f"Contour plot saved to {contour_file_path}")


        plt.draw()
        plt.pause(0.001)
        plt.show()

    # based on previous describe1: Risk Field & Risk Value merge into prompts
    # add Memory about risk pattern
    def describe3(self, decisionFrame: int) -> Tuple[str, float]:
        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM) # 数字后面要增加到15或者20
        self.dbBridge.insertVehicle(decisionFrame, surroundVehicles)


        lst = [vehicleKeyInfoMore(self.ego)]
        ego_speed = self.ego.speed 
        

        for veh in surroundVehicles:
            lst.append(vehicleKeyInfoMore(veh))


        road_speed_level = self.calculate_road_speed_level(surroundVehicles, self.ego)

        printCarInfo(lst)

        #print("all cars info in describe3: ", lst)
        #print("=============================")


        # print("=== Ego Car Information Start ===")
        ego_info = lst[0]
        
        # print(f"carId: {ego_info['carId']}")
        # print(f"x: {ego_info['x']}")
        # print(f"shift_y: {ego_info['shift_y']}")
        # print(f"speed: {ego_info['speed']}")
        # print(f"heading: {ego_info['heading']}")
        # print("=== Ego Car Information End ===")

        currentLaneIndex: LaneIndex = self.ego.lane_index
        if self.isInJunction(self.ego):
            roadCondition = "You are driving in an intersection, you can't change lane. "
            roadCondition += f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, speed is {self.ego.speed:.2f} m/s, and acceleration is {self.ego.action['acceleration']:.2f} m/s^2.\n"
            SVDescription = self.describeSVJunctionLane(currentLaneIndex)
        else:
            roadCondition = self.processNormalLane(currentLaneIndex)
            SVDescription = self.describeSVNormalLane(currentLaneIndex)


        risk_description, risk_vec4, risk_pattern_vec, return_crash_happened = getSurroundingRiskPattern(self, decisionFrame)

        return roadCondition + SVDescription + risk_description, ego_speed, risk_vec4, risk_pattern_vec, road_speed_level, return_crash_happened
    
    def getLeftLeftFrontV(self) -> Optional[Dict[str, Union[int, float]]]:

        currentLaneIndex = self.ego.lane_index

        ego_x = self.ego.position[0]

        leftLeftLaneId = currentLaneIndex[2] - 2


        if leftLeftLaneId < 0:
            # print("No left-left lane exists.")
            return None


        leftLeftLaneIndex = (currentLaneIndex[0], currentLaneIndex[1], leftLeftLaneId)


        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)


        closest_vehicle = None
        closest_distance = float('inf')


        for sv in surroundVehicles:
            if sv.lane_index == leftLeftLaneIndex:  
                sv_x = sv.position[0]  
                if sv_x > ego_x:  
                    distance = abs(ego_x - sv_x)  
                    if distance < closest_distance:  
                        closest_vehicle = sv
                        closest_distance = distance


        if closest_vehicle:
            return {
                'carId': id(closest_vehicle) % 1000,
                'x': closest_vehicle.position[0],
                'shift_y': closest_vehicle.position[1],
                'speed': closest_vehicle.speed,
                'heading': closest_vehicle.heading,
                'laneId': closest_vehicle.lane_index[2],
                'dis': closest_distance
            }
        else:

            return None
        
    def getRightRightFrontV(self) -> Optional[Dict[str, Union[int, float]]]:

        currentLaneIndex = self.ego.lane_index
     
        ego_x = self.ego.position[0]
   
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        numLanes = len(sideLanes)

 
        rightRightLaneId = currentLaneIndex[2] + 2

  
        if rightRightLaneId >= numLanes:
            # print("No right-right lane exists.")
            return None


        rightRightLaneIndex = (currentLaneIndex[0], currentLaneIndex[1], rightRightLaneId)


        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)


        closest_vehicle = None
        closest_distance = float('inf')

    
        for sv in surroundVehicles:
            if sv.lane_index == rightRightLaneIndex:  
                sv_x = sv.position[0]  
                if sv_x > ego_x:  
                    distance = abs(ego_x - sv_x)  
                    if distance < closest_distance:  
                        closest_vehicle = sv
                        closest_distance = distance


        if closest_vehicle:
            return {
                'carId': id(closest_vehicle) % 1000,
                'x': closest_vehicle.position[0],
                'shift_y': closest_vehicle.position[1],
                'speed': closest_vehicle.speed,
                'heading': closest_vehicle.heading,
                'laneId': closest_vehicle.lane_index[2],
                'dis': closest_distance
            }
        else:
            return None
        
    def getRightRightBehindV(self) -> Optional[Dict[str, Union[int, float]]]:

        currentLaneIndex = self.ego.lane_index

        ego_x = self.ego.position[0]

        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        numLanes = len(sideLanes)


        rightRightLaneId = currentLaneIndex[2] + 2


        if rightRightLaneId >= numLanes:
            # print("No right-right lane exists.")
            return None


        rightRightLaneIndex = (currentLaneIndex[0], currentLaneIndex[1], rightRightLaneId)


        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)


        closest_vehicle = None
        closest_distance = float('inf')


        for sv in surroundVehicles:
            if sv.lane_index == rightRightLaneIndex: 
                sv_x = sv.position[0]  
                if sv_x <= ego_x: 
                    distance = abs(ego_x - sv_x) 
                    if distance < closest_distance:  
                        closest_vehicle = sv
                        closest_distance = distance


        if closest_vehicle:
            return {
                'carId': id(closest_vehicle) % 1000,
                'x': closest_vehicle.position[0],
                'shift_y': closest_vehicle.position[1],
                'speed': closest_vehicle.speed,
                'heading': closest_vehicle.heading,
                'laneId': closest_vehicle.lane_index[2],
                'dis': closest_distance
            }
        else:

            return None
        
    def getLeftLeftBehindV(self) -> Optional[Dict[str, Union[int, float]]]:

        currentLaneIndex = self.ego.lane_index
   
        ego_x = self.ego.position[0]

        leftLeftLaneId = currentLaneIndex[2] - 2

  
        if leftLeftLaneId < 0:
            # print("No left-left lane exists.")
            return None


        leftLeftLaneIndex = (currentLaneIndex[0], currentLaneIndex[1], leftLeftLaneId)


        surroundVehicles = self.getSurroundVehicles(LOOK_SURROUND_V_NUM)


        closest_vehicle = None
        closest_distance = float('inf')

 
        for sv in surroundVehicles:
            if sv.lane_index == leftLeftLaneIndex:  
                sv_x = sv.position[0]  
                if sv_x <= ego_x:  
                    distance = abs(ego_x - sv_x)  
                    if distance < closest_distance:  
                        closest_vehicle = sv
                        closest_distance = distance


        if closest_vehicle:
            return {
                'carId': id(closest_vehicle) % 1000,
                'x': closest_vehicle.position[0],
                'shift_y': closest_vehicle.position[1],
                'speed': closest_vehicle.speed,
                'heading': closest_vehicle.heading,
                'laneId': closest_vehicle.lane_index[2],
                'dis': closest_distance
            }
        else:

            return None
        
    def calculate_road_speed_level(self, vehicles: List[IDMVehicle], ego_vehicle: MDPVehicle) -> float:

        speeds = [vehicle.speed for vehicle in vehicles if id(vehicle) != id(ego_vehicle)]


        if len(speeds) <= 2:
            print("not more than 2 vehicles:", speeds)
            return sum(speeds) / len(speeds) if speeds else 0.0


        speeds.remove(max(speeds))
        speeds.remove(min(speeds))


        average_speed = sum(speeds) / len(speeds)
        # print(f"Average speed after removing max and min: {average_speed:.2f} m/s")
        return average_speed