
import matplotlib
from matplotlib import pyplot as plt 
from matplotlib.patches import Polygon  
from matplotlib.text import Text  


from highway_env.vehicle.controller import MDPVehicle  
from highway_env.vehicle.behavior import IDMVehicle 
from highway_env.road.road import RoadNetwork  
from highway_env.road.lane import (  
    StraightLane, CircularLane, SineLane, PolyLane, PolyLaneFixedWidth
)
from highway_env.utils import Vector
from typing import List, Union  
import numpy as np  

from rich import print
from typing import List, Tuple, Optional, Union, Dict


class ScePlotter:

    
    def generateArc(
        self, center: Vector, radius: float,
        start_radian: float, end_radian: float,
        direction: bool
    ):

        if direction == 1:  
            start_radian, end_radian = end_radian, start_radian
     
        theta = np.linspace(start_radian, end_radian, num=50)
      
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        return x, y

    def plotNetwork(self, network: RoadNetwork, ax: plt.Axes):

        roadLanes = network.lanes_list()  
        for lane in roadLanes:
            if isinstance(lane, StraightLane):  
                ax.plot(
                    [lane.start[0], lane.end[0]],  
                    [lane.start[1], lane.end[1]],  
                    linewidth=lane.width * 2.8,  
                    color='#c8d6e5',  
                    alpha=0.3  
                )
            elif isinstance(lane, CircularLane):  
                x, y = self.generateArc(
                    lane.center, lane.radius,
                    lane.start_phase, lane.end_phase,
                    lane.direction
                )
                ax.plot(x, y, 
                       linewidth=lane.width * 2.8, 
                       color='#c8d6e5', 
                       alpha=0.3)
            elif isinstance(lane, SineLane):  
                raise NotImplementedError('SineLane is not supported currently.')
            elif isinstance(lane, PolyLane):  
                raise NotImplementedError('PolyLane is not supported currently.')
            elif isinstance(lane, PolyLaneFixedWidth):  
                raise NotImplementedError('PolyLaneFixedWidth is not supported currently.')
            else:
                raise TypeError('Unknown lane type')  

    def getShape(self, vehicle: Union[IDMVehicle, MDPVehicle]):

        radian = np.pi - vehicle.heading  
        
        rotation_matrix = np.array(
            [
                [np.cos(radian), -np.sin(radian)],
                [np.sin(radian), np.cos(radian)]
            ]
        )

        half_length = vehicle.LENGTH / 2
        half_width = vehicle.WIDTH / 2
  
        vertices = np.array(
            [
                [half_length, half_width], [half_length, -half_width],
                [-half_length, -half_width], [-half_length, half_width]
            ]
        )
       
        rotated_vertices = np.dot(vertices, rotation_matrix)
      
        translated_vertices = rotated_vertices + vehicle.position
        return translated_vertices.tolist()  

    def plotSce(
        self, network: RoadNetwork,
        SVs: List[IDMVehicle], ego: MDPVehicle,
        fileName: str
    ):

        fig, ax = plt.subplots()  
        
 
        self.plotNetwork(network, ax)
        
    
        egoVertices = self.getShape(ego) 
        egoRectangle = Polygon(egoVertices, facecolor='#ff9f43')  
        ax.add_patch(egoRectangle)  
     
        egoText = Text(ego.position[0], ego.position[1], 'ego')
        ax.add_artist(egoText)
        

        for sv in SVs:
            svVertices = self.getShape(sv)
            svRectangle = Polygon(svVertices, facecolor='#1dd1a1')  
            ax.add_patch(svRectangle)
      
            svText = Text(sv.position[0], sv.position[1], f'{id(sv)%1000}')
            ax.add_artist(svText)
        
 
        ax.set_xlim(ego.position[0]-50, ego.position[0]+150)
        ax.set_ylim(ego.position[1]-20, ego.position[1]+20)
        ax.invert_yaxis()  
        ax.set_aspect('equal', adjustable='box')  
        
        plt.savefig(fileName, bbox_inches='tight', dpi=360)
        plt.close('all')  

    def plotSce2(
        self,
        network: RoadNetwork,
        SVs: List[IDMVehicle], 
        ego: MDPVehicle,  
        fileName: str,  
        isFinal: bool = False  
    ) -> Tuple[str, Optional[IDMVehicle]]:

        matplotlib.use('Agg')  

        fig, ax = plt.subplots()  

        return_str = "not final, not using"
        return_vehicle = None
        
        if isFinal:
            # print("[yellow]Surrounded Vehicle States:[/yellow]")
            return_str = "After COLLISION, the ego and Surrounded Vehicle States as below:\n"

            return_str += "\n" + f"{'ID':<10}{'Position':<15}{'Speed (m/s)':<15}{'Acceleration (m/s²)':<20}{'Heading (radians)':<20}"
            # print(f"{'ID':<10}{'Position':<15}{'Speed (m/s)':<15}{'Acceleration (m/s²)':<20}{'Heading (radians)':<20}")
       
            return_str += f"\n{'Ego':<10}{str([round(ego.position[0], 2), round(ego.position[1], 2)]):<15}{ego.speed:<15.2f}{ego.action.get('acceleration', 0):<20.2f}{ego.heading:<20.2f}"
            # print(f"{'Ego':<10}{str([round(ego.position[0], 2), round(ego.position[1], 2)]):<15}{ego.speed:<15.2f}{ego.action.get('acceleration', 0):<20.2f}{ego.heading:<20.2f}")

            closest_vehicle1 = None
            closest_vehicle2 = None
            min_distance1 = float('inf') 
            min_distance2 = float('inf')
            for i, sv in enumerate(SVs, start=1):
            
                distance1 = np.linalg.norm(np.array(sv.position) - np.array(ego.position))  
                distance2 = abs(sv.position[0] - ego.position[0])

                if distance1 < min_distance1:
                    min_distance1 = distance1
                    closest_vehicle1 = sv

                if distance2 < min_distance2:
                    min_distance2 = distance2
                    closest_vehicle2 = sv
 
                if closest_vehicle1.lane_index == ego.lane_index:  
           
                    return_vehicle = closest_vehicle1                 
                else:
                  
                    if closest_vehicle2 is not None and distance2 < 5.5:    
                        return_vehicle = closest_vehicle2
                    else:
                        return_vehicle = closest_vehicle1  

                return_str += f"\n{str(id(sv)%1000):<10}{str([round(sv.position[0], 2), round(sv.position[1], 2)]):<15}{sv.speed:<15.2f}{sv.action.get('acceleration', 0):<20.2f}{sv.heading:<20.2f}"
                # print(f"{id(sv)%1000:<10}{str([round(sv.position[0], 2), round(sv.position[1], 2)]):<15}{sv.speed:<15.2f}{sv.action.get('acceleration', 0):<20.2f}{sv.heading:<20.2f}")

            if closest_vehicle1:
                print(f"[green]Closest vehicle1 found in the same lane: ID={id(closest_vehicle1)%1000}, Distance={min_distance1:.2f} meters[/green]")
            if closest_vehicle2:
                print(f"[green]Closest vehicle2 found in the same lane: ID={id(closest_vehicle2)%1000}, Distance={min_distance2:.2f} meters[/green]")
            
            print(f"[red]Ego collided with vehicle ID={id(return_vehicle)%1000}[/red]")

        self.plotNetwork(network, ax)
     
        egoVertices = self.getShape(ego)
        egoRectangle = Polygon(egoVertices, facecolor='#ff9f43')
        ax.add_patch(egoRectangle)
        ax.add_artist(Text(ego.position[0], ego.position[1], 'ego' , fontsize=6))

    
        for sv in SVs:
            svVertices = self.getShape(sv)
            svRectangle = Polygon(svVertices, facecolor='#1dd1a1')
            ax.add_patch(svRectangle)
            svText = Text(sv.position[0], sv.position[1], f'{id(sv)%1000}' ,fontsize=4)
            ax.add_artist(svText)

  
        ax.set_xlim(ego.position[0]-50, ego.position[0]+150)
        ax.set_ylim(ego.position[1]-20, ego.position[1]+20)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')

  
        plt.savefig(fileName, dpi=360)
        plt.close(fig)  

        return return_str, return_vehicle  
