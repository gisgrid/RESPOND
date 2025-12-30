import os
import textwrap
import time
from rich import print

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import OpenAICallbackHandler

from respond.scenario.envScenario import EnvScenario

from highway_env.vehicle.controller import MDPVehicle
from typing import Tuple, Optional, Union, Dict


delimiter = "####"

class RiskPatternReflectionAgent:
    def __init__(
        self, sce: EnvScenario,
        temperature: float = 0, verbose: bool = False
    ) -> None:
        self.sce = sce      
        oai_api_type = os.getenv("OPENAI_API_TYPE")
        if oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                callbacks=[
                    OpenAICallbackHandler()
                ],
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=temperature,
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
            )
        elif oai_api_type == "openai":
            print("Use OpenAI API")
            self.llm = ChatOpenAI(
                temperature=temperature,
                callbacks=[
                    OpenAICallbackHandler()
                ],
                # model_name=os.getenv("OPENAI_CHAT_MODEL"),
                model_name=os.getenv("OPENAI_REFLECTION_MODEL"), 
                max_tokens=2000,
                request_timeout=60,
                streaming=True,
                base_url="https://api.openai-proxy.org/v1",
            )
            print("[cyan]using model:[/cyan]", os.getenv("OPENAI_CHAT_MODEL"))


    def reflection_in_risk_pattern_mode(
        self,
        ego_car: MDPVehicle, 
        previous_scenario: str,  
        current_scenario: str,  
        action_done: int,  
        car_crash: Optional[Dict[str, Union[int, float]]]  
    ) -> Tuple[str, int]:  

        delimiter = "####"

        system_message = textwrap.dedent(f"""\
        You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex driving scenarios.
        You will be given 3 parts of information:
        1. The previous driving scenario description, which is the scenario before the ego car took the action that caused a collision.
        2. The action that ego car took in the previous scenario, which caused a collision. 
        3. And the specific car that ego car collided with.
                                         
        You should carefully check the previous scenario description, the action that ego car took, and find out the reason why the mistake happened, and then give a better action to avoid the mistake, and to serve the driving intention better.
        
        You answer should use the following format:
        {delimiter} Aanalysis of the mistake:
        <Your analysis of why the mistake happened, and why the action that ego car took caused a collision.>
        {delimiter} The reflection text in short, to point out what the new action should be takedn, conclude the mistake before cause collision and why the new action would be better.
        <Your reflection text, which should be a short text here, better not more than 100 words.>
        {delimiter} The corrected action which you think is better in that previous scenario and can avoid the collision:
        <Your corrected action, which should be a int number of the action_id, with a delimiter {delimiter} just before the integer action_id, without any action name or explanation. The output decision must be unique and not ambiguous. the action_id should within the range of available actions mentioned in the previous scenario description, it should be one of the following: 0, 1, 2, 3, 4.>
        
        
        below is three examples of the ouput format, you should follow the format strictly, and the decision thinking process you could refer as well (to considering the risk values and together with the distance between you and other cars, and the relative speeds).
        Example output 1:
        {delimiter} Aanalysis of the mistake:
        The mistake happened because the ego car did not check the distance and speed of the car in front of it before taking the action. The action that ego car took was to accelerate, which caused a collision with the car in front of it.
        {delimiter} The reflection text in short, to conclude the mistake before cause collision why the new action would be better.
        {delimiter} The new action is to IDLE, since the front risk value is already 1.0 and the distance between ego and the front car in same lane is less than 30 meter, the relative velocity (front vehicle velocity minus ego) is -5 m/s, which means it is not properly to do accelerate.
        {delimiter} The corrected action which will be better in that previous scenario and can avoid the collision:
        {delimiter} 1

        Example output 2:
        {delimiter} Aanalysis of the mistake:
        The mistake happened because the ego car did not check the distance and speed of the cars which in the left handed lane, to see whether they are very closed, before taking the Turn-left action. 
        {delimiter} The reflection text in short, to conclude the mistake before cause collision why the new action would be better.
        {delimiter} The new action is Acceleration, since the front risk value is under 0.5, the distance between ego and the front car in the same lane is more than 40 meter, the relative velocity (front vehicle velocity minus ego) is -5 m/s, which means that even after one second, the distance between ego and the front vehicle is still larger than 30m, the risk value and the forecasted distance after action taken is still under control. 
        If the ego car's speed is not higher than the road speed level (the average speed of the road), and the front risk value is not so high, and the front distance is large enouth, the ego car should not try to use turn-left or turn-right action, to avoid turning too frequently. 
        The ego car should try to use turn action, while the front risk value and the behind risk value is relative high, and the left or right lane is safe and the room (the distance) after ego turning is large enough compare to the current lane.
        {delimiter} The corrected action which will be better in that previous scenario and can avoid the collision:
        {delimiter} 3

        Example output 3:
        {delimiter} Aanalysis of the mistake:
        The mistake happened because the ego car did not check the distance and speed of the cars which in the right handed lane, to see whether they are very closed, before taking the Turn-right action. 
        {delimiter} The reflection text in short, to conclude the mistake before cause collision why the new action would be better.
        {delimiter} The new action is Deceleration, since the front risk value is higher than 0.9, and the distance between ego and the front car in the same lane is less than 30 meter, the ego speed is already large than the front car. As contrast, the behind risk value is lower than 0.4, and the distance between ego and the behind car is bigger than the distance between ego to the front vehicle. 
        If the ego car's speed is not much lower than the road speed level (the average speed of the road), and the front risk value is high, but the behind risk value is low which means that the distance between ego and the behind car in the same lane is enough.  The ego car should not try to use turn-left or turn-right action, to avoid turning too frequently. 
        The ego car should try to use turn action, while the front risk value and the behind risk value is relative high, and the left or right lane is safe and the room (the distance) after ego turning is large enough compare to the current lane.
        {delimiter} The corrected action which will be better in that previous scenario and can avoid the collision:
        {delimiter} 4
        
        above is the examples of the output format, you should follow the format strictly, and the decision thinking process you could refer as well.
        
        """)

        human_message = textwrap.dedent(f"""\
        You are a self-reflection assistant who is responsible for helping the ego car to reflect on the previous driving scenario and the action that caused a collision.
        Here is the three parts of information given to you:
        {delimiter} Previous driving scenario description:
        {previous_scenario}
        {delimiter} The action that ego car took in the previous scenario, which caused a collision:
        {action_done}
        {delimiter} The specific car that ego car collided with:
        {car_crash}
        """)

        start_time = time.time()
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]
        print("Self-reflection is running, make take time...")
        print("[red]extra print out the messages[/red] for checking reflection input messages:", messages)
        
        response = self.llm(messages)
        
        print("\n[green]Self-reflection done, response: [/green]", response.content)
        print("[red]extra print out the messages finish [/red]" )

        try:
            reflection_text = response.content.split(delimiter)[-3].strip()
            print("[green]Reflection text: [/green]", reflection_text)
        except IndexError:
            print("[red]Error: Not enough parts in the response content to extract reflection text.[/red]")
            reflection_text = "Reflection text could not be extracted due to insufficient parts."

        decision_action = response.content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            
            print("[red]Output is not a int number, checking the output...[/red]")
            result = -1  
            
            """
            You are a output checking assistant who is responsible for checking the output of another agent.
            
            The output you received is: {decision_action}

            Your should just output the right int type of action_id, with no other characters or delimiters.
            i.e. :
            | Action_id | Action Description                                     |
            |--------|--------------------------------------------------------|
            | 0      | Turn-left: change lane to the left of the current lane |
            | 1      | IDLE: remain in the current lane with current speed   |
            | 2      | Turn-right: change lane to the right of the current lane|
            | 3      | Acceleration: accelerate the vehicle                 |
            | 4      | Deceleration: decelerate the vehicle                 |

            You answer format would be:
            {delimiter} <correct action_id within 0-4>
            """
            """messages = [
                HumanMessage(content=check_message),
            ]

            with get_openai_callback() as cb:
                check_response = self.llm(messages)

            result = int(check_response.content.split(delimiter)[-1])
            """
       
        print("Reflection done. Time taken: {:.2f}s".format(
            time.time() - start_time))

        return reflection_text, result  
