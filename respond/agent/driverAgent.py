import os
import textwrap
import time
from rich import print
from typing import List

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback, OpenAICallbackHandler, StreamingStdOutCallbackHandler

from respond.scenario.envScenario import EnvScenario

from highway_env.vehicle.controller import MDPVehicle
from typing import List, Tuple, Optional, Union, Dict


delimiter = "####"
example_message = textwrap.dedent(f"""\
        {delimiter} Driving scenario description:
        You are driving on a road with 4 lanes, and you are currently driving in the second lane from the left. Your speed is 25.00 m/s, acceleration is 0.00 m/s^2, and lane position is 363.14 m. 
        There are other vehicles driving around you, and below is their basic information:
        - Vehicle `912` is driving on the same lane of you and is ahead of you. The speed of it is 23.30 m/s, acceleration is 0.00 m/s^2, and lane position is 382.33 m.
        - Vehicle `864` is driving on the lane to your right and is ahead of you. The speed of it is 21.30 m/s, acceleration is 0.00 m/s^2, and lane position is 373.74 m.
        - Vehicle `488` is driving on the lane to your left and is ahead of you. The speed of it is 23.61 $m/s$, acceleration is 0.00 $m/s^2$, and lane position is 368.75 $m$.

        {delimiter} Your available actions:
        IDLE - remain in the current lane with current speed Action_id: 1
        Turn-left - change lane to the left of the current lane Action_id: 0
        Turn-right - change lane to the right of the current lane Action_id: 2
        Acceleration - accelerate the vehicle Action_id: 3
        Deceleration - decelerate the vehicle Action_id: 4
        """)
example_answer = textwrap.dedent(f"""\
        Well, I have 5 actions to choose from. Now, I would like to know which action is possible. 
        I should first check if I can acceleration, then idle, finally decelerate.  I can also try to change lanes but with caution and not too frequently.

        - I want to know if I can accelerate, so I need to observe the car in front of me on the current lane, which is car `912`. The distance between me and car `912` is 382.33 - 363.14 = 19.19 m, and the difference in speed is 23.30 - 25.00 = -1.7 m/s. Car `912` is traveling 19.19 m ahead of me and its speed is 1.7 m/s slower than mine. This distance is too close and my speed is too high, so I should not accelerate.
        - Since I cannot accelerate, I want to know if I can maintain my current speed. I need to observe the car in front of me on the current lane, which is car `912`. The distance between me and car `912` is 382.33 - 363.14 = 19.19 m, and the difference in speed is 23.30 - 25.00 = -1.7 m/s. Car `912` is traveling 19.19 m ahead of me and its speed is 1.7 m/s slower than mine. This distance is too close and my speed is too high, so if I maintain my current speed, I may collide with it.
        - Maintain my current speed is not a good idea, so I can only decelearate to keep me safe on my current lane. Deceleraion is a feasible action.
        - Besides decelearation, I can also try to change lanes. I should carefully check the distance and speed of the cars in front of me on the left and right lanes. Noted that change-lane is not a frequent action, so I should not change lanes too frequently.
        - I first try to change lanes to the left. The car in front of me on the left lane is car `488`. The distance between me and car `488` is 368.75-363.14=5.61 m, and the difference in speed is 23.61 - 25.00=-1.39 m/s. Car `488` is traveling 5.61 m ahead of me and its speed is 1.39 m/s slower than mine. This distance is too close, the safety lane-change distance is 25m. Besides, my speed is higher than the front car on the left lane. If I change lane to the left, I may collide with it.                                           So I cannot change lanes to the left.
        - Now I want to see if I can change lanes to the right. The car in front of me on the right lane is car 864. The distance between me and car 864 is 373.74-363.14 = 10.6 m, and the difference in speed is 23.61-25.00=-3.7 m/s. Car 864 is traveling 10.6 m ahead of me and its speed is 3.7 m/s slower than mine. The distance is too close and my speed is higher than the front car on the right lane. the safety lane-change distance is 25m. if I change lanes to the right, I may collide with it. So I cannot change lanes to the right.
        - Now my only option is to slow down to keep me safe.
        Final Answer: Deceleration
                                         
        Response to user:#### 4
        """)


class DriverAgent:
    def __init__(
        self, sce: EnvScenario,
        temperature: float = 0, verbose: bool = False
    ) -> None:
        self.sce = sce
        oai_api_type = os.getenv("OPENAI_API_TYPE")
         
        if oai_api_type == "openai":
            print("Use OpenAI API")
            if not os.getenv("OPENAI_BASE_URL"):    #use default base url
                self.llm = ChatOpenAI(
                    temperature=temperature,
                    callbacks=[
                        OpenAICallbackHandler()
                    ],
                    model_name=os.getenv("OPENAI_CHAT_MODEL"),
                    max_tokens=2000,
                    request_timeout=60,
                    streaming=True,
                )
            else:
                self.llm = ChatOpenAI(
                    temperature=temperature,
                    callbacks=[
                        OpenAICallbackHandler()
                    ],
                    model_name=os.getenv("OPENAI_CHAT_MODEL"),
                    max_tokens=2000,
                    request_timeout=60,
                    streaming=True,
                    base_url=os.getenv("OPENAI_BASE_URL"),
                )
            print("[cyan]using model:[/cyan]", os.getenv("OPENAI_CHAT_MODEL"))

    def few_shot_decision(self, scenario_description: str = "Not available", previous_decisions: str = "Not available", available_actions: str = "Not available", driving_intensions: str = "Not available", fewshot_messages: List[str] = None, fewshot_answers: List[str] = None):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

        system_message = textwrap.dedent(f"""\
        You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 

        Make sure to include {delimiter} to separate every step.
        """)

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving Intensions:
        {driving_intensions}
        {delimiter} Available actions:
        {available_actions}

        You can stop reasoning once you have a valid action to take. 
        """
        human_message = human_message.replace("        ", "")

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),

        ]
        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )

        start_time = time.time()
        local_time = time.localtime(start_time)
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)

        #print("[cyan]print out more info for reference:[/cyan]")
        print(formatted_time)
        
        #print(messages)
        #print("[cyan]print out more info finish ——————————————————————:[/cyan]")    

        print("[cyan]Agent answer:[/cyan]")
        response_content = ""
        for chunk in self.llm.stream(messages):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        decision_action = response_content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a int number, checking the output...")
            check_message = f"""
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
            messages = [
                HumanMessage(content=check_message),
            ]

            print("[cyan]print out more info in case action not in number:[/cyan]")
            print(messages)
            print("[cyan]print out more info finish ——————————————————————:[/cyan]")

            with get_openai_callback() as cb:
                check_response = self.llm(messages)

            print("[cyan]print out llm response in case action not in number:[/cyan]")
            print(check_message)
            print("[cyan]print out more info finish ——————————————————————:[/cyan]")
            
            result = int(check_response.content.split(delimiter)[-1])

        few_shot_answers_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_answers_store += fewshot_answers[i] + \
                "\n---------------\n"
        print("Result:", result)
        return result, response_content, human_message, few_shot_answers_store

    # zero-shot decision with risk value (embedded in the scenario description)
    def zero_shot_decision(self, scenario_description: str = "Not available", previous_decisions: str = "Not available", available_actions: str = "Not available", driving_intensions: str = "Not available"):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

        system_message = textwrap.dedent(f"""\
        You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 

        Make sure to include {delimiter} to separate every step.
        """)
        # belows are adding risk value related description

        system_message = system_message + "Besides the current scenario description, you will also be provided with a Driving Risk Value which describes the Risk around you. The Risk Value is a float number from 0.00 to 1.00. The value 0.00 means no risk, and 1.00 means the highest risk. Overall Left Risk Value, which use to remind you that if you turn left, the risk possibility. Overall Right Risk Value, which use to remind you that if you turn right, the risk possibility. Front Risk Value, which use to remind you that the risk possibility in your front (the risk you will collide the front car, based on you and the front car's relative velocity and distance). Behind Risk Value, which use to remind you that the risk possibility behind you (the risk you will be collided by the car behind you, based on you and the behind car's relative velocity and distance)."
        system_message = system_message + "\nSome tips and references on risk value:\n1. basically you should keep staying in your current lane in the majority time, to avoid change lane too offen. you will be provided the action that you have just performed before this action. For example, if you have just turned right and then changed to current lane, in most of time, you had better not choose turn left to go back again. unless the behind and front risk are too high in current lane. if you have just turned left and then changed to current lane, in most of time, you had better not choose turn right to go back again. In current situation, Your previous action is: " + previous_decisions + ";\n"
        system_message = system_message + "2. When the risk value of behind risk is high, and the front, left, and right sides are 0.00 or low, you can consider taking action to the direction with low risk, but not necessary to the loweast. Normally the risk value under 0.75 will be safe, and you should avoid changed lane too much, unless the currently risks (the front risk & the behind risk) is comparative too high than the other lane;\n 3. You can comprehensively compare the values of front risk, behind risk, left risk, and right risk, and choose the one with 0.00, or the one with the comparative low value (not need the lowest), which is usually a safer approach.\n"
        system_message = system_message + "\nYou should comprehensively consider the current driving situation and the conditions of each vehicle, as well as the risk value given by the system, so as to make reasonable recommendations for the next driving action."

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. 

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving Intensions:
        {driving_intensions}
        {delimiter} Available actions:
        {available_actions}

        You can stop reasoning once you have a valid action to take. 
        """
        human_message = human_message.replace("        ", "")

        messages = [
            SystemMessage(content=system_message),
            # HumanMessage(content=example_message),
            # AIMessage(content=example_answer),
        ]

        messages.append(
            HumanMessage(content=human_message)
        )

        start_time = time.time()
        local_time = time.localtime(start_time)
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)


        print(formatted_time)
        
        #print(messages)
        #print("[cyan]print out more info finish ——————————————————————:[/cyan]")
            

        print("[cyan]Agent answer:[/cyan]")
        response_content = ""
        for chunk in self.llm.stream(messages):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)
        print("\n")
        decision_action = response_content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result < 0 or result > 4:
                raise ValueError
        except ValueError:
            print("Output is not a int number, checking the output...")
            check_message = f"""
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
            messages = [
                HumanMessage(content=check_message),
            ]

            print("[cyan]print out more info in case action not in number:[/cyan]")
            print(messages)
            print("[cyan]print out more info finish ——————————————————————:[/cyan]")

            with get_openai_callback() as cb:
                check_response = self.llm(messages)

            print("[cyan]print out llm response in case action not in number:[/cyan]")
            print(check_message)
            print("[cyan]print out more info finish ——————————————————————:[/cyan]")
            
            result = int(check_response.content.split(delimiter)[-1])

        print("Result:", result)
        return result, response_content, human_message

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
            
            #check_message = f"""
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
