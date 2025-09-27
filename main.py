from transition import Transition
from ML_sequential import sequential
from train_sequential import train_sequential

import preference_handler
import argparse
import time

import pandas as pd
from typing import Callable


class Dialogue_management_system:
    """
    A class to manage the dialogue system for restaurant recommendations.
    """
    def __init__(
            self, 
            classifier_func: Callable, 
            transitions: list[Transition], 
            start_state: str,
            allow_preference_change: bool=False,
            all_caps: bool=False,
            system_delay: bool=False
        ):
        """
        The constructor for Dialogue_management_system class.
        @params:
            - classifier_func (Callable): A function that takes in a user utterance and returns a dialogue act.
            - transitions (list[Transition]): A list of Transition objects that define the possible transitions in the dialogue system.
            - start_state (str): The state the system starts in.
        """
        self.classifier = classifier_func
        self.transitions = transitions
        self.current_state = start_state

        self.allow_preference_change = allow_preference_change
        self.all_caps = all_caps
        self.system_delay = system_delay
        
        self.available_suggestions = []
        self.gathered_suggestions = False

        self.pricerange = None
        self.area = None
        self.food = None 
        self.additional = None
        
        self.preference_statement = preference_handler.PreferenceHandler()
        self.preference_extractor = self.preference_statement.parse_preference_statement

        self.available_suggestions = []

    def state_transition(self, user_utterance: str):
        """
        This function takes in a user utterance and returns the next state of the dialogue system.
        It uses the classifier function to classify the user utterance into a dialogue act.
        """
        # Classify using trained model
        dialogue_act = self.classifier(
            [user_utterance],
            "sequential_orig.keras",
            "sequential_orig.pickle"
        )[0]

        # Find patterns for pricerange, area, food.
        self.extract_preferences(user_utterance)
        
        if self.pricerange and self.area and self.food and not self.gathered_suggestions:
            self.gathered_suggestions = True
            self.available_suggestions = self.preference_statement.find_matching_restaurants(self.area, self.food, self.pricerange)
            
        for transition in self.transitions:
            if transition.original_state == self.current_state and dialogue_act in transition.dialogue_act:
                if transition.condition(self):
                    self.current_state = transition.next_state
                    return transition.next_state
                
        return self.current_state

    def extract_preferences(self, user_utterance: str):
        """
        This function extracts the user's preferences for food, area, and price range from their utterance.
        """
        statement_parser = self.preference_extractor(user_utterance)
        for key, value in statement_parser.items():
            if value is not None:  # only update if we got something useful
                if key == "pricerange":
                    if self.pricerange is None or self.allow_preference_change:
                        self.pricerange = value
                    else:
                        self._print(
                            f"\033[93mYou already selected a pricerange " \
                            f"({self.pricerange}), and changing it is not " \
                            f"allowed.\033[0m"
                        )
                elif key == "area":
                    if self.area is None or self.allow_preference_change:
                        self.area = value
                    else:
                        self._print(
                            f"\033[93mYou already selected an area " \
                            f"({self.area}), and changing it is not " \
                            f"allowed.\033[0m"
                        )
                elif key == "food":
                    if self.food is None or self.allow_preference_change:
                        self.food = value
                    else:
                        self._print(
                            f"\033[93mYou already selected a food " \
                            f"({self.food}), and changing it is not " \
                            f"allowed.\033[0m"
                        )
                elif key == "additional":
                    if self.additional is None or self.allow_preference_change:
                        self.additional = value
                    else:
                        self._print(
                            f"\033[93mYou already selected additional preferences " \
                            f"({self.additional}), and changing it is not " \
                            f"allowed.\033[0m"
                        )

    def loop(self):
        """
        This function starts the dialogue loop, allowing the user to interact with the system.
        """
        self._print("\033[93mHello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?\033[0m")
        while True:
            
            user = input(">").lower()
            if user == "exit" or user == "end_conversation":
                break
            
            if self.current_state == "welcome":
                self.pricerange = None
                self.area = None
                self.food = None 
                self.available_suggestions = []
                self.gathered_suggestions = False
                
            previous_state = self.current_state
            
            next_state = self.state_transition(user)

            
            self.print_next_conversation_step(previous_state == next_state)
            if next_state == "end_conversation":
                break
            
    def print_next_conversation_step(self, repeat=False):
        """
        This function prints the next step in the conversation based on the current state and whether it is a repeat.
        """
        match self.current_state, repeat:
            case "welcome", False:
                self._print("\033[93mHello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?\033[0m")
            case "ask_area", False:
                self._print("\033[93mIn which area would you like to eat?\033[0m")
            case "ask_pricerange", False:
                self._print("\033[93mWhat price range are you looking for?\033[0m")
            case "ask_food", False:
                self._print("\033[93mWhat kind of food would you like?\033[0m")
            case "give_suggestion", False:
                if len(self.available_suggestions) > 0:
                    suggestion = self.available_suggestions.pop(0)
                    self._print(f"\033[93mI have found a restaurant that matches your preferences. It is {suggestion} food in the {self.area} area with a {self.pricerange} price range.\033[0m")
                else:
                    self._print("\033[93mI am sorry, I do not have any more suggestions that match your criteria.\033[0m")
            case "pick_suggested_or_restart", False:
                self._print("\033[93mUnfortunately there are no other restaurants matching your criteria. You can either pick the suggested restaurant or start over. Would you like to pick the suggested restaurant?\033[0m")
            case "end_conversation", False:
                self._print("\033[93mThank you for using the Cambridge restaurant system. Goodbye!\033[0m")
            case "welcome", True:
                self._print("\033[93mI am sorry, I did not understand that. How may I help you?\033[0m")
            case "ask_area", True:
                self._print("\033[93mI am sorry, I did not understand that. In which area would you like to eat?\033[0m")
            case "ask_pricerange", True:
                self._print("\033[93mI am sorry, I did not understand that. What price range are you looking for (cheap, moderate or expensive)?\033[0m")
            case "ask_food", True:
                self._print("\033[93mI am sorry, we do not have that kind of food. What kind of food would you like?\033[0m")
            case "give_suggestion", True:
                if len(self.available_suggestions) > 0:
                    suggestion = self.available_suggestions.pop(0)
                    self._print(f"\033[93mI have found another restaurant that matches your preferences called {suggestion}. It is {self.food} food in the {self.area} area with a {self.pricerange} price range.\033[0m")
                else:
                    self._print("\033[93mI am sorry, I did not understand that. Unfortunately there are no other restaurants matching your criteria.\033[0m")
            case "pick_suggested_or_restart", True:
                self._print("\033[93mI am sorry, I did not understand that. Unfortunately there are no other restaurants matching your criteria. Will you accept the given suggestion or start over?\033[0m")
            case "end_conversation", True:
                self._print("\033[93mI am sorry, I did not understand that. Thank you for using the Cambridge restaurant system. Goodbye!\033[0m")
        
    def _print(self, msg: str):
        """
        
        """
        if self.all_caps:
            msg = msg.upper()
        if self.system_delay:
            time.sleep()
        print(msg)
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run the Dialogue Management System.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model before running the dialogue system"
    )
    parser.add_argument(
        "--allow-preference-change",
        action="store_true",
        default=False,
        help="Allow the user to adjust their preferences (default: False)"
    )
    parser.add_argument(
        "all-caps",
        action="store_true",
        default=False,
        help="Output the response of the system in all caps (default: False)"
    )
    parser.add_argument(
        "system-delay",
        action="store_true",
        default=False,
        help="Add a 1 second delay before each system response (default: False)"
    )
    args = parser.parse_args()

    transitions = [
        Transition(original_state="welcome", dialogue_act=["hello"], next_state="ask_area"),
        Transition(original_state="welcome", dialogue_act=["inform"], condition=lambda d: d.area is not None and d.pricerange is not None and d.food is not None, next_state="give_suggestion"),
        Transition(original_state="welcome", dialogue_act=["inform"], condition=lambda d: d.area is not None and d.pricerange is not None and d.food is None, next_state="ask_food"),
        Transition(original_state="welcome", dialogue_act=["inform"], condition=lambda d: d.area is not None and d.pricerange is None, next_state="ask_pricerange"),
        Transition(original_state="welcome", dialogue_act=["inform"], condition=lambda d: d.area is None, next_state="ask_area"),
        
        Transition(original_state="ask_area", dialogue_act=["inform"], condition=lambda d: d.area is None, next_state="ask_area"),
        Transition(original_state="ask_area", dialogue_act=["inform"], condition=lambda d: d.pricerange is not None and d.food is not None and d.area is not None, next_state="give_suggestion"),
        Transition(original_state="ask_area", dialogue_act=["inform"], condition=lambda d: d.pricerange is not None and d.food is None, next_state="ask_food"),
        Transition(original_state="ask_area", dialogue_act=["inform"], condition=lambda d: d.pricerange is None, next_state="ask_pricerange"),
        
        Transition(original_state="ask_pricerange", dialogue_act=["inform"], condition=lambda d:d.pricerange is None, next_state="ask_pricerange"),
        Transition(original_state="ask_pricerange", dialogue_act=["inform"], condition=lambda d: d.food is not None and d.pricerange is not None and d.area is not None, next_state="give_suggestion"),
        Transition(original_state="ask_pricerange", dialogue_act=["inform"], condition=lambda d: d.food is None, next_state="ask_food"),
        Transition(original_state="ask_pricerange", dialogue_act=["inform"], condition=lambda d: d.area is None, next_state="ask_area"),
        
        Transition(original_state="ask_food", dialogue_act=["inform"], condition=lambda d: d.food is None, next_state="ask_food"),
        Transition(original_state="ask_food", dialogue_act=["inform"], condition=lambda d: d.food is not None and d.pricerange is not None and d.area is not None, next_state="give_suggestion"),
        Transition(original_state="ask_food", dialogue_act=["inform"], condition=lambda d: d.pricerange is None, next_state="ask_pricerange"),
        Transition(original_state="ask_food", dialogue_act=["inform"], condition=lambda d: d.area is None, next_state="ask_area"),

        # For the repeat, null, bye, reqmore, reqalts, request dialogue acts we stay in the same state
        Transition(original_state="welcome", dialogue_act=["repeat", "null", "bye", "reqmore", "reqalts", "request"], next_state="welcome"),
        Transition(original_state="ask_area", dialogue_act=["repeat", "null", "bye", "reqmore", "reqalts", "request"], next_state="ask_area"),
        Transition(original_state="ask_pricerange", dialogue_act=["repeat", "null", "bye", "reqmore", "reqalts", "request"], next_state="ask_pricerange"),
        Transition(original_state="give_suggestion", dialogue_act=["repeat", "null", "bye", "reqmore", "reqalts", "request"], next_state="give_suggestion"),

        # If the user agrees on the suggestion it should move to end conversation
        Transition(original_state="give_suggestion", dialogue_act=["ack", "affirm", "thankyou"], next_state="end_conversation"),

        # If the user does not like the suggestions and there are other suggestions the user should get a new suggestion
        Transition(original_state="give_suggestion", dialogue_act=["deny", "negate", "reqalts"], condition=lambda d:len(d.available_suggestions) > 0, next_state="give_suggestion"),

        # If the user does not like the suggestion the user can pick the last suggested suggestion or end the conversation
        Transition(original_state="give_suggestion", dialogue_act=["deny", "negate", "reqalts"], condition=lambda d:len(d.available_suggestions) == 0, next_state="pick_suggested_or_restart"),

        # If the user takes the last offered suggestion it ends the conversation
        Transition(original_state="pick_suggestion_or_restart", dialogue_act=["repeat", "null"], next_state="pick_suggestion_or_restart"),
        Transition(original_state="pick_suggested_or_restart", dialogue_act=["ack", "affirm", "thankyou", "bye"], next_state="end_conversation"),
        
        # If the user does not like the last offered suggestion it goes back to the welcome state
        Transition(original_state="pick_suggested_or_restart", dialogue_act=["deny", "negate", "restart"], next_state="welcome"),
    ]
    
    #Train the sequential model so we have something to use (Mainly for being able to just run 1 file for handing in the assignment)
    if args.train:
        orig_train = pd.read_csv("train_orig.csv")
        model_sequential_orig, tokenizer_sequential_orig = train_sequential("sequential_orig", orig_train)
    
    dms = Dialogue_management_system(
        sequential,
        transitions,
        "welcome",
        args.allow_preference_change,
        args.all_caps,
        args.system_delay
    )
    dms.loop()
    