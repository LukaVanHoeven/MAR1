from transition import Transition
from typing import Callable


class Dialogue_management_system:
    """
    
    """

    def __init__(
            self, 
            classifier_func: Callable, 
            transitions: list[Transition], 
            start_state: str
        ):
        """
        
        """
        self.classifier = classifier_func
        self.transitions = transitions
        self.current_state = start_state

        self.pricerange = None
        self.area = None
        self.food = None 

    def state_transition(self, user_utterance: str):
        """
        
        """
        # Classify using trained model
        dialogue_act = self.classifier(
            [user_utterance],
            "sequential_orig.keras",
            "sequential_orig.pickle"
        )[0]

        print(f"{dialogue_act = }")

        # Find patterns for pricerange, area, food.
        self.extract_preferences(user_utterance)

        for transition in self.transitions:
            if transition.original_state == self.current_state and transition.dialogue_act == dialogue_act:
                if transition.condition(self):
                    self.current_state = transition.next_state
                    return transition.next_state
                
        return self.current_state

    def extract_preferences(self, user_utterance: str):

        self.pricerange = "cheap"
        # self.area = "west"
        # self.food = "spanish"

    def loop(self):
        print("Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?")
        while True:
            user = input(">").lower()
            if user == "exit" or user == "end_conversation":
                break
            
            next_state = self.state_transition(user)
            self.area = "west"

            print(f"USER: {user}")
            print(f"NEXT_STATE: {next_state}")
            print("------------------")


if __name__ == "__main__":
    from ML_sequential import sequential

    transitions = [
        Transition(original_state="welcome", dialogue_act="hello", next_state="ask_area"),
        Transition(original_state="welcome", dialogue_act="inform", condition=lambda d: d.area is None, next_state="ask_area"),
        Transition(original_state="welcome", dialogue_act="inform", condition=lambda d: d.area is not None and d.pricerange is None, next_state="ask_pricerange"),
        Transition(original_state="welcome", dialogue_act="inform", condition=lambda d: d.area is not None and d.pricerange is not None and d.food is None, next_state="ask_food"),
        Transition(original_state="welcome", dialogue_act="inform", condition=lambda d: d.area is not None and d.pricerange is not None and d.food is not None, next_state="give_suggestion"),
        Transition(original_state="ask_area", dialogue_act="inform", condition=lambda d: d.pricerange is None, next_state="ask_pricerange"),
        Transition(original_state="ask_area", dialogue_act="inform", condition=lambda d: d.pricerange is not None and d.food is None, next_state="ask_food"),
        Transition(original_state="ask_area", dialogue_act="inform", condition=lambda d: d.pricerange is not None and d.food is not None, next_state="give_suggestion"),
        Transition(original_state="ask_pricerange", dialogue_act="inform", condition=lambda d: d.food is None, next_state="ask_food"),
        Transition(original_state="ask_pricerange", dialogue_act="inform", condition=lambda d: d.food is not None, next_state="give_suggestion"),
    ]

    dms = Dialogue_management_system(sequential, transitions, "welcome")
    dms.loop()
    