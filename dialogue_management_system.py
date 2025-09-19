from transition import Transition
from typing import Callable
import preference_statement

class Dialogue_management_system:
    """
    A class to manage the dialogue system for restaurant recommendations.
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
        
        self.available_suggestions = ["dummy1", "dummy2", "dummy3"]

        self.pricerange = None
        self.area = None
        self.food = None 
        
        self.preference_extractor = preference_statement.PreferenceStatement().parse_preference_statement
        
        self.available_suggestions = []

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
            if transition.original_state == self.current_state and dialogue_act in transition.dialogue_act:
                if transition.condition(self):
                    self.current_state = transition.next_state
                    return transition.next_state
                
        return self.current_state

    def extract_preferences(self, user_utterance: str):
        statement_parser = self.preference_extractor(user_utterance)
        for key, value in statement_parser.items():
            if value is not None:  # only update if we got something useful
                if key == "pricerange":
                    self.pricerange = value
                elif key == "area":
                    self.area = value
                elif key == "food":
                    self.food = value

    def loop(self):
        print("Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?")
        while True:
            user = input(">").lower()
            if user == "exit" or user == "end_conversation":
                break
            previous_state = self.current_state
            
            next_state = self.state_transition(user)

            # HERE we should have a checker for possible suggestions 
            self.print_next_conversation_step(previous_state == next_state)
            self.area = "west"
            print(f"USER: {user}")
            print(f"NEXT_STATE: {next_state}")
            print("------------------")
            
    def print_next_conversation_step(self, repeat=False):
        match self.current_state, repeat:
            case "welcome", False:
                print("Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?")
            case "ask_area", False:
                print("In which area would you like to eat?")
            case "ask_pricerange", False:
                print("What price range are you looking for?")
            case "ask_food", False:
                print("What kind of food would you like?")
            case "give_suggestion", False:
                if len(self.available_suggestions) > 0:
                    suggestion = self.available_suggestions.pop()
                    print(f"I have found a restaurant that matches your preferences. It is {suggestion} food in the {self.area} area with a {self.pricerange} price range.")
                else:
                    print("I am sorry, I do not have any more suggestions that match your criteria.")
            case "pick_suggested_or_restart", False:
                print("Unfortunately there are no other restaurants matching your criteria. Would you like to pick the suggested restaurant or start over?")
            case "end_conversation", False:
                print("Thank you for using the Cambridge restaurant system. Goodbye!")
            case "welcome", True:
                print("I am sorry, I did not understand that. How may I help you?")
            case "ask_area", True:
                print("I am sorry, I did not understand that. In which area would you like to eat?")
            case "ask_pricerange", True:
                print("I am sorry, I did not understand that. What price range are you looking for (cheap, moderate or expensive)?")
            case "ask_food", True:
                print("I am sorry, we do not have that kind of food. What kind of food would you like?")
            case "give_suggestion", True:
                print(f"I have found another restaurant that matches your preferences. It is {self.food} food in the {self.area} area with a {self.pricerange} price range.")
            case "pick_suggested_or_restart", True:
                print("I am sorry, I did not understand that. Unfortunately there are no other restaurants matching your criteria. Would you like to pick the suggested restaurant or start over?")
            case "end_conversation", True:
                print("I am sorry, I did not understand that. Thank you for using the Cambridge restaurant system. Goodbye!")

if __name__ == "__main__":
    from ML_sequential import sequential

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
        Transition(original_state="pick_suggestion_or_restart", dialogue_act=["repeat", "null", "bye", "reqmore", "reqalts", "request"], next_state="pick_suggestion_or_restart"),

        # If the user agrees on the suggestion it should move to end conversation
        Transition(original_state="give_suggestion", dialogue_act=["ack", "affirm", "thankyou"], next_state="end_conversation"),

        # If the user does not like the suggestions and there are other suggestions the user should get a new suggestion
        Transition(original_state="give_suggestion", dialogue_act=["deny", "negate", "reqalts"], condition=lambda d:len(d.available_suggestions) > 0, next_state="give_suggestion"),

        # If the user does not like the suggestion the user can pick the last suggested suggestion or end the conversation
        Transition(original_state="give_suggestion", dialogue_act=["deny", "negate", "reqalts"], condition=lambda d:len(d.available_suggestions) == 0, next_state="pick_suggested_or_restart"),

        # If the user takes the last offered suggestion it ends the conversation
        Transition(original_state="pick_suggested_or_restart", dialogue_act=["ack", "affirm", "thankyou"], next_state="end_conversation"),
        
        # If the user does not like the last offered suggestion it goes back to the welcome state
        Transition(original_state="pick_suggested_or_restart", dialogue_act=["deny", "negate", "restart"], next_state="welcome"),
    ]

    dms = Dialogue_management_system(sequential, transitions, "welcome")
    dms.loop()
    