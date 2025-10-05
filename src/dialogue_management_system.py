from .transition import Transition

from .preference_handler import PreferenceHandler
import time

from typing import Callable

from .rulebased import rulebased

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
            system_delay: bool=False,
            use_baseline: bool=False
        )-> None:
        """
        The constructor for Dialogue_management_system class.
        @param classifier_func (Callable): A function that takes in a
            user utterance and returns a dialogue act.
        @param transitions (list[Transition]): A list of Transition
            objects that define the possible transitions in the dialogue
            system.
        @param start_state (str): The state the system starts in.
        @param allow_preference_change (bool): If True this allows the
            user to change their preferences after already haven given
            a preference of the same category. If False this locks a
            given preference as permanent.
        @param all_caps (bool): If True all system outputs will be
            displayed in all caps. If False all system outputs will be
            in lower case.
        @param system_delay (bool): If True every time the system gives
            an output it will wait 1 second before printing. If False
            the output will be printed ASAP.
        @param use_baseline (bool): If True the system will use the
            rule-based baseline. If False it will use the machine
            learning classifier.
        """
        self.classifier = classifier_func
        self.transitions = transitions
        self.current_state = start_state

        self.allow_preference_change = allow_preference_change
        self.all_caps = all_caps
        self.system_delay = system_delay
        self.use_baseline = use_baseline

        self.available_suggestions = []
        self.picked_suggestion = None
        self.requested_additional_info = []
        self.gathered_suggestions = False

        self.pricerange = None
        self.area = None
        self.food = None
        self.additional = None

        self.preference_statement = PreferenceHandler()
        self.preference_extractor = \
            self.preference_statement.parse_preference_statement


    def state_transition(self, user_utterance: str)-> str:
        """
        This function takes in a user utterance and returns the next
        state of the dialogue system. It uses the classifier function or
        the rule-based baseline to classify the user utterance into a 
        dialogue act.

        @param user_utterance (str): The input from the user.

        @return (str): The name of the next state.
        """
        # Classify using trained model or rule-based baseline
        if self.use_baseline:
            dialogue_act = rulebased(user_utterance)
        else:
            dialogue_act = self.classifier(
                [user_utterance],
                "models/sequential_orig.keras",
                "models/sequential_orig.pickle"
            )[0]

        # Find patterns for pricerange, area, food.
        if (dialogue_act == "inform" or dialogue_act == "null") and \
            self.current_state != "ask_additional_info":
            self.extract_preferences(user_utterance)

        if self.current_state == "ask_additional_info":
            self.requested_additional_info = \
                self.preference_statement.parse_info_request(user_utterance)

        if self.pricerange and self.area and self.food and \
                not self.gathered_suggestions:
            self.gathered_suggestions = True
            self.available_suggestions = \
                self.preference_statement.find_matching_restaurants(
                    self.area,
                    self.food,
                    self.pricerange
                )

        for transition in self.transitions:
            if transition.original_state == self.current_state and \
                    dialogue_act in transition.dialogue_act:
                if transition.condition(self):
                    self.current_state = transition.next_state
                    return transition.next_state

        return self.current_state

    def extract_preferences(self, user_utterance: str)-> None:
        """
        This function extracts the user's preferences for food, area,
        and price range from their utterance.

        @param user_utterance (str): The input from the user.
        """
        statement_parser = self.preference_extractor(user_utterance)
        for key, value in statement_parser.items():
            if value is not None and self.current_state != "give_suggestion":  # only update if we got something useful
                if key == "pricerange":
                    if self.pricerange is None or self.allow_preference_change:
                        self.pricerange = value
                    else:
                        self._print(
                            f"You already selected a pricerange " \
                            f"({self.pricerange}), and changing it is not " \
                            f"allowed."
                        )
                elif key == "area":
                    if self.area is None or self.allow_preference_change:
                        self.area = value
                    else:
                        self._print(
                            f"You already selected an area " \
                            f"({self.area}), and changing it is not " \
                            f"allowed."
                        )
                elif key == "food":
                    if self.food is None or self.allow_preference_change:
                        self.food = value
                    else:
                        self._print(
                            f"You already selected a food " \
                            f"({self.food}), and changing it is not " \
                            f"allowed."
                        )
                elif key == "additional":
                    if self.additional is None or self.allow_preference_change:
                        self.additional = value
                    else:
                        self._print(
                            f"You already selected additional preferences " \
                            f"({self.additional}), and changing it is not " \
                            f"allowed."
                        )

    def loop(self)-> None:
        """
        This function contains the dialogue loop, allowing the user to
        interact with the system.
        """
        self._print(
            "Hello, welcome to the Cambridge restaurant system! You can " \
            "ask for restaurants by area, price range or food type. How " \
            "may I help you?"
        )
        while True:

            user = input(">").lower()
            if user == "exit" or user == "end_conversation":
                break
            elif user in ["restart", "start", "start over"]:
                self.current_state = "welcome"
                self.pricerange = None
                self.area = None
                self.food = None
                self.additional = None
                self.available_suggestions = []
                self.gathered_suggestions = False
                self.picked_suggestion = None
                self.requested_additional_info = []
                self.print_next_conversation_step(repeat=False)
                continue

            if self.current_state == "welcome":
                self.pricerange = None
                self.area = None
                self.food = None
                self.additional = None
                self.available_suggestions = []
                self.gathered_suggestions = False

            previous_state = self.current_state

            next_state = self.state_transition(user)

            self.print_next_conversation_step(previous_state == next_state)
            if next_state == "end_conversation":
                break

    def print_next_conversation_step(self, repeat: bool=False)-> None:
        """
        This function prints the next step in the conversation based on
        the current state and whether it is a repeat.

        @param repeat (bool): If true the state has not transitioned to
            a new state, so the system output needs to be different.
        """
        match self.current_state, repeat:
            case "welcome", False:
                self._print("Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?")
            case "ask_area", False:
                self._print("In which area would you like to eat?")
            case "ask_pricerange", False:
                self._print("What price range are you looking for?")
            case "ask_food", False:
                self._print("What kind of food would you like?")
            case "extra_requirements", False:
                self._print("Do you have any additional requirements? For example do you want the restaurant to be romantic, family-friendly, or quick?")
            case "ask_additional_info", True | False:
                if len(self.requested_additional_info) > 0:
                    if "phone" in self.requested_additional_info:
                        self._print(f"The phone number of {self.picked_suggestion['restaurantname']} is {self.picked_suggestion['phone']}. Do you need other information or exit?")
                    if "address" in self.requested_additional_info:
                        self._print(f"The address of {self.picked_suggestion['restaurantname']} is {self.picked_suggestion['addr']}. Do you need other information or exit?")
                    if "postcode" in self.requested_additional_info:
                        self._print(f"The postcode of {self.picked_suggestion['restaurantname']} is {self.picked_suggestion['postcode']}. Do you need other information or exit?")
                else:
                    self._print("Would you like more information about the suggested restaurant (phone, address or postcode)?")
            case "give_suggestion", False:
                if self.additional is not None:
                    if not self.available_suggestions:
                        self._print("I am sorry, I did not understand that. Unfortunately there are no other restaurants matching your criteria. Do you want to start over or exit?")
                    
                    else:
                        picked_restaurant, reason = self.preference_statement.characteristic_of_restaurant(
                        self.available_suggestions, self.additional
                        )
                        if picked_restaurant is not None:
                            if isinstance(picked_restaurant, tuple):
                                picked_restaurant = picked_restaurant[0]
                            self.picked_suggestion = picked_restaurant
                            self._print(reason)
                elif len(self.available_suggestions) > 0:
                    suggestion = self.available_suggestions.pop(0)
                    self.picked_suggestion = suggestion
                    self._print(f"I have found a restaurant that matches your preferences. It is {suggestion['restaurantname']} food in the {self.area} area with a {self.pricerange} price range. Do you like it?")
                else:
                    self._print("I am sorry, I do not have any more suggestions that match your criteria. Do you want to start over or exit?")
            case "pick_suggested_or_restart", False:
                self._print("Unfortunately there are no other restaurants matching your criteria. Would you like to pick the suggested restaurant or start over?")
            case "end_conversation", False:
                self._print("Thank you for using the Cambridge restaurant system. Goodbye!")
            case "welcome", True:
                self._print("I am sorry, I did not understand that. How may I help you?")
            case "ask_area", True:
                self._print("I am sorry, I did not understand that. In which area would you like to eat?")
            case "ask_pricerange", True:
                self._print("I am sorry, I did not understand that. What price range are you looking for (cheap, moderate or expensive)?")
            case "ask_food", True:
                self._print("I am sorry, we do not have that kind of food. What kind of food would you like?")
            case "extra_requirements", True:
                self._print("I am sorry, I did not understand that. Do you have any additional requirements? For example do you want the restaurant to be romantic, family-friendly, or quick?")
            case "give_suggestion", True:
                picked_restaurant, reason = self.preference_statement.characteristic_of_restaurant(
                        self.available_suggestions, self.additional
                    )
                if picked_restaurant is not None:
                    if isinstance(picked_restaurant, tuple):
                        picked_restaurant = picked_restaurant[0]
                    self.picked_suggestion = picked_restaurant
                    self._print(reason)
                elif len(self.available_suggestions) > 0:
                    suggestion = self.available_suggestions.pop(0)
                    self.picked_suggestion = suggestion
                    self._print(f"I have found another restaurant that matches your preferences called {suggestion['restaurantname']}. It is {self.food} food in the {self.area} area with a {self.pricerange} price range. Do you like it?")
                else:
                    self._print("I am sorry, I did not understand that. Unfortunately there are no other restaurants matching your criteria. Do you want to start over or exit?")
            case "pick_suggested_or_restart", True:
                self._print("I am sorry, I did not understand that. Unfortunately there are no other restaurants matching your criteria. Would you like to pick the suggested restaurant?")
            case "end_conversation", True:
                self._print("I am sorry, I did not understand that. Thank you for using the Cambridge restaurant system. Goodbye!")

    def _print(self, msg: str)-> None:
        """
        This function handles the printing of the system outputs. It
        handles whether the message needs to be in all caps or not and
        if you need to wait one second to see the system output.

        The system output is also printed in yellow.

        @param msg (str): The message that needs to be printed.
        """
        if self.all_caps:
            msg = msg.upper()
        if self.system_delay:
            time.sleep(1)
        print(f"\033[93m{msg}\033[0m")
