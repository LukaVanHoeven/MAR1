import argparse
from .transition import Transition
from .sequential import train_sequential, sequential
from .dialogue_management_system import Dialogue_management_system
import pandas as pd


def recommendation()-> None:
    """
    Runs the restaurant recommendation dialogue management system. The 
    diagram this was based on is found in the `diagrams` folder.

    First we parse all configurable options that were given in the 
    command line argument, then we define all possible transitions in
    the system, lastly we build the dialogue management system.
    """
    parser = argparse.ArgumentParser(description="Run the Dialogue Management System.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model before running the dialogue system."
    )
    parser.add_argument(
        "--allow_preference_change",
        action="store_true",
        default=False,
        help="Allow the user to adjust their preferences (default: False)."
    )
    parser.add_argument(
        "--all_caps",
        action="store_true",
        default=False,
        help="Output the response of the system in all caps (default: False)."
    )
    parser.add_argument(
        "--system_delay",
        action="store_true",
        default=False,
        help="Add a 1 second delay before each system response (default: False)."
    )
    parser.add_argument(
        "--use_baseline",
        action="store_true",
        default=False,
        help="Use a baseline model instead of a Machine Learning model."
    )
    args = parser.parse_args()

    transitions = [
        Transition(original_state="welcome", dialogue_act=["hello"], next_state="ask_area"),
        Transition(original_state="welcome", dialogue_act=["inform", "null"], condition=lambda d: d.area is not None and d.pricerange is not None and d.food is not None, next_state="extra_requirements"),
        Transition(original_state="welcome", dialogue_act=["inform", "null"], condition=lambda d: d.area is not None and d.pricerange is not None and d.food is None, next_state="ask_food"),
        Transition(original_state="welcome", dialogue_act=["inform", "null"], condition=lambda d: d.area is not None and d.pricerange is None, next_state="ask_pricerange"),
        Transition(original_state="welcome", dialogue_act=["inform", "null"], condition=lambda d: d.area is None, next_state="ask_area"),
        
        Transition(original_state="ask_area", dialogue_act=["inform", "null"], condition=lambda d: d.area is None, next_state="ask_area"),
        Transition(original_state="ask_area", dialogue_act=["inform", "null"], condition=lambda d: d.pricerange is not None and d.food is not None and d.area is not None, next_state="extra_requirements"),
        Transition(original_state="ask_area", dialogue_act=["inform", "null"], condition=lambda d: d.pricerange is not None and d.food is None, next_state="ask_food"),
        Transition(original_state="ask_area", dialogue_act=["inform", "null"], condition=lambda d: d.pricerange is None, next_state="ask_pricerange"),
        
        Transition(original_state="ask_pricerange", dialogue_act=["inform", "null"], condition=lambda d:d.pricerange is None, next_state="ask_pricerange"),
        Transition(original_state="ask_pricerange", dialogue_act=["inform", "null"], condition=lambda d: d.food is not None and d.pricerange is not None and d.area is not None, next_state="extra_requirements"),
        Transition(original_state="ask_pricerange", dialogue_act=["inform", "null"], condition=lambda d: d.food is None, next_state="ask_food"),
        Transition(original_state="ask_pricerange", dialogue_act=["inform", "null"], condition=lambda d: d.area is None, next_state="ask_area"),
        
        Transition(original_state="ask_food", dialogue_act=["inform", "null"], condition=lambda d: d.food is None, next_state="ask_food"),
        Transition(original_state="ask_food", dialogue_act=["inform", "null"], condition=lambda d: d.food is not None and d.pricerange is not None and d.area is not None, next_state="extra_requirements"),
        Transition(original_state="ask_food", dialogue_act=["inform", "null"], condition=lambda d: d.pricerange is None, next_state="ask_pricerange"),
        Transition(original_state="ask_food", dialogue_act=["inform", "null"], condition=lambda d: d.area is None, next_state="ask_area"),

        # For the repeat, null, bye, reqmore, reqalts, request dialogue acts we stay in the same state
        Transition(original_state="welcome", dialogue_act=["repeat", "null", "bye", "reqmore", "reqalts", "request"], next_state="welcome"),
        Transition(original_state="ask_area", dialogue_act=["repeat", "null", "bye", "reqmore", "reqalts", "request"], next_state="ask_area"),
        Transition(original_state="ask_pricerange", dialogue_act=["repeat", "null", "bye", "reqmore", "reqalts", "request"], next_state="ask_pricerange"),
        Transition(original_state="give_suggestion", dialogue_act=["repeat", "null", "bye", "reqmore", "reqalts", "request"], next_state="extra_requirements"),

        Transition(original_state="extra_requirements", dialogue_act=["inform", "null", "negate"], next_state="give_suggestion"),
        Transition(original_state="extra_requirements", dialogue_act=["null", "repeat", "request"], next_state="extra_requirements"),

        # If the user agrees on the suggestion it should move to asking if the user wants more info
        Transition(original_state="give_suggestion", dialogue_act=["ack", "affirm", "thankyou"], next_state="ask_additional_info"),
        
        # If the user wants more info about the suggestion we should give it
        Transition(original_state="ask_additional_info", dialogue_act=["inform", "null"], next_state="ask_additional_info"),
        Transition(original_state="ask_additional_info", dialogue_act=["null", "repeat", "request"], next_state="ask_additional_info"),
        Transition(original_state="ask_additional_info", dialogue_act=["deny", "negate", "thankyou", "bye"], next_state="end_conversation"),

        # If the user does not like the suggestions and there are other suggestions the user should get a new suggestion
        Transition(original_state="give_suggestion", dialogue_act=["deny", "negate", "reqalts"], condition=lambda d:len(d.available_suggestions) > 0 and d.additional is None, next_state="give_suggestion"),

        # If the user does not like the suggestion the user can pick the last suggested suggestion or end the conversation
        Transition(original_state="give_suggestion", dialogue_act=["deny", "negate", "reqalts"], condition=lambda d:len(d.available_suggestions) == 0 or d.additional is not None, next_state="pick_suggested_or_restart"),

        # If the user takes the last offered suggestion it ends the conversation
        Transition(original_state="pick_suggestion_or_restart", dialogue_act=["repeat", "null"], next_state="pick_suggestion_or_restart"),
        Transition(original_state="pick_suggested_or_restart", dialogue_act=["ack", "affirm", "thankyou", "bye"], next_state="ask_additional_info"),
        
        # If the user does not like the last offered suggestion it goes back to the welcome state
        Transition(original_state="pick_suggested_or_restart", dialogue_act=["deny", "negate", "restart"], next_state="welcome"),
    ]
    
    #Train the sequential model so we have something to use (Mainly for being able to just run 1 file for handing in the assignment)
    if args.train:
        orig_train = pd.read_csv("train_orig.csv")
        train_sequential("sequential_orig", orig_train)
    
    dms = Dialogue_management_system(
        sequential,
        transitions,
        "welcome",
        args.allow_preference_change,
        args.all_caps,
        args.system_delay
    )
    dms.loop()
    