from preference_statement import parse_preference_statement
from ML_sequential import sequential

class DialogSystem:
    states = ("welcome", "ask_area", "ask_pricerange", "ask_food_type", "give_suggestion", "`pick_suggested_or_restart", "end_conversation")
    current_state = "welcome"
    known_user_preferences = {
        "area": None,
        "pricerange": None,
        "food_type": None
    }
        
    def handle_user_utterance(self, current_state, user_utterance):
        predicted_text_type = sequential([user_utterance], "sequential_orig.keras", "sequential_orig.pickle")[0]
        statement_parser = parse_preference_statement(user_utterance)
        
        next_state = ""
        
        if predicted_text_type == "inform":
            for key, value in statement_parser.items():
                if value is not None:  # only update if we got something useful
                    self.known_user_preferences[key] = value
                    
            match current_state:
                #These cases will result in the state diagram being maintained, because we know that some things are filled or a loop is implied
                case "welcome" | "ask_area" | "ask_pricerange" | "ask_food_type":
                    if self.known_user_preferences["area"] is None:
                        next_state = "ask_area"
                    elif self.known_user_preferences["food_type"] is None:
                        next_state = "ask_food_type"
                    elif self.known_user_preferences["pricerange"] is None:
                        next_state = "ask_pricerange"
        
        if predicted_text_type in ("null", "repeat", "hello"):
            next_state = current_state
        
        if current_state == "give_suggestion":
            if predicted_text_type in ("ack", "affirm", "thankyou"):
                next_state = "end_conversation"
            elif predicted_text_type in ("deny", "negate", "reqmore", "reqalts"):
                next_state = "pick_suggested_or_restart"
            else:
                next_state = current_state  # stay in the same state if the input is not recognized

        if current_state == "pick_suggested_or_restart":
            if predicted_text_type == "restart":
                self.known_user_preferences = {
                    "area": None,
                    "pricerange": None,
                    "food_type": None
                }
                next_state = "welcome"
            elif predicted_text_type in ("ack", "affirm", "thankyou"):
                next_state = "end_conversation"
            else:
                next_state = current_state  # stay in the same state if the input is not recognized
                
        return next_state

ds = DialogSystem()
print(ds.handle_user_utterance("welcome", "I want a cheap restaurant"))