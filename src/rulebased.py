#Dialog act rules dictionary
rules = {
    "thankyou" : ["thank you", "thanks", "thank"],
    "bye" : ["thats all", "goodbye", "good bye", "bye", "see you"],
    "hello" : ["hi", "hello", "welcome", "halo"],
    "restart" : ["reset", "start over", "start again"],
    "request" : ["can i", "what is", "whats", "do you", "could i", "could you", "can you", "where", "tell me", "phone number", "address", "post code", "postcode", "what kind"],
    "repeat" : ["repeat", "back", "again"],
    "reqalts" : ["how about", "what about", "something else", "any other", "other", "another", "is there any", "different", "are there any", "anything else"],
    "reqmore" : ["more"],
    "confirm" : ["is that", "does it", "stop it", "is there", "is it", "do they", "is this"],
    "deny" : ["i dont", "dont want", "no not"],
    "negate" : ["no", "wrong"],
    "ack" : ["okay", "kay", "fine"],
    "affirm" : ["yes", "correct", "perfect", "right", "yeah", "yea", "ye"],
    "null" : ["cough"]
}

def rulebased(text: str)-> str:
    """
    Function to classify an utterance with a dialog act using the rules
    dictionary.

    @param text (str): The user utterance.

    @return (str): The predicted label
    """
    t = text.split()
    for dialog_act, words in rules.items():
        for word in words:
            if " " in word:
                if word in text:
                    return dialog_act
            else:
                if word in t:
                   return dialog_act
    return "inform"
