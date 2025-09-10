import pandas as pd

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

ds_train = pd.read_csv("train_dedup.csv")

pred = []

#Function to classify an utterance with a dialog act using the rules dictionary
def matching(text):
    t= text.split()
    for dialog_act, words in rules.items():
        for word in words:
            if " " in word:
                if word in text:
                    return dialog_act
            else:
                if word in t:
                   return dialog_act
    return "inform"

#Apply the rule-based classifier to each utterance 
for i, row in ds_train.iterrows():
    txt = row["text"]
    p = matching(txt)
    pred.append(p)

#New dataframe with the real dialog act and the predicted dialog act for each utterance
results = pd.DataFrame({
    "text": ds_train["text"],
    "train_dialogact": ds_train["label"],
    "predicted_dialogact": pred
})

results.to_csv("predictions_rulebased.csv", index=False)

#Compute accuracy
correctly_labeled_inst = 0
tot_inst = 0
for train, predicted in zip(results["train_dialogact"], results["predicted_dialogact"]):
    tot_inst += 1
    if train == predicted:
        correctly_labeled_inst += 1
accuracy = correctly_labeled_inst / tot_inst
print (accuracy)

#User prompt that returns the dialog act for the given utterance
user_input = ""
while user_input != "exit":
    user_input = input("Write your sentence> ").strip().lower()
    if user_input == "exit":
        break
    predicted_dialogact = matching(user_input)
    print(predicted_dialogact)