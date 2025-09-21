# Mar 1b

Names: Bas de Blok, Mykola Chuprynskyy, Melissa Rueca, Luka van Hoeven

## How to use:

Run dialogue_management_system.py --train to run the interactive system and train the model beforehand
Run dialogue_management_system.py to run the interactive system without training the model (if you have already trained the model beforehand)

## Example

1. Run the system
2. Say hello
3. Put in an area or area, pricerange and type of food all at the same time
   - Inputting a sentence with "british" "moderate" and "centre" will give you 2 possible suggestions, this lets you test the logic for the system handling a rejection of the first choice.
4. Potentially get a suggestion
5. Reject the suggestion
6. Potentially get a new suggestion

## File descriptions

dialogue_management_system.py: Contains the code for the dialogue management system, including the user interaction loop, inferring the models and calling the code for extracting preferences.

preference_statement.py: Contains a class for both the extraction of preferences from a sentence as well as getting the restaurants from the restaurant_info.csv file.

transition.py: Contains a class regarding the transition logic for the dialogue management system.

ML_sequential.py : contains a function for inferring a sequential model
train_sequential.py: contains a function for training the sequential model
