# Mar part 1

Names: Bas de Blok, Mykola Chuprynskyy, Melissa Rueca, Luka van Hoeven

## How to use:

1. Install the requirements.txt
2. Run the program
   - Running `python main.py` in your terminal will launch the base application.
      - This will show you an option menu where you can type in 2 different options before pressing enter:
         - `1`: Train and evaluate the text classification models of assignment 1a.
         - `2`: Use the dialogue management system for restaurant recommendation from assignment 1b and 1c.

### Configurability
The dialogue management system has a couple of configurable options that can be used through adding the following arguments to the command line behind `python main.py`:
- `--train` -> Trains the Machine Learning model that's used by the system, this is necessary if you're running for the first time.
- `--allow_preference_change` -> Allows the user to adjust their preferences.
- `--all_caps` -> Output the response of the system in all caps.
- `--system_delay` -> Adds a 1 second delay before each system response.
- `--use_baseline` -> Use a baseline model instead of a Machine Learning model.

## Example run of dialogue management system

1. Run the system
2. Say hello
3. Put in an area or area, pricerange and type of food all at the same time
   - Inputting a sentence with "british" "moderate" and "centre" will give you 2 possible suggestions, this lets you test the logic for the system handling a rejection of the first choice.
4. Potentially get a suggestion
5. Reject the suggestion
6. Potentially get a new suggestion

## File descriptions

main.py: Allows the user to choose between training and evaluating classification models and using the dialogue management system.

antecedent.py: Adds random antecedents to `data/restaurant_info.csv`

dialogue_management_system.py: Contains the code for the dialogue management system, including the user interaction loop, inferring the models and calling the code for extracting preferences.

evaluate.py: Contains the evaluation functions for evaluating the tekst classification models.

logistic_regression.py: Contains the functions for training and inferring a logistic regression model.

majority_class.py: Contains the inferrence function for the majority class baseline classifier.

parse.py: Parses the `data/dialog_acts.dat` file into `data/test_dedup.csv`, `data/test_orig.csv`, `data/train_dedup.csv` and `data/train_orig.csv`.

preference_handler.py: Contains a class for both the extraction of preferences from a sentence as well as getting the restaurants from the restaurant_info.csv file.

restaurant_recommendation.py: Parses the command line argument and uses them in a dialogue management system. 

rulebased.py: Contains the inferrence function for the rule based baseline classifier.

sequential.py: Contains the functions for training and inferring a sequential model.

text_classification.py: Implements the training and evaluating of the text classification models. 

transition.py: Contains a class regarding the transition logic for the dialogue management system.
