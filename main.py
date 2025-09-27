import sys
from src.text_classification import assignment1a
from src.restaurant_recommendation import recommendation

if __name__ == "__main__":
    import os
    print(os.path.exists('models/sequential_orig.keras'))
    no_choice_made = True
    while no_choice_made:
        print("Type the number corresponding to your preferred interaction " \
            "or type 'exit' to close the programme:"
        )
        print("(1) Train and evaluate the text classification models")
        print("(2) Use the dialogue management system for restaurant " \
            "recommendation"
        )

        choice = input("")

        if choice == "exit":
            print("Closing the programme!")
            sys.exit()
        try:
            if int(choice) not in range(1, 3):
                print("Invalid choice, try again\n\n")
                continue
            else:
                no_choice_made = False
        except ValueError:
            print("Invalid choice, try again\n\n")
            continue
    
    if choice == "1":
        assignment1a()
    elif choice == "2":
        recommendation()