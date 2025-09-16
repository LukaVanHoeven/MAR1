import pandas as pd
import Levenshtein

file = pd.read_csv("restaurant_info.csv")
valid_food_types = file["food"].unique()
valid_area_types = ("centre", "south", "north", "east", "west")
valid_price_range_types = file["pricerange"].unique()

category_words = {
    "area": ["area", "part", "region", "side"],
    "food": ["food", "cuisine"],
    "pricerange": ["price", "pricerange", "pricing"]
}

valid_words = *valid_food_types, *valid_area_types, *valid_price_range_types
threshold_distance = 2
    
def parse_preference_statement(input):
    """This function parses the user input in the form of a string, and returns a dictionary with the user's preferences for food, area, and price range.

    Args:
        input (str): The user input string.

    Returns:
        dict: A dictionary with the user's preferences for food, area, and price range.
    """
    
    result = {
        "food": None,
        "area": None,
        "pricerange": None
    }
    input = input.lower()
    
    for index, word in enumerate(input.split(" ")):
        #Levenshtein distance to the closest valid word for every valid word
        word_to_use = word.lower()
        
        distances = [Levenshtein.distance(word, w) for w in valid_words]
        min_distance = min(distances)
        
        #If the distance is smaller than some threshold, we consider it a valid word and use the corrected version
        if min_distance < threshold_distance:
            smallest_index = distances.index(min_distance)
            word_to_use = valid_words[smallest_index]
        
        if word_to_use == "any":
            word_after_any = input.split(" ")[index + 1]
            for category, words in category_words.items():
                if word_after_any in words:
                    result[category] = "any"
                    break
        
        if word_to_use in valid_area_types:
            result["area"] = word_to_use
        if word_to_use in valid_food_types:
            result["food"] = word_to_use
        if word_to_use in valid_price_range_types:
            result["pricerange"] = word_to_use
            
    return result
