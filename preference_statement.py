import pandas as pd
import Levenshtein

class PreferenceStatement:
    def __init__(self):
        self.valid_food_types = ('british', 'modern european', 'italian', 'romanian', 'seafood', 'chinese', 'steakhouse', 'asian oriental', 'french', 'portuguese', 'indian', 'spanish', 'european', 'vietnamese', 'korean', 'thai', 'moroccan', 'swiss', 'fusion', 'gastropub', 'tuscan', 'international', 'traditional', 'mediterranean', 'polynesian', 'african', 'turkish', 'bistro', 'north american', 'australasian', 'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan')
        self.valid_area_types = ("centre", "south", "north", "east", "west")
        self.valid_price_range_types = ("cheap", "moderate", "expensive")
        self.category_words = {
            "area": ["area", "part", "region", "side"],
            "food": ["food", "cuisine"],
            "pricerange": ["price", "pricerange", "pricing"]
        }

        self.valid_words = *self.valid_food_types, *self.valid_area_types, *self.valid_price_range_types
        self.threshold_distance = 2

    def parse_preference_statement(self, input):
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

            distances = [Levenshtein.distance(word, w) for w in self.valid_words]
            min_distance = min(distances)
            
            #If the distance is smaller than some threshold, we consider it a valid word and use the corrected version
            if min_distance < self.threshold_distance:
                smallest_index = distances.index(min_distance)
                word_to_use = self.valid_words[smallest_index]
            
            if word_to_use == "any":
                word_after_any = input.split(" ")[index + 1]
                for category, words in self.category_words.items():
                    if word_after_any in words:
                        result[category] = "any"
                        break

            if word_to_use in self.valid_area_types:
                result["area"] = word_to_use
            if word_to_use in self.valid_food_types:
                result["food"] = word_to_use
            if word_to_use in self.valid_price_range_types:
                result["pricerange"] = word_to_use
                
        return result
