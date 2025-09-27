import pandas as pd
import Levenshtein
from pathlib import Path


class PreferenceHandler:
    def __init__(self):
        self.valid_food_types = ('british', 'modern european', 'italian', 'romanian', 'seafood', 'chinese', 'steakhouse', 'asian oriental', 'french', 'portuguese', 'indian', 'spanish', 'european', 'vietnamese', 'korean', 'thai', 'moroccan', 'swiss', 'fusion', 'gastropub', 'tuscan', 'international', 'traditional', 'mediterranean', 'polynesian', 'african', 'turkish', 'bistro', 'north american', 'australasian', 'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan')
        self.valid_area_types = ("centre", "south", "north", "east", "west")
        self.valid_price_range_types = ("cheap", "moderate", "expensive")
        self.category_words = {
            "area": ["area", "part", "region", "side"],
            "food": ["food", "cuisine"],
            "pricerange": ["price", "pricerange", "pricing"]
        }
        
        self.possible_extra_requirements = {
            "romantic": {
                "food_quality": 1,
                "crowdedness": 0,
                "length_of_stay": 1
                },
            "family-friendly": {
                "crowdedness": 1,
                "length_of_stay": 0
            },
            "touristic": {
                "food_quality": 1,
                "crowdedness": 0
            },
            "quick meal": {
                "length_of_stay": 0,
                "crowdedness": 0
            },
            "business": {
                "food_quality": 1,
                "length_of_stay": 1
            },
            "trashy": {
                "food_quality": 0,
                "crowdedness": 1,
                "length_of_stay": 0
            }
        }
        self.valid_extra_preferences = list(self.possible_extra_requirements.keys())

        self.valid_words = *self.valid_food_types, *self.valid_area_types, *self.valid_price_range_types, *self.valid_extra_preferences, "any"
        self.threshold_distance = 3
        data_folder = Path(__file__).resolve().parent.parent / "data"
        self.data = pd.read_csv(data_folder / "restaurant_info.csv")

    def find_matching_restaurants(self, area, food, pricerange) -> list:
        """
        This function finds restaurants that match the given preferences for area, food, and price range.
        """
        matching_restaurants = self.data
        if area:
            if area != "any":
                matching_restaurants = matching_restaurants[matching_restaurants['area'] == area]
        if food:
            if food != "any":
                matching_restaurants = matching_restaurants[matching_restaurants['food'] == food]
        if pricerange:
            if pricerange != "any":
                matching_restaurants = matching_restaurants[matching_restaurants['pricerange'] == pricerange]
        return matching_restaurants['restaurantname'].tolist()

    def return_matching(self, restaurant, requirements):
        #This function returns a dictionary which contains true if a requirement is met, false otherwise
        return {req[0]: restaurant[req[0]] == req[1] for req in requirements.items()}

    def characteristic_of_restaurant(self, restaurants: list, user_requirement):
        """
        This function takes a list of restaurants and a user requirement (e.g., "romantic", "family-friendly") and returns the restaurants that match the requirement and reason.
        """
        # We will make a new list of restaurants with their respective points for each fulfilled requirement
        restaurants = [(r, self.return_matching(r, self.possible_extra_requirements[user_requirement])) for r in restaurants]

        # If no restaurants match any requirement, return a message
        if all(sum(x[1].values()) == 0 for x in restaurants):
            return "", "I am sorry, but I could not find any restaurants that match your requirement."

        # Pick the restaurant with the highest number of matched requirements
        picked_restaurant = max(restaurants, key=lambda x: sum(x[1].values()))
        picked_match_parameters = picked_restaurant[1]
        
        max_points = len(self.possible_extra_requirements[user_requirement])
        suffices_all_requirements = sum(picked_match_parameters.values()) == max_points
        
        reason = ""
        
        if suffices_all_requirements:
            reason = f"I found a restaurant which is very {user_requirement},  "
        else:
            reason = f"I found a restaurant which is somewhat {user_requirement}, "

        if "food_quality" in picked_match_parameters:
            reason += f"the food is {'good' if self.possible_extra_requirements[user_requirement]['food_quality']== 1 else 'average'} and "

        if "crowdedness" in picked_match_parameters:
            reason += f"it is {'not crowded' if self.possible_extra_requirements[user_requirement]['crowdedness']== 0 else 'somewhat crowded'} and "

        if "length_of_stay" in picked_match_parameters:
            reason += f"the length of stay is {'appropriate' if self.possible_extra_requirements[user_requirement]['length_of_stay']== 1 else 'long'}. "

        if suffices_all_requirements:
            reason += f"Overall, it is a great choice for a {user_requirement} restaurant. " 
        else:
            positive_aspects = [k.replace('_', ' ') for k, v in picked_match_parameters.items() if v]
            negative_aspects = [k.replace('_', ' ') for k, v in picked_match_parameters.items() if not v]
            reason += f"It is {user_requirement} because of the {', '.join(positive_aspects)}, but it is not because of the {', '.join(negative_aspects)}. "
            
        return picked_restaurant, reason

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
            "pricerange": None,
            "additional": None
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
            if word_to_use in self.valid_extra_preferences:
                result["additional"] = word_to_use

        return result

# #Example usage
# pref = PreferenceHandler()
# #Test the characteristics function
# restaurants = [
#     {
#         "restaurantname": "The Gourmet Kitchen",
#         "food_quality": 0,
#         "crowdedness": 1,
#         "length_of_stay": 1
#     },
#     {
#         "restaurantname": "Family Diner",
#         "food_quality": 0,
#         "crowdedness": 1,
#         "length_of_stay": 1
#     },
#     {
#         "restaurantname": "Tourist's Delight",
#         "food_quality": 0,
#         "crowdedness": 1,
#         "length_of_stay": 1
#     }
# ]
# print(pref.characteristic_of_restaurant(restaurants, "trashy"))