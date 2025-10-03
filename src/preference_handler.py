import pandas as pd
import Levenshtein
from pathlib import Path
from typing import TypedDict


class PreferenceHandler:
    def __init__(self):
        """
        The constructor for the PreferenceHandles class.
        Initializes valid food types, area types, price range types, 
        category words, possible extra requirements, valid extra 
        preferences, valid words, threshold distance,  and loads 
        restaurant data from a CSV file.
        """
        self.valid_food_types = (
            'british',
            'modern european',
            'italian',
            'romanian',
            'seafood',
            'chinese',
            'steakhouse',
            'asian oriental',
            'french',
            'portuguese',
            'indian',
            'spanish',
            'european',
            'vietnamese',
            'korean',
            'thai',
            'moroccan',
            'swiss',
            'fusion',
            'gastropub',
            'tuscan',
            'international',
            'traditional',
            'mediterranean',
            'polynesian',
            'african',
            'turkish',
            'bistro',
            'north american',
            'australasian',
            'persian',
            'jamaican',
            'lebanese',
            'cuban',
            'japanese',
            'catalan'
        )
        self.valid_area_types = ("centre", "south", "north", "east", "west")
        self.valid_price_range_types = ("cheap", "moderate", "expensive")
        self.category_words = {
            "area": ["area", "part", "region", "side"],
            "food": ["food", "cuisine"],
            "pricerange": ["price", "pricerange", "pricing"]
        }
        
        self.info_types = {
            "phone": ["phone", "number", "phonenumber", "phone number"],
            "address": ["address", "location", "place"],
            "postcode": ["postcode", "zip", "zip code", "postal code"]
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

        self.valid_words = \
            *self.valid_food_types, \
            *self.valid_area_types, \
            *self.valid_price_range_types, \
            *self.valid_extra_preferences, \
            "any"
        self.threshold_distance = 3
        data_folder = Path(__file__).resolve().parent.parent / "data"
        self.data = pd.read_csv(data_folder / "restaurant_info.csv")

    def find_matching_restaurants(
        self,
        area: str,
        food: str,
        pricerange: str
    ) -> list:
        """
        This function finds restaurants that match the given preferences
        for area, food, and price range. All params should be words that
        are in the valid lists defined in the constructor. If a parameter 
        is "any" it means that the user has no preference for that 
        parameter.

        @param area (str): The preferred area of the restaurant.
        @param food (str): The preferred type of food.
        @param pricerange (str): The preferred price range of the 
            restaurant.
        
        @return (list): A list of dictionaries representing the matching 
            restaurants.
        """
        matching_restaurants = self.data
        if area:
            if area != "any":
                matching_restaurants = matching_restaurants[
                    matching_restaurants['area'] == area
                ]
        if food:
            if food != "any":
                matching_restaurants = matching_restaurants[
                    matching_restaurants['food'] == food
                ]
        if pricerange:
            if pricerange != "any":
                matching_restaurants = matching_restaurants[
                    matching_restaurants['pricerange'] == pricerange
                ]
        #return it as a list of dictionaries
        matching_restaurants = matching_restaurants.to_dict(orient='records')
        return matching_restaurants

    def return_matching(self, restaurant: dict, requirements: dict) -> dict:
        """
        This function checks if a restaurant meets the given requirements.

        @param restaurant (dict): A dictionary representing a restaurant
            with its attributes.
        @param requirements (dict): A dictionary of requirements to
            check against the restaurant's attributes.
        
        @return (dict): A dictionary indicating whether each requirement
            is met (True) or not (False).
        """        
        return {
            req[0]: restaurant[req[0]] == req[1] for req in requirements.items()
        }

    def characteristic_of_restaurant(
        self,
        restaurants: list,
        user_requirement: str
    )-> tuple[dict, str]:
        """
        This function takes a list of restaurants and a user requirement
        (e.g., "romantic", "family-friendly") and returns the restaurants
        that match the requirement and reason.
        
        @param restaurants (list): A list of restaurant dictionaries.
        @param user_requirement (str): The user requirement to match
            (e.g., "romantic", "family-friendly").
        
        @return (tuple[dict, str]): Tuple containing:
            - The restaurant that best matches the user requirement.
            - A string explaining why the restaurant was chosen based on
                the user's requirement.
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
            reason = f"I found a restaurant which is very {user_requirement}, it is called {picked_restaurant[0]['restaurantname']}. "
        else:
            reason = f"I found a restaurant which is somewhat {user_requirement}, it is called {picked_restaurant[0]['restaurantname']}. "

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

    def parse_preference_statement(self, input) -> dict[str, str]:
        """
        This function parses the user input in the form of a string, 
        and returns a dictionary with the user's preferences for food, 
        area, and price range.

        @param input (str): The user input string.

        Returns:
            dict: A dictionary with the user's preferences for food, 
                area, and price range.
        """
        result = {
            "food": None,
            "area": None,
            "pricerange": None,
            "additional": None
        }
        input = input.lower()

        for index, word in enumerate(input.split(" ")):
            # Levenshtein distance to the closest valid word for every valid word
            word_to_use = word.lower()

            distances = [Levenshtein.distance(word, w) for w in self.valid_words]
            min_distance = min(distances)

            # If the distance is smaller than some threshold, we consider it a 
            # valid word and use the corrected version
            if min_distance < self.threshold_distance:
                smallest_index = distances.index(min_distance)
                word_to_use = self.valid_words[smallest_index]

            if word_to_use == "any":
                word_after_any = input.split(" ")
                word_after_any = word_after_any[
                    index + 1
                ] if index + 1 < len(word_after_any) else ""
                if word_after_any == "":
                    continue
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

    def parse_info_request(self, input) -> list:
        """
        This function parses the user input in the form of a string, 
        and returns a list with the types of information requested by 
        the user.

        @param input (str): The user input string.

        Returns:
            list: A list with the types of information requested by the user.
        """
        result = []
        input = input.lower()
        
        for key, value in self.info_types.items():
            for word in value:
                if word in input:
                    result.append(key)
                    break
        return result