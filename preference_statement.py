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
        self.threshold_distance = 3
        self.data = pd.read_csv("restaurant_info.csv")
        
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
    
    def characteristic_of_restaurant(self, restaurants: list, user_requirement):
        """
        This function takes a list of restaurants and a user requirement (e.g., "romantic", "family-friendly") and returns the restaurants that match the requirement and reason.
        """ 
        ## Note to self on how to improve: make a list of reasons why a restaurant is romantic, family-friendly, touristy, etc. and use that to match the restaurants instead of hardcoding it here. Also more useful for the "reason" part of the return value.
        original_restaurants = restaurants.copy()
        reason = ""
        match user_requirement:
            case "romantic":
                restaurants = [r for r in restaurants if r.length_of_stay == 1 and r.crowdedness == 0]
            case "family-friendly":
                restaurants = [r for r in restaurants if r.crowdedness == 1 and r.length_of_stay == 0]        
            case "tourist": 
                restaurants = [r for r in restaurants if r.food_quality == 1 and r.crowdedness == 0]
                
        if len(restaurants) == 0:
            match user_requirement:
                case "romantic":
                    restaurants = [r for r in original_restaurants if r.length_of_stay == 1 or r.crowdedness == 0]
                case "family-friendly":
                    restaurants = [r for r in original_restaurants if r.crowdedness == 1 or r.length_of_stay == 0]
                case "tourist": 
                    restaurants = [r for r in original_restaurants if r.food_quality == 1 or r.crowdedness == 0]
            if len(restaurants) == 0:
                return "", "I am sorry, but I could not find any restaurants that match your requirement."
            else:
                picked_restaurant = restaurants[0]
                reason = f"I did not find a restaurant that is entirely {user_requirement}, but it partially suffices. {picked_restaurant.name} is {user_requirement} because it is {'a must-visit' if picked_restaurant.food_quality == 1 else 'not a must-visit'}, it is {'busy' if picked_restaurant.crowdedness == 1 else 'not busy'}, and the average length of stay is {'long' if picked_restaurant.length_of_stay == 1 else 'short'}."
        else:
            picked_restaurant = restaurants[0]
            reason = f"{picked_restaurant.name} is {user_requirement} because it is {'a must-visit' if picked_restaurant.food_quality == 1 else 'not a must-visit'}, it is {'busy' if picked_restaurant.crowdedness == 1 else 'not busy'}, and the average length of stay is {'long' if picked_restaurant.length_of_stay == 1 else 'short'}."
            
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
