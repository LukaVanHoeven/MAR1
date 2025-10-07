import asyncio
from .dialogue_management_system import Dialogue_management_system
class Dialogue_Management_Server(Dialogue_management_system):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Initialize the websocket server here
        self.websocket = None
        return
    
    def set_websocket(self, websocket):
        self.websocket = websocket
        self._print(
            "Hello, welcome to the Cambridge restaurant system! You can " \
            "ask for restaurants by area, price range or food type. How " \
            "may I help you?"
        )
        return
    
    def reset(self):
        self.current_state = "welcome"
        self.pricerange = None
        self.area = None
        self.food = None
        self.additional = None
        self.available_suggestions = []
        self.gathered_suggestions = False
        self.picked_suggestion = None
        self.requested_additional_info = []
        return
    
    def utter(self, utterance)-> str:
        """
        This function contains the dialogue loop, allowing the user to
        interact with the system.
        """
        user = utterance
        if user == "exit" or user == "end_conversation":
            self._print("Thank you for using the restaurant system. Goodbye!")
            return "end_conversation"
        elif user in ["restart", "start", "start over"]:
            self.current_state = "welcome"
            self.pricerange = None
            self.area = None
            self.food = None
            self.additional = None
            self.available_suggestions = []
            self.gathered_suggestions = False
            self.picked_suggestion = None
            self.requested_additional_info = []
            self.print_next_conversation_step(repeat=False)

        if self.current_state == "welcome":
            self.pricerange = None
            self.area = None
            self.food = None
            self.additional = None
            self.available_suggestions = []
            self.gathered_suggestions = False
        previous_state = self.current_state

        next_state = self.state_transition(user)

        self.print_next_conversation_step(previous_state == next_state)
        if next_state == "end_conversation":
            return "end_conversation"
        
        
        return "continue"
    
    def _print(self, msg):
        if self.websocket:
            asyncio.create_task(self.send_message(msg))
        return
    
    async def send_message(self, msg):
        if self.websocket:
            await self.websocket.send(msg)
            await self.websocket.send("[END]")
        return