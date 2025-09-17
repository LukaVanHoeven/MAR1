from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Transition:
    """
    Class for keeping track of a transition in the dialog management
    system.

    @params:
        - original_state (str): The state the system is currently in.
        - dialogue_act (str): The dialogue act that leeds to the transition.
        - condition (str): Condition that needs to be met before the system
        can make the transition. If no condition is given, this function will
        always return True
        - next_state (str): The state the system moves to.
    """
    original_state: str
    dialogue_act: str
    next_state: str
    condition: Callable[[], bool] = field(default=lambda: True)