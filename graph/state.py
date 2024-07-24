from typing import TypedDict

class AppState(TypedDict):
    user_input: str
    initial_prompt: str
    evaluations: str
    final_prompt: str
    model_name: str