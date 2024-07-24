from graph.state import AppState
from graph.chains import PromptFinalizerChain, PromptGraderChain, PromptGeneratorChain


def generate_prompt(state: AppState) -> AppState:
    user_input = state["user_input"] 
    model_name = state["model_name"]
    chain = PromptGeneratorChain(model_name=model_name)
    prompt = chain.invoke({"user_input": user_input})
    state["initial_prompt"] = prompt
    return state

def grade_prompt(state: AppState) -> AppState:
    initial_prompt = state["initial_prompt"]
    model_name = state["model_name"]
    chain = PromptGraderChain(model_name=model_name)
    evaluations = chain.invoke({"initial_prompt": initial_prompt})
    state["evaluations"] = evaluations
    return state

def finalize_prompt(state: AppState) -> AppState:
    evaluations = state["evaluations"]
    initial_prompt = state["initial_prompt"]
    model_name = state["model_name"]
    chain = PromptFinalizerChain(model_name=model_name)
    final_prompt = chain.invoke({"initial_prompt": initial_prompt, "evaluations": evaluations})
    state["final_prompt"] = final_prompt
    return state