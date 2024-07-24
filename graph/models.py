from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

MODELS_MAP = {
    "gpt-4o": {
        "params": {
            "temperature": 0,
        },
    },
    "gpt-4-turbo": {
        "params": {
            "temperature": 0,
        },
    },
    "gpt-4": {
        "params": {
            "temperature": 0,
        },
    },
    "gpt-3.5-turbo": {
        "params": {
            "temperature": 0,
        },
    },
    "gemma-7b-it": {
        "params": {
            "temperature": 0,
        },
    },
    "gemma2-9b-it": {
        "params": {
            "temperature": 0,
        },
    },
    "llama3-70b-8192": {
        "params": {
            "temperature": 0,
        },
    },
    "llama3-8b-8192": {
        "params": {
            "temperature": 0,
        },
    },
    "mixtral-8x7b-32768": {
        "params": {
            "temperature": 0,
        },
    }
}


def load_model(model_name):
    if model_name not in MODELS_MAP:
        raise ValueError(f"Model {model_name} not found")
    model_params = MODELS_MAP[model_name]["params"]
    if model_name.startswith("gpt"):
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(model=model_name, api_key=api_key, **model_params)
    else:
        api_key = os.getenv("GROQ_API_KEY")
        return ChatGroq(model=model_name, api_key=api_key, **model_params)
