import os
import environ

OPENAI_API_KEY = environ.Env().get("OPENAI_API_KEY")
REPLICATE_API_TOKEN = environ.Env().get("REPLICATE_API_TOKEN")

def setup_apis():
    import openai
    import replicate
    
    openai.api_key = OPENAI_API_KEY
    replicate.api_token = REPLICATE_API_TOKEN 