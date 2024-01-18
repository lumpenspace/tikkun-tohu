from src.openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from os import environ as os_env
load_dotenv()

oai_client = OpenAI(api_key=os_env.get("OPENAI_API_KEY"))

