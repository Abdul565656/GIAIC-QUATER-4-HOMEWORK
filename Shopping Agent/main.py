from dotenv import load_dotenv
import os
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from agents import Agent, Runner, function_tool
import requests
import rich

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
print(gemini_api_key)
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@function_tool
def get_products():
    url = "https://template-03-api.vercel.app/api/products"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        products = response.json()   # Parse JSON response into Python dictionary/list
        return products
    except requests.RequestException as e:
        print("Error fetching products:", e)
        return None
    
agent = Agent(
    name="Shopping Agent",
    instructions="You are a shopping agent you have to give the info about the products also tell user recomended best prooduct",
    tools=[get_products]
)

result = Runner.run_sync(
    agent,
    input("Which product you want to get input.. "),
    run_config=config
)
rich.print(result.final_output)