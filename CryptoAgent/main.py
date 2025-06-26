import os
import asyncio
import aiohttp
from flask import Flask, request, render_template
from dotenv import load_dotenv

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
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
async def get_crypto_price(symbol: str) -> str:
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                price = data.get("price")
                return f"The current price of {symbol.upper()} is ${price}"
            else:
                return f"Could not fetch price for {symbol.upper()}"

agent = Agent(
    name="Crypto Agent",
    instructions="You are a crypto assistant. Fetch real-time coin prices using Binance.",
    model=model,
    tools=[get_crypto_price],
)

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_input = request.form["query"]
        result = asyncio.run(Runner.run(agent, input=user_input, run_config=config))
        response = result.final_output
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
