from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv, find_dotenv
import requests
import random
import chainlit as cl

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

@function_tool
def how_many_jokes():
    return random.randint(1, 10)

@function_tool
def get_weather(city: str)-> str:
    """Get the current weather for a given city."""
    try:
        result = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key=546cbe7a4c654d228c6102334252602&q={city}"
        )
        data = result.json()
        return f"The current weather in {city} is {data['current']['temp_c']}°C with {data['current']['condition']['text']}."
    except Exception as e:
        return f"Could not retrieve weather data for {city}. Error: {str(e)}"
    
agent = Agent(
    name="GIAIC Agent",
    instructions="If the user asks for a joke, tell them a joke. If they ask for the weather, provide the current weather for the specified city. Otherwise, tell them that you are only give them a knowledge about weather of tell any joke.",
    tools=[how_many_jokes, get_weather],
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to the AI assistant! How can I assist you today?").send()

@cl.on_message
async def handle_massage(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({
        "role": "user",
        "content": message.content
    })
    result = Runner.run_streamed(
        agent, 
        input=history,
        run_config=run_config,
    )
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({
        "role": "assistant",
        "content": result.final_output
    })
    cl.user_session.set("history", history)

result = Runner.run_sync(
    agent, 
    "What is the weather in Karachi?",
    run_config=run_config,
)

print(result.final_output)