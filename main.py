from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv, find_dotenv
from openai.types.responses import ResponseTextDeltaEvent
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

agent = Agent(
    name="Islamic Knowledge Expert",
    instructions="Only respond to questions related to Islamic beliefs, prayers, and etiquette.If the question is not related to Islamic Knowledge, respond with: Sorry, I only answer questions about Islamic Knowledge.",
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to the Islamic Knowledge Expert Assistant! How can I assist you today?").send()

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

