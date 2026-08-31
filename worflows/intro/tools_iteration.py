# %%
import json
import os
import requests

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

endpoint = os.getenv("ENDPOINT_URL")
deployment_name = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=f"{api_key}", base_url=f"{endpoint}")


def get_weather(latitude, longitude):
    """This is a public api to get the weather of location"""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
    data = response.json()
    return data["current"]


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful weather assistant."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Basingstoke today?"},
]


class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A response to the user's question."
    )


# ReAct loop: Reason (model decides) -> Act (call tool) -> Observe (feed result back) -> repeat
# until the model stops requesting tools, capped by max_iterations as a safety net.
max_iterations = 5

for i in range(max_iterations):
    completion = client.chat.completions.parse(
        model=deployment_name,
        messages=messages,
        tools=tools,
        response_format=WeatherResponse,
    )

    message = completion.choices[0].message
    messages.append(message.model_dump())

    if not message.tool_calls:
        # Model returned its final structured answer -- exit the loop
        final_response = message.parsed
        break

    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        result = call_function(function_name, args)
        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
        )
else:
    raise RuntimeError(f"Exceeded max_iterations={max_iterations} without a final answer")

print(final_response.temperature)
print(final_response.response)

# %%
