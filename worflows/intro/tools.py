# %%
import json
import os
import requests

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel,Field

load_dotenv()

endpoint = os.getenv("ENDPOINT_URL")
deployment_name = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=f"{api_key}",base_url=f"{endpoint}")

def get_weather(latitude, longitude):
    """This is a public api to get the weather of location"""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
    data = response.json()
    return data["current"]


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

completion = client.chat.completions.create(
    model=deployment_name,
    messages=messages,
    tools=tools,
)

message = completion.choices[0].message
messages.append(message.model_dump())

#print(completion.model_dump_json(indent=2))


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    
for tool_call in message.tool_calls:
    function_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = call_function(function_name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )


class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A response to the user's question."
    )

completion_2 = client.chat.completions.parse(
    model=deployment_name,
    messages=messages,
    tools=tools,
    response_format=WeatherResponse
)

final_response = completion_2.choices[0].message.parsed
#print(final_response)
final_response.temperature
final_response.response   



# %%
