# %%
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
endpoint = os.getenv("ENDPOINT_URL")
deployment_name = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=f"{api_key}",base_url=f"{endpoint}")

class CalenderEvent(BaseModel):
    name:str
    date:str
    participants:list[str]

completion = client.chat.completions.parse(
    model=deployment_name,
    messages=[
        {"role": "system", "content": "Extract the event details , please put the exact event date , dont use tomorrow , today , yesterday instead find the date using the current date"},
        {"role": "user", "content": "Subhadeep and Debadrita are going to wach football match in Manchester tommorow"},
    ],
    response_format=CalenderEvent
)

response = completion.choices[0].message.parsed
print(response)


# %%
