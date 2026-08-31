import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


endpoint = os.getenv("ENDPOINT_URL")
deployment_name = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=f"{api_key}",base_url=f"{endpoint}")

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {"role": "system", "content": "You're a helpful assistant."},
        {"role": "user", "content": "Write a limerick about the Python programming language"},
    ]
)

response = completion.choices[0].message.content
print(response)

# %%
    