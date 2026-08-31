# %%
import json
import os


from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

endpoint = os.getenv("ENDPOINT_URL")
deployment_name = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=f"{api_key}",base_url=f"{endpoint}")

def search_kb(question: str):
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    with open("kb.json", "r") as f:
        return json.load(f)


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the temperature in tokyo?"},
]


OUT_OF_SCOPE_KEYWORDS = ["temperature", "weather", "forecast"]


def is_out_of_scope(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in OUT_OF_SCOPE_KEYWORDS)


def call_function(name, args):
    if name == "search_kb":
        # Enforce the "ignore temperature-related questions" rule in code, rather
        # than relying on the model to honor it from the tool description alone.
        if is_out_of_scope(args.get("question", "")):
            return {"error": "This knowledge base does not cover temperature/weather questions."}
        return search_kb(**args)


class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int | None = Field(
        default=None, description="The record id of the answer, or None if not found in the knowledge base."
    )

max_iterations = 5

for i in range(max_iterations):
    completion = client.chat.completions.parse(
        model=deployment_name,
        messages=messages,
        tools=tools,
        response_format=KBResponse,
    )

    message = completion.choices[0].message
    messages.append(message.model_dump(exclude={"parsed"}))

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

print(final_response.answer)
print(final_response.source)
# %%
