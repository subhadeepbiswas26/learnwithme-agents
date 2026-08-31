# %%
import json
import os
import requests
import logging
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

endpoint = os.getenv("ENDPOINT_URL")
deployment_name = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=f"{api_key}", base_url=f"{endpoint}")


class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""

    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


def extract_event_info(user_input: str) -> EventExtraction:
    """First LLM call to determine if input is a calendar event"""
    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.chat.completions.parse(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Analyze if the text describes a calendar event.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format=EventExtraction,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}"
    )
    return result

class EventDetails(BaseModel):
    """Second LLM call: Parse specific event details"""

    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")

def parse_event_details(description: str) -> EventDetails:
    """Second LLM call to extract specific event details"""
    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference. Only include participants who are explicitly named in the text. Do not invent, assume, or use example names (e.g. Alice, Bob, Charlie) — if no participants are mentioned, return an empty list.",
            },
            {"role": "user", "content": description},
        ],
        response_format=EventDetails,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        f"Parsed event details - Name: {result.name}, Date: {result.date}, Duration: {result.duration_minutes}min"
    )
    logger.debug(f"Participants: {', '.join(result.participants)}")
    return result

class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )
    participants: list[str] = Field(description="List of participants")

def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """Third LLM call to generate a confirmation message"""
    logger.info("Generating confirmation message")

    completion = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation message for the event. Sign off with your name; Susie. Use only the participants listed in the provided event details — do not add, invent, or assume any names. If the participants list is empty, don't mention any names.",
            },
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        response_format=EventConfirmation,
    )
    result = completion.choices[0].message.parsed
    logger.info("Confirmation message generated successfully")
    return result

def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Main function implementing the prompt chain with gate check"""
    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: Extract basic info
    initial_extraction = extract_event_info(user_input)

    # Gate check: Verify if it's a calendar event with sufficient confidence
    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calendar_event: {initial_extraction.is_calendar_event}, confidence: {initial_extraction.confidence_score:.2f}"
        )
        return None

    logger.info("Gate check passed, proceeding with event processing")

    # Second LLM call: Get detailed event information
    event_details = parse_event_details(initial_extraction.description)

    # Third LLM call: Generate confirmation
    confirmation = generate_confirmation(event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation


user_input = "Hi i need to setup a meeting on Project Roadmap with Rob"

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    print(f"Participants : {result.participants}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")

# %%
