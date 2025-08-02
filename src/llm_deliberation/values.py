import json
import os
from google import genai
from pydantic import BaseModel


class Values(BaseModel):
    answers: list[str]


def collapse_values(flat_values, messages):
    """
    Reshape a flat list of values into a nested list of lists, matching the shape of messages.

    Args:
        flat_values: List[Any] -- flat output from batch run, length = total messages
        messages: List[List[Any]] -- original nested messages; only the shape is used

    Returns:
        List[List[Any]] -- nested structure matching messages
    """
    output = []
    i = 0
    for msg_list in messages:
        inner = []
        for _ in msg_list:
            if i >= len(flat_values):
                raise ValueError("Not enough flat_values to fill out the messages structure.")
            inner.append(flat_values[i])
            i += 1
        output.append(inner)
    if i != len(flat_values):
        raise ValueError("Some flat_values left unused; messages structure does not match flat_values length.")
    return output


def process_value_batches(batch_id):
    """
    Process a Gemini AI batch job to extract values from batch responses.

    Downloads and processes the output file from a completed Gemini batch job,
    extracting the 'answers' field from each response and converting them to sets.

    Args:
        batch_id (str): The unique identifier of the Gemini batch job
    Returns:
        List[set]: A list of sets, where each set contains the values/answers 
                   extracted from one batch response
    Raises:
        KeyError: If the expected response structure is not found
        json.JSONDecodeError: If response content is not valid JSON
    """
    # Initialize Gemini client with API key from environment
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    # Retrieve the batch job details
    batch = client.batches.get(name=batch_id)
    # Get the output file ID from the batch destination
    file_id = batch.dest.file_name
    # Download the batch results file
    file = client.files.download(file=file_id)
    # Parse the file content - each line is a separate JSON response
    lines = file.decode('utf-8').strip().split('\n')
    outputs = [json.loads(line) for line in lines]
    # Extract values from each response and convert to sets for deduplication
    # Navigate the nested response structure: response -> candidates[0] -> content -> parts[0] -> text
    values = [set(json.loads(output['response']['candidates'][0]['content']['parts'][0]['text'])['answers'])
              for output in outputs]
    return values


values = [
   "Trust creation and maintenance",
   "Constructive dialogue",
   "Respect and dignity",
   "Professional ethics and integrity",
   "Social etiquette",
   "Religious respect and accommodation",
   "Linguistic respect and inclusivity",
   "Cultural understanding and respect",
   "Cultural heritage and tradition",
   "Financial wellbeing",
   "Sexual freedom and pleasure",
   "Protection of self and others from harm",
   "Environmental consciousness",
   "Authentic expression",
   "Workplace boundaries",
   "Parental care",
   "Consumer and client protection",
   "Child welfare",
   "Animal and pet welfare",
   "Worker welfare and dignity",
   "Workplace etiquette and respect",
   "Economic justice and fairness",
   "Healthcare equity and access",
   "Consent and personal boundaries",
   "Property rights protection",
   "Personal autonomy",
   "Emotional safety and support",
   "Mental health sensitivity and support",
   "Power dynamics values",
   "Privacy and confidentiality",
   "Religious and spiritual authenticity",
   "Emotional intelligence and regulation",
   "Emotional intimacy",
   "Prosocial altruism",
   "Honest communication",
   "Intergenerational respect and relationships",
   "Supportive and caring relationships",
   "Family bonds and cohesion",
   "Conflict resolution and reconciliation",
   "Public good and community engagement",
   "Accessibility",
   "Reciprocal relationship quality",
   "Environmental consciousness",
   "Empathy and understanding",
   "Personal growth",
   "Achievement and recognition",
   "Balance and moderation",
   "Physical health and wellbeing",
   "Personal accountability and responsibility"
]
