import json
import os
import pandas as pd
import pickle
from google.genai import Client
from google.genai import types
from outlines import Template
from pyprojroot import here

from llm_deliberation.values import Values

AGENT = 2
SYSTEM_PROMPT = here("prompts/identify_values_dilemma.txt")
system_prompt = Template.from_file(here(SYSTEM_PROMPT))()
response_schema = Values.model_json_schema()

with open(here('data/analysis/exp6_async_h2h.pkl'), 'rb') as file:
    df = pickle.load(file)

N_SCENARIOS = df.shape[0]
client = Client(api_key=os.getenv('GEMINI_API_KEY'))

# Collect all verdicts (one per message) for the batch
prompts = []
for idx in range(N_SCENARIOS):
    messages = df[f'Agent_{AGENT}_messages'].iloc[idx]
    for message in messages:
        verdict = "\n".join(message.splitlines()[1:])
        prompts.append(verdict)

batch_input_file = here(f"data/jobs/exp5_agent{AGENT}.jsonl")
with open(batch_input_file, "w") as fout:
    for idx, verdict in enumerate(prompts):
        fout.write(json.dumps({
            "key": f"request{idx}",
            "request": {
                "contents": [{
                    'parts': [{'text': verdict}],
                    'role': 'user'
                }],
                "system_instruction": {"parts": [{'text': system_prompt}]},
                "generation_config": {
                    "max_output_tokens": 5000,
                    "temperature": 1,
                    "response_mime_type": "application/json",
                    "response_json_schema": Values.model_json_schema()
                }
            }
        }) + "\n")

uploaded_file = client.files.upload(
    file=batch_input_file,
    config=types.UploadFileConfig(
        display_name='agent1_values',
        mime_type='jsonl'))

batch_job = client.batches.create(
    model="models/gemini-2.5-flash",
    src=uploaded_file.name
)

print(f"Batch job: {batch_job.name}")
