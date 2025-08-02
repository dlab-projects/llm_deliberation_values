import asyncio, os
import pandas as pd
import pickle

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from outlines import Template
from pyprojroot import here
from tqdm import tqdm

from llm_deliberation.async_delib import async_delib


# ── config ───────────────────────────────────────────────────────────────────
MAX_ROUNDS = 4
TEMPERATURE = 1
DILEMMAS_PATH = here("data/processed/scenarios_verdicts.csv")
PROMPT_PATH = here("prompts/asynchronous_deliberation_h2h_v2.txt")
TRACKING_PATH = here("data/tracking/asynchronous_h2h_claude37_gpt41_exp3_prompt3.pkl") 

# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DILEMMAS_PATH)
dilemmas = df['selftext_cleaned'].iloc[:1000]
template = Template.from_file(PROMPT_PATH)

# ── model clients ────────────────────────────────────────────────────────────
model_names = ['Agent1', 'Agent2']
agent1 = AnthropicChatCompletionClient(
    model="claude-3-7-sonnet-20250219",
    temperature=TEMPERATURE,
    api_key=os.getenv("ANTHROPIC_API_KEY"))
agent2 = OpenAIChatCompletionClient(
    model="gpt-4.1-2025-04-14",
    temperature=TEMPERATURE,
    api_key=os.getenv("OPENAI_API_KEY"))

clients = {'Agent1': agent1, 'Agent2': agent2}


if os.path.exists(TRACKING_PATH):
    with open(TRACKING_PATH, 'rb') as file:
        outputs = pickle.load(file)
else:
    outputs = []

n_outputs = len(outputs)
sub = dilemmas.iloc[n_outputs:]

for idx, dilemma in tqdm(enumerate(sub), total=len(sub)):
    result = asyncio.run(async_delib(
        dilemma=dilemma,
        clients=clients,
        model_names=model_names,
        system_prompt_template=template,
        verbose=False,
        max_rounds=MAX_ROUNDS))
    outputs.append(result)

    with open(TRACKING_PATH, 'wb') as file:
        pickle.dump(outputs, file)


# OpenAIChatCompletionClient(
#     model="gemini-2.5-flash",
#     temperature=TEMPERATURE,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     model_info={
#         "family": "GEMINI_2_5_FLASH",  # align with Gemini's model family
#         "function_calling": True,
#         "structured_output": False,
#         "json_output": False,
#         "vision": False,
#         "reasoning": "none"
#     },
#     api_key=os.getenv("GEMINI_API_KEY")),

# OpenAIChatCompletionClient(
#     model="gemini-2.0-flash",
#     temperature=TEMPERATURE,
#     api_key=os.getenv("GEMINI_API_KEY")),