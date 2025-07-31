import asyncio, os
import pandas as pd
import pickle

from collections import defaultdict
from typing import Dict, List
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from outlines import Template
from pyprojroot import here
from tqdm import tqdm

from llm_evaluations_everyday_dilemmas.deliberation import extract_verdict

# ── config ───────────────────────────────────────────────────────────────────
MAX_ROUNDS = 4
TEMPERATURE = 1
DILEMMAS_PATH = here("data/processed/candidates.csv")
PROMPT_PATH = here("src/llm_evaluations_everyday_dilemmas/prompts/asynchronous_deliberation_h2h.txt")
TRACKING_PATH = here("data/tracking/asynchronous_h2h_gpt4.1_gemini2.0-flash.pkl") 

# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DILEMMAS_PATH)
dilemmas = df['selftext_cleaned'].iloc[:1000]
template = Template.from_file(PROMPT_PATH)

# ── model clients ────────────────────────────────────────────────────────────
model_names = ['Agent1', 'Agent2']
clients = dict(zip(model_names, 
[
    # AnthropicChatCompletionClient(model="claude-3-7-sonnet-20250219",
    #                               temperature=TEMPERATURE,
    #                               api_key=os.getenv("ANTHROPIC_API_KEY")),
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
    OpenAIChatCompletionClient(
        model="gpt-4.1",
        temperature=TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY")),
    OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        temperature=TEMPERATURE,
        api_key=os.getenv("GEMINI_API_KEY")),
]))


# ── main deliberation loop ───────────────────────────────────────────────────
async def deliberate(dilemma, max_rounds=4):
    all_results = []
    histories: Dict[str, List] = defaultdict(list)

    for idx, name in enumerate(model_names, start=1):
        histories[name].extend([
            SystemMessage(content=template(agent=idx)),
            UserMessage(content=dilemma, source="Moderator")])

    for round in range(1, max_rounds + 1):
        coroutines = [client.create(histories[name])
                      for name, client in clients.items()]
        results = await asyncio.gather(*coroutines)

        contents = [result.content for result in results]

        for name, content in zip(model_names, contents):
            message = AssistantMessage(content=content, source=name)
            histories[name].append(message)
            all_results.append(message)

        verdicts = [extract_verdict(content) for content in contents]

        if len(set(verdicts)) == 1:
            #print(all_results[-2].content)
            #print(all_results[-1].content)
            break
        else:
            new_message_content = \
                f"Round {round} Summary:\n" + \
                "\nAgent 1 said:\n" + contents[0] + "\n" + \
                "\nAgent 2 said:\n" + contents[1] + "\n" + \
                f"\nConsensus was not reached. We proceed to Round {round + 1}.\n"
            update_message = UserMessage(content=new_message_content, source="Moderator")
            #print(new_message_content)
            [history.append(update_message) for history in histories.values()]
    return all_results


if os.path.exists(TRACKING_PATH):
    with open(TRACKING_PATH, 'rb') as file:
        outputs = pickle.load(file)
else:
    outputs = []

n_outputs = len(outputs)
sub = dilemmas.iloc[n_outputs:]

for idx, dilemma in tqdm(enumerate(sub), total=len(sub)):
    result = asyncio.run(deliberate(dilemma, max_rounds=MAX_ROUNDS))
    outputs.append(result)

    with open(TRACKING_PATH, 'wb') as file:
        pickle.dump(outputs, file)
