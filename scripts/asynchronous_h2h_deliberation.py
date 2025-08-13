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
PROMPT_PATH = here("prompts/asynchronous_deliberation_h2h_v3.txt")
TRACKING_PATH = here("data/tracking/exp10_async_h2h.pkl") 

# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DILEMMAS_PATH)
dilemmas = df['selftext_cleaned'].iloc[:1000]
template = Template.from_file(PROMPT_PATH)

# ── model clients ────────────────────────────────────────────────────────────
model_names = ['Agent1', 'Agent2']

# agent1 = AnthropicChatCompletionClient(
#     model="claude-3-7-sonnet-20250219",
#     temperature=TEMPERATURE,
#     api_key=os.getenv("ANTHROPIC_API_KEY"))
agent1 = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    temperature=TEMPERATURE,
    api_key=os.getenv("GEMINI_API_KEY"))
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
    # dilemma = """I was talking to my boyfriend about his upcoming boys trip that is he is taking with his 3 friends when I jokingly said that I better not see him hanging out with girls in a hot tub (he takes this trip every year and there is a hot tub). He replied, "oh well (we will call him Eli) Eli's little cousin might be coming and she is a girl". (Also this girl is from the town their boys trip is in so she has bombarded their boys trip before). Emphasis on the little, I asked how old she was, and he says 20. LITTLE????? So then I asked her what her name was (so that I could look her up on social media cause im crazy). Keep in mind, this whole conversation happened over the phone because we are long distance. It took my boyfriend a solid 5 seconds to "think" of said ‘little cousin’s’ name. We will call her Hailey. I immediately find Hailey's instagram cause im like that, and gasped. So the fact that my boyfriend said "little cousin" like she was going to be 12-14 years old pissed me off. Upon looking at her instagram profile, I also noticed that my boyfriend had multiple of Hailey's pictures liked (she was in a swimsuit). So that made me suspicious because he acted like he couldn't think of her name. Now, I could care less if my boyfriend hangs out with females, but I am just suspicious because he tried to act like he barely knew the girl, and swept significant details about her under the rug like she is definitely over the age of 20 and that should not be considered "little" along with the fact that he knows exactly who she is and has hungout with her multiple times while on this boys trip in the past."""
    try:
        result = asyncio.run(async_delib(
            dilemma=dilemma,
            clients=clients,
            model_names=model_names,
            system_prompt_template=template,
            verbose=False,
            max_rounds=MAX_ROUNDS))
    except:
        print(f"Error processing dilemma {idx + n_outputs}")
        break
    outputs.append(result)

    with open(TRACKING_PATH, 'wb') as file:
        pickle.dump(outputs, file)

