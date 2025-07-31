import re
import pandas as pd

from collections import Counter
from pydantic import BaseModel, Field
from statistics import mode


VERDICT_REGEX = re.compile(r"My current verdict:\s*(.+)", re.IGNORECASE)

class VerdictResponseDeliberation(BaseModel):
    """Pydantic model for structured verdict responses."""
    agent: str = Field(..., description="The name of the agent rendering the verdict.")
    round: int = Field(..., description="The current deliberation round number.")
    verdict: str = Field(..., description="The verdict chosen from the allowed categories (e.g., YTA, NTA).")
    explanation: str = Field(..., description="The detailed explanation for the verdict.")


def make_check_agreement(round_length=3):
    agent_messages = []  # Only store assistant responses

    def check_agreement(message: list) -> bool:
        latest = message[0]
        if latest.source != "user":
            agent_messages.append(latest)

        if len(agent_messages) < round_length or len(agent_messages) % round_length != 0:
            return False

        last_round = agent_messages[-round_length:]
        verdicts = []
        for m in last_round:
            match = VERDICT_REGEX.search(m.content)
            if match:
                verdict = match.group(1).strip().upper().rstrip(".!?:;")
            else:
                verdict = None
            verdicts.append(verdict)
        return len(set(verdicts)) == 1 and None not in verdicts

    return check_agreement


def extract_verdict(text: str) -> str | None:
    """
    Extracts the verdict from a model's message text.

    Expects a sentence like: "My current verdict: YTA." or similar.

    Returns
    -------
    str or None
        The extracted verdict, uppercased and stripped of trailing punctuation,
        or None if not found.
    """
    match = VERDICT_REGEX.search(text)
    if not match:
        return None
    return match.group(1).strip().upper().rstrip(".!?:;")