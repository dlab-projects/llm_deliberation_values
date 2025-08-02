import re
import pandas as pd

from collections import Counter
from pydantic import BaseModel, Field
from statistics import mode


VERDICT_REGEX = re.compile(r"My current verdict:\s*(.+)", re.IGNORECASE)


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


def get_final_verdict_safe(verdicts, n_agents):
    """Get final verdict with proper handling of ties."""
    if len(verdicts) < n_agents:
        return None

    final_round_verdicts = verdicts[-n_agents:]
    # Remove None values for counting
    valid_verdicts = [v for v in final_round_verdicts if v is not None]

    if not valid_verdicts:
        return None

    # Count occurrences
    verdict_counts = Counter(valid_verdicts)
    max_count = max(verdict_counts.values())

    # Check if there's a clear majority (more than half)
    if max_count > len(valid_verdicts) / 2:
        # Find the verdict with max count
        for verdict, count in verdict_counts.items():
            if count == max_count:
                return verdict

    # No clear majority
    return None


def extract_agent_chains(agents, verdicts, n_agents):
    """Extract verdict chains for each agent across rounds."""
    agent_chains = {}

    # Determine number of rounds
    total_messages = len(agents)
    n_rounds = total_messages // n_agents

    # Group messages by rounds
    for round_idx in range(n_rounds):
        start_idx = round_idx * n_agents
        end_idx = start_idx + n_agents

        round_agents = agents[start_idx:end_idx]
        round_verdicts = verdicts[start_idx:end_idx]

        for idx, (agent, verdict) in enumerate(zip(round_agents, round_verdicts)):
            agent_key = f"Agent_{idx+1}"
            if agent_key not in agent_chains:
                agent_chains[agent_key] = []
            agent_chains[agent_key].append(verdict)

    return agent_chains


def extract_agent_messages(agents, explanations, n_agents):
    """Extract message chains for each agent across rounds."""
    agent_messages = {}

    # Determine number of rounds
    total_messages = len(agents)
    n_rounds = total_messages // n_agents

    # Group messages by rounds
    for round_idx in range(n_rounds):
        start_idx = round_idx * n_agents
        end_idx = start_idx + n_agents
        
        round_agents = agents[start_idx:end_idx]
        round_explanations = explanations[start_idx:end_idx]
        
        for idx, (agent, explanation) in enumerate(zip(round_agents, round_explanations)):
            agent_key = f"Agent_{idx+1}"
            if agent_key not in agent_messages:
                agent_messages[agent_key] = []
            agent_messages[agent_key].append(explanation)
    
    return agent_messages


def process_round_robin_deliberation(results, n_agents):
    agents = []
    verdicts = []
    explanations = []

    messages = results.messages if hasattr(results, 'messages') else results

    for message in messages:
        if hasattr(message, 'source') and message.source == "user":
            continue
        agents.append(message.source)
        text = message.content
        explanations.append(text)
        verdicts.append(extract_verdict(text))

    final_verdict = mode(verdicts[-n_agents:])
    return agents, explanations, verdicts, final_verdict


def process_deliberation_results(results_list, n_agents, max_rounds, individual_rounds=False):
    """
    Process a list of deliberation results into a comprehensive dataframe.

    Args:
        results_list: List of result objects, each with a .messages attribute
        n_agents: Number of agents in each deliberation
        max_rounds: Maximum number of rounds expected across all experiments

    Returns:
        pandas.DataFrame with columns for verdict chains, agent-specific data, and final verdicts
    """

    data = []

    for idx, results in enumerate(results_list):
        # Process single result using your existing function
        agents, explanations, verdicts, _ = process_round_robin_deliberation(results, n_agents)

        # Calculate actual number of rounds
        total_messages = len(agents)
        n_rounds = total_messages // n_agents

        # Extract agent-specific chains
        agent_verdict_chains = extract_agent_chains(agents, verdicts, n_agents)
        agent_message_chains = extract_agent_messages(agents, explanations, n_agents)

        # Get final verdict with proper tie handling
        final_verdict = get_final_verdict_safe(verdicts, n_agents)

        # Create row data
        row_data = {
            'n_rounds': n_rounds,
            'final_verdict': final_verdict,
            'verdict_chain': verdicts.copy()
        }

        # Add agent-specific verdict chains
        for agent_num in range(1, n_agents + 1):
            agent_key = f'Agent_{agent_num}'
            if agent_key in agent_verdict_chains:
                row_data[f'{agent_key}_verdicts'] = agent_verdict_chains[agent_key].copy()

        # Add agent-specific message chains
        for agent_num in range(1, n_agents + 1):
            agent_key = f'Agent_{agent_num}'
            if agent_key in agent_message_chains:
                row_data[f'{agent_key}_messages'] = agent_message_chains[agent_key].copy()

        if individual_rounds:
            # Add individual round columns for each agent
            for round_num in range(1, max_rounds + 1):
                for agent_num in range(1, n_agents + 1):
                    col_name = f'Agent_{agent_num}_Round_{round_num}'

                    # Check if this round and agent exists in the data
                    if round_num <= n_rounds:
                        message_idx = (round_num - 1) * n_agents + (agent_num - 1)
                        if message_idx < len(explanations):
                            row_data[col_name] = explanations[message_idx]
                        else:
                            row_data[col_name] = None
                    else:
                        row_data[col_name] = None

            # Add agent names as separate columns
            for agent_num in range(1, n_agents + 1):
                col_name = f'Agent_{agent_num}_name'
                # Get the agent name from the first occurrence of this agent
                agent_name = None
                for round_num in range(n_rounds):
                    agent_idx_in_round = (agent_num - 1)
                    message_idx = round_num * n_agents + agent_idx_in_round
                    if message_idx < len(agents):
                        agent_name = agents[message_idx]
                        break
                row_data[col_name] = agent_name

        data.append(row_data)

    return pd.DataFrame(data)


def change_of_minds(verdicts):
    return verdicts.apply(lambda x: (len(x) > 1) & (len(set(x)) > 1)).sum()