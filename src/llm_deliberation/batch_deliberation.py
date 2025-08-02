import json
import os
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from google import genai
from google.genai import types


def batch_deliberation_google(model, system_prompt, histories, schema, batch_input_file="TEMP.jsonl", max_output_tokens=2500, temperature=1):
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    with open(batch_input_file, "w") as fout:
        for dilemma_id, history in histories.items():
            fout.write(json.dumps({
                "key": dilemma_id,
                "request": {
                    "contents": history,
                    "system_instruction": {"parts": [{'text': system_prompt}]},
                    "generation_config": {
                        "max_output_tokens": max_output_tokens,
                        "temperature": temperature,
                        "response_mime_type": "application/json",
                        "response_json_schema": schema,
                        "thinking_config": {"thinking_budget": 0}
                    }
                }
            }) + "\n")

    uploaded_file = client.files.upload(
        file=batch_input_file,
        config=types.UploadFileConfig(mime_type='jsonl'))

    batch_job = client.batches.create(
        model=model,
        src=uploaded_file.name)
    return batch_job


def get_batch_results_google(batch_id):
    """
    Retrieve and parse batch processing results from Google GenAI.

    Args:
        batch_id (str): The ID of the batch job to retrieve results for
    Returns:
        list: List of dictionaries containing parsed JSON results from each line
              of the batch output file, or None if batch is still in progress
    """
    # Initialize the Google GenAI client with API key
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    # Retrieve the batch job information
    batch = client.batches.get(name=batch_id)

    # Check if batch has completed and has output file
    if not hasattr(batch, 'dest') or not batch.dest or not batch.dest.file_name:
        print("Batch In Progress")
        return None

    # Get the output file ID from the completed batch
    file_id = batch.dest.file_name
    # Download the results file
    file = client.files.download(file=file_id)
    # Parse the file content - each line is a JSON object
    lines = file.decode('utf-8').strip().split('\n')
    outputs = [json.loads(line) for line in lines]
    return outputs


def process_batch_results_google(outputs):
    """
    Convert Google batch results to DataFrame with structured columns.
    """

    rows = []

    for item in outputs:
        key = item['key']
        response = item['response']

        # Extract the numeric ID (part after underscore)
        dilemma_id = int(key.split('_')[-1])

        # Get the text content and parse JSON
        text_content = response['candidates'][0]['content']['parts'][0]['text']
        text_content = text_content.strip('```json').strip('```').strip()
        tool_data = json.loads(text_content)

        agent = tool_data['agent'].replace(' ', '')  # "Agent 1" -> "Agent1"
        round_num = tool_data['round']
        verdict = tool_data['verdict']
        explanation = tool_data['explanation']

        row = {
            f"{agent.lower()}_round{round_num}_verdict": verdict,
            f"{agent.lower()}_round{round_num}_explanation": explanation
        }
        rows.append((dilemma_id, row))

    # Sort by dilemma_id and create DataFrame
    rows.sort(key=lambda x: x[0])
    df = pd.DataFrame([row[1] for row in rows], index=[row[0] for row in rows])
    return df


def batch_deliberation_anthropic(model, system_prompt, histories, schema, max_output_tokens=2500, temperature=1):
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    tools = [
        {
            "name": "json_output",
            "description": "Enforces structured outputs for verdicts.",
            "input_schema": schema
        }
    ]
    requests = [
        Request(
            custom_id=dilemma_id,
            params=MessageCreateParamsNonStreaming(
                model=model,
                system=system_prompt,
                max_tokens=max_output_tokens,
                messages=history,
                temperature=temperature,
                tools=tools)
            )
        for dilemma_id, history in histories.items()]
    batch = client.messages.batches.create(requests=requests)
    return batch

def get_batch_results_anthropic(batch_id):
    """
    Get batch results - returns dict of outputs if ready, otherwise status/error message.
    """
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    try:
        batch = client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            print(f"Batch {batch.processing_status}")
            return

        if batch.request_counts.errored > 0:
            results = client.messages.batches.results(batch_id)
            errors = [f"{r.custom_id}: {r.result.error}" for r in results if r.result.type == "errored"]
            print(f"Errors: {'; '.join(errors)}")
            return

        # Success - return dict of results
        results = client.messages.batches.results(batch_id)
        return {r.custom_id: r.result.message for r in results if r.result.type == "succeeded"}

    except Exception as e:
        return f"Error: {e}"

def process_batch_results_anthropic(outputs):
    """
    Convert batch results to DataFrame with structured columns.
    """

    rows = []

    for key, message in outputs.items():
        # Extract the numeric ID (part after underscore)
        dilemma_id = int(key.split('_')[-1])
        
        # Find the tool use block
        tool_data = None
        for block in message.content:
            if block.type == "tool_use":
                tool_data = block.input
                break
        
        if tool_data:
            agent = tool_data['agent'].replace(' ', '')  # "Agent 2" -> "Agent2"
            round_num = tool_data['round']
            verdict = tool_data['verdict']
            explanation = tool_data['explanation']
            
            row = {
                f"{agent.lower()}_round{round_num}_verdict": verdict,
                f"{agent.lower()}_round{round_num}_explanation": explanation
            }
            rows.append((dilemma_id, row))

    # Sort by dilemma_id and create DataFrame
    rows.sort(key=lambda x: x[0])
    df = pd.DataFrame([row[1] for row in rows], index=[row[0] for row in rows])
    return df


def batch_deliberation_openai(model, histories, schema, batch_input_file="TEMP.jsonl", max_output_tokens=2500, temperature=1):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    # Open the specified file in write mode
    with open(batch_input_file, "w") as file:
        # Iterate through each dilemma to create a batch request payload
        for dilemma_id, history in histories.items():
            # Construct the payload for a single OpenAI chat completion request
            payload = {
                "custom_id": dilemma_id,  # Unique ID for this request
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": history,
                    "temperature": temperature,
                    "max_tokens": max_output_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "VerdictResponseDeliberation",
                            "schema": schema
                        }
                    }
                }
            }

            # Write the JSON payload as a single line followed by a newline
            file.write(json.dumps(payload) + "\n")

    # Upload the generated JSONL file to OpenAI's file storage
    batch_input_file = client.files.create(
        purpose='batch',
        file=open(batch_input_file, "rb"))

    # Create a new batch job using the uploaded file's ID.
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h")

    return batch


def get_batch_results_openai(batch_id):
    """
    Get batch results - returns dict of outputs if ready, otherwise status/error message.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    try:
        batch = client.batches.retrieve(batch_id)

        if batch.status != "completed":
            print(f"Batch {batch.status}")
            return

        if batch.request_counts.failed > 0:
            output_file = client.files.content(batch.output_file_id)
            lines = output_file.content.decode('utf-8').strip().split('\n')

            errors = []
            for line in lines:
                result = json.loads(line)
                if result.get('error'):
                    errors.append(f"{result['custom_id']}: {result['error']['message']}")

            if errors:
                print(f"Errors: {'; '.join(errors)}")
                return

        # Success - return dict of results
        output_file = client.files.content(batch.output_file_id)
        lines = output_file.content.decode('utf-8').strip().split('\n')

        results = {}
        for line in lines:
            result = json.loads(line)
            if not result.get('error'):
                results[result['custom_id']] = result['response']

        return results

    except Exception as e:
        return f"Error: {e}"


def process_batch_results_openai(outputs):
    """
    Convert OpenAI batch results to DataFrame with structured columns.
    """
    import pandas as pd
    import json
    
    rows = []
    
    for key, response in outputs.items():
        # Extract the numeric ID (part after underscore)
        dilemma_id = int(key.split('_')[-1])
        
        # Get the text content and parse JSON
        text_content = response['body']['choices'][0]['message']['content']
        tool_data = json.loads(text_content)
        
        agent = tool_data['agent'].replace(' ', '')  # "Agent 3" -> "Agent3"
        round_num = tool_data['round']
        verdict = tool_data['verdict']
        explanation = tool_data['explanation']
        
        row = {
            f"{agent.lower()}_round{round_num}_verdict": verdict,
            f"{agent.lower()}_round{round_num}_explanation": explanation
        }
        rows.append((dilemma_id, row))
    
    # Sort by dilemma_id and create DataFrame
    rows.sort(key=lambda x: x[0])
    df = pd.DataFrame([row[1] for row in rows], index=[row[0] for row in rows])
    return df


def remaining_idxs(df: pd.DataFrame, agent_names: list[str], round_num: int) -> pd.Index:
    vcols = [f"{a}_round{round_num}_verdict" for a in agent_names]
    sub   = df[vcols]

    all_missing = sub.isna().all(axis=1)          # deliberation already ended
    all_equal   = sub.nunique(axis=1, dropna=True) == 1
    active_mask = ~(all_missing | all_equal)

    return df.index[active_mask]

def default_msg(round_num: int, info: dict[str, dict]) -> str:
    lines = [f"Round {round_num} Summary:"]
    for a, rec in info.items():
        verdict      = rec["verdict"]
        explanation  = rec["explanation"]
        lines.append(f"\n{a} Verdict: {verdict}")
        lines.append(f"\n{a} Explanation: {explanation}")
    lines.append(f"\nConsensus was not reached. We proceed to Round {round_num+1}.")
    return "\n".join(lines)

def append_user(hist: list, msg: str, provider: str) -> None:
    if provider in {"openai", "anthropic"}:
        hist.append({"role": "user", "content": msg})
    elif provider == "google":
        hist.append({"parts": [{"text": msg}], "role": "user"})
    else:
        raise ValueError(f"unknown provider {provider}")

def advance_round(
    df: pd.DataFrame,
    round_num: int,
    cfg: dict[str, dict],
    msg_builder=default_msg,
) -> None:
    agents  = list(cfg.keys())                         # infer agent names
    agent_keys = [agent.lower().replace(' ', '') for agent in agents]
    active  = remaining_idxs(df, agent_keys, round_num)
    resolved = df.index.difference(active)

    # prune finished dilemmas
    for a in agents:
        for idx in resolved:
            cfg[a]["histories"].pop(f"dilemma_{idx}", None)

    # append next-round prompt
    for idx in active:
        info = {
            agent_name: {
                "verdict": df.at[idx, f"{agent_key}_round{round_num}_verdict"],
                "explanation": df.at[idx, f"{agent_key}_round{round_num}_explanation"],
            }
            for agent_name, agent_key in zip(agents, agent_keys)
        }
        msg = msg_builder(round_num, info)
        for a in agents:
            append_user(cfg[a]["histories"][f"dilemma_{idx}"], msg, cfg[a]["provider"])