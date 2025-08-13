from collections import defaultdict
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage

from llm_deliberation.deliberation import extract_verdict

async def round_robin_delib(dilemma, clients, model_names, system_prompt_template, max_rounds=4, verbose=True):
    all_results = []
    histories = defaultdict(list)

    if verbose:
        print("=" * 50)
        print("BEGINNING DELIBERATION")
        print("=" * 50)
        print(f"Dilemma: {dilemma}")

    # Initialize each agent's history with the system message and dilemma
    for idx, name in enumerate(model_names, start=1):
        histories[name].extend([
            SystemMessage(content=system_prompt_template(agent=idx)),
            UserMessage(content=dilemma, source="Moderator")
        ])

    for round_num in range(1, max_rounds + 1):
        round_outputs = {}
        contents = []
        # One-by-one, each agent sees the updated conversation so far
        for name, client in zip(model_names, clients.values()):
            response = await client.create(histories[name])
            content = response.content
            contents.append(content)
            round_outputs[name] = content
            if verbose:
                print("-" * 50)
                print(f"{name} (Round {round_num}):")
                print(content)

            # Append assistant message to *that* agent’s history
            message = AssistantMessage(content=content, source=name.replace(' ', ''))
            histories[name].append(message)
            all_results.append(message)

            # Add a user message *to all other* agents' histories showing what was said
            for idx, other_name in enumerate(model_names):
                if other_name != name:
                    histories[other_name].append(
                        UserMessage(content=f"{name} said:\n{content}", source="Moderator")
                    )

        # Extract verdicts and check agreement
        verdicts = {name: extract_verdict(output) for name, output in round_outputs.items()}

        if len(set(verdicts.values())) == 1:
            if verbose:
                print("CONSENSUS, DELIBERATION COMPLETE")
            break
        else:
            if verbose:
                print("NO CONSENSUS, PROCEEDING TO NEXT ROUND\n")
            new_message_content = \
                f"Round {round_num} Summary:\n" + \
                "\nAgent 1 said:\n" + contents[0] + "\n" + \
                "\nAgent 2 said:\n" + contents[1] + "\n" + \
                f"\nConsensus was not reached. We proceed to Round {round_num + 1}.\n"

            for name in model_names:
                histories[name].append(UserMessage(content=new_message_content, source="Moderator"))

    return all_results
