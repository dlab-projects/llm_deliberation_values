import asyncio
from collections import defaultdict
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage

from llm_deliberation.deliberation import extract_verdict


async def async_delib(dilemma, clients, model_names, system_prompt_template, max_rounds=4, verbose=False):
    all_results = []
    histories = defaultdict(list)

    if verbose:
        print("=" * 50)
        print("BEGINNING DELIBERATION")
        print("=" * 50)
        print(f"Dilemma: {dilemma}")

    for idx, name in enumerate(model_names, start=1):
        histories[name].extend([
            SystemMessage(content=system_prompt_template(agent=idx)),
            UserMessage(content=dilemma, source="Moderator")
        ])

    for round in range(1, max_rounds + 1):
        coroutines = [client.create(histories[name])
                      for name, client in clients.items()]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        contents = [result.content for result in results]

        for name, content in zip(model_names, contents):
            message = AssistantMessage(content=content, source=name)
            histories[name].append(message)
            all_results.append(message)

        verdicts = [extract_verdict(content) for content in contents]

        if verbose:
            print("=" * 50)
            print(f"ROUND {round} RESULTS")
            print("=" * 50)
            for name, content in zip(model_names, contents):
                print("-" * 25)
                print(f"{name}")
                print(f"{content}")
                print("-" * 25)

        if len(set(verdicts)) == 1:
            if verbose:
                print("CONSENSUS, DELIBERATION COMPLETE")
            break
        else:
            if verbose:
                print("NO CONSENSUS, PROCEEDING TO NEXT ROUND")
            new_message_content = \
                f"Round {round} Summary:\n" + \
                "\nAgent 1 said:\n" + contents[0] + "\n" + \
                "\nAgent 2 said:\n" + contents[1] + "\n" + \
                f"\nConsensus was not reached. We proceed to Round {round + 1}.\n"
            update_message = UserMessage(content=new_message_content, source="Moderator")
            [history.append(update_message) for history in histories.values()]
    return all_results