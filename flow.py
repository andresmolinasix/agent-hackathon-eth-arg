from pocketflow import Flow


def create_agent_flow():
    from nodes import (
        DecideAction,
        SearchWeb,
        AnswerQuestion,
        WriteFile,
        FetchPrice,
        FetchYields,
        FetchVolume,
        AssessRisk,
    )
    # First, create instances of each node
    decide = DecideAction()
    search = SearchWeb()
    write = WriteFile()
    answer = AnswerQuestion()
    fetch_price = FetchPrice()
    fetch_yields = FetchYields()
    fetch_volume = FetchVolume()
    assess_risk = AssessRisk()

    # Now connect them together - this is where the magic happens!
    # Each connection defines where to go based on the action returned by post()

    # If DecideAction returns "search", go to SearchWeb
    decide - "search" >> search

    # If DecideAction wants to write immediately, go to WriteFile
    decide - "write" >> write

    # If DecideAction returns "answer", go to AnswerQuestion
    decide - "answer" >> answer

    # Financial data tools
    decide - "fetch_price" >> fetch_price
    decide - "fetch_yields" >> fetch_yields
    decide - "fetch_volume" >> fetch_volume
    decide - "assess_risk" >> assess_risk

    # After SearchWeb completes, go back to DecideAction
    search - "decide" >> decide

    # After an answer is produced, optionally write it to disk
    answer - "write" >> write

    # Data tools return to planner
    fetch_price - "answer" >> answer
    fetch_price - "decide" >> decide
    fetch_yields - "decide" >> decide
    fetch_volume - "decide" >> decide

    # Risk assessment can produce an answer directly
    assess_risk - "answer" >> answer

    # After writing, fall back to DecideAction unless the node signals completion
    write - "decide" >> decide

    return Flow(start=decide)
