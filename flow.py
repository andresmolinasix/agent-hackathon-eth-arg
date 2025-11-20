from pocketflow import Flow


def create_agent_flow():
    from nodes import DecideAction, SearchWeb, AnswerQuestion, WriteFile
    # First, create instances of each node
    decide = DecideAction()
    search = SearchWeb()
    write = WriteFile()
    answer = AnswerQuestion()

    # Now connect them together - this is where the magic happens!
    # Each connection defines where to go based on the action returned by post()

    # If DecideAction returns "search", go to SearchWeb
    decide - "search" >> search

    # If DecideAction wants to write immediately, go to WriteFile
    decide - "write" >> write

    # If DecideAction returns "answer", go to AnswerQuestion
    decide - "answer" >> answer

    # After SearchWeb completes, go back to DecideAction
    search - "decide" >> decide

    # After an answer is produced, optionally write it to disk
    answer - "write" >> write

    # After writing, fall back to DecideAction unless the node signals completion
    write - "decide" >> decide

    return Flow(start=decide)
