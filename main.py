import sys

from flow import create_agent_flow

def main():
    "Simple fucntion to process question"
    default_question = "Who won the Nobel Prize in Physics in 2024?"
    
    #Get the question from command line arguments if provided
    question = default_question
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            question = arg[2:]
            break
    
    agent_flow = create_agent_flow()
    shared = {"question": question}
    print(f"Processing question: {question}")
    agent_flow.run(shared)
    print("Final answer ")
    print(f"Answer: {shared.get('answer', 'No answer found')}")
    
if __name__ == "__main__":
    main()