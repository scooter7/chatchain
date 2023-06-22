import os
import re
import copy
from typing import List, Union

from langchain.agents import Tool, LLMSingleActionAgent
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
import sys
from github import Github

# Check if the OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Check if the GitHub token is set
if "GITHUB_TOKEN" not in os.environ:
    print("Please set the GITHUB_TOKEN environment variable.")
    sys.exit(1)

# Retrieve the OpenAI API key and GitHub token from environment variables
openai_api_key = os.environ["OPENAI_API_KEY"]
github_token = os.environ["GITHUB_TOKEN"]

# Set up the GitHub API
g = Github(github_token)
repo = g.get_repo("scooter7/chatchain")


template = """Answer the following questions as best you can, but speaking as a college admissions counselor. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to answer as a passionate and informative travel expert when giving your final answer.

Question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser:
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class DDGS:
    def __enter__(self):
        # Code to set up the DDGS object or perform any necessary initialization
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Code to clean up or finalize any resources used by the DDGS object
        pass

    def run(self, query):
        # Code for running the search using DuckDuckGo
        pass


def search_online(input_text):
    with DDGS() as ddgs:
        search = ddgs.run(f"site:holyfamily.edu/ things to do{input_text}")
    return search


def search_general(input_text):
    with DDGS() as ddgs:
        search = ddgs.run(f"{input_text}")
    return search


class ConversationState:
    def __init__(self, input_text: str, agent_actions: List[AgentAction], observation: str = None):
        self.input_text = input_text
        self.agent_actions = agent_actions
        self.observation = observation

    def __str__(self):
        actions = "\n".join([f"Thought: {action.log}\nObservation: {action.observation}" for action in self.agent_actions])
        return f"Question: {self.input_text}\n{actions}"


class AgentExecutor:
    def __init__(self, agent: LLMSingleActionAgent, tools: List[Tool], verbose: bool = False):
        self.agent = agent
        self.tools = tools
        self.verbose = verbose
        self.chat_history = []

    def execute(self, input_text: str):
        # Create a new instance of the agent for each conversation
        agent = copy.deepcopy(self.agent)

        # Create the initial conversation state
        conversation_state = ConversationState(input_text=input_text, agent_actions=[])

        # Iterate through the conversation steps
        while not agent.is_finished(conversation_state):
            # Generate agent action
            agent_action = agent.generate_action(conversation_state)

            # Execute agent action
            observation = None
            if agent_action.tool in self.tools:
                tool = next((tool for tool in self.tools if tool.name == agent_action.tool), None)
                if tool:
                    observation = tool.func(agent_action.tool_input)

            # Update conversation state
            conversation_state.agent_actions.append(agent_action)
            conversation_state.observation = observation

            # Display conversation progress if verbose mode is enabled
            if self.verbose:
                print(conversation_state)

            # Append conversation history
            self.chat_history.append(str(conversation_state) + "\n")

        # Return the final answer and chat history
        return agent.get_final_answer(conversation_state), "\n".join(self.chat_history)


def save_chat_history(file_path: str, chat_history: str):
    # Get the directory of the main app file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the "content" directory if it doesn't exist
    content_dir = os.path.join(base_dir, "content")
    os.makedirs(content_dir, exist_ok=True)
    # Create the full file path within the "content" directory
    full_file_path = os.path.join(content_dir, file_path)
    # Write the chat history to the file
    with open(full_file_path, "w") as file:
        file.write(chat_history)



if __name__ == "__main__":
    tools = [
        Tool(
            name="Search general",
            func=search_general,
            description="Useful for when you need to answer general Holy Family University questions"
        )
    ]
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()
    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    agent = LLMSingleActionAgent(llm_chain=llm_chain, output_parser=output_parser)

    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    input_text = "What are the things to do on Holy Family University's campus?"
    final_answer, chat_history = executor.execute(input_text)

    print(f"Final Answer: {final_answer}")
    print(f"Chat History:\n{chat_history}")

    save_chat_history("chat_history.txt", chat_history)
