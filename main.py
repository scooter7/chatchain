from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
import os
import chainlit as cl

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

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


class CustomOutputParser(AgentOutputParser):
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

@cl.on_chat_start
async def main():
    res = await cl.AskUserMessage(content="Hi! What is your name?").send()
    if res:
        await cl.Message(
            content=f"Nice to meet you: {res['content']}!",
        ).send()
    res = await cl.AskUserMessage(content="What is your email address?").send()
    if res:
        await cl.Message(
            content=f"Thanks! How can I help you?",
        ).send()
        
@cl.langchain_factory(use_async=False)
def agent():
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
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"], allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    return agent_executor
