import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

def create_ollama_rag_agent(llm, vectorstore):
    """Create the RAG agent."""
    retriever = vectorstore.as_retriever()
    
    document_retriever_tool = create_retriever_tool(
        retriever,
        "document_retriever",
        "Searches and returns information from the uploaded documents."
    )
    
    tavily_search_tool = TavilySearchResults(max_results=3)
    
    tools = [document_retriever_tool, tavily_search_tool]
    
    prompt_template = PromptTemplate.from_template(
        """You are an AI assistant named OllaRAG. You are a helpful and harmless assistant.

        TOOLS:
        ------
        You have access to the following tools:
        {tools}

        To use a tool, please use the following format:
        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```

        Begin!

        Previous conversation history:
        {chat_history}

        New input: {input}
        {agent_scratchpad}
        """
    )
    
    agent = create_react_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

def run_ollama_rag_agent(agent_executor, input_text, chat_history):
    """Run the RAG agent and get the response."""
    response = agent_executor.invoke({"input": input_text, "chat_history": chat_history})
    return response['output']

def convert_chat_history(chat_history):
    converted_history = []
    for message in chat_history:
        if message["role"] == "user":
            converted_history.append(HumanMessage(content=message["content"]))
        else:
            converted_history.append(AIMessage(content=message["content"]))
    return converted_history