import os
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool

def create_ollama_rag_agent(llm, vectorstore, tavily_api_key):
    """Create the RAG agent."""
    tools = []
    if vectorstore:
        retriever = vectorstore.as_retriever()
        document_retriever_tool = create_retriever_tool(
            retriever,
            "document_retriever",
            "Searches and returns information from the uploaded documents."
        )
        tools.append(document_retriever_tool)
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        tavily_search_tool = TavilySearchResults(max_results=3)
        tools.append(tavily_search_tool)
    
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

def run_ollama_rag_agent(agent_executor, input_text, chat_history, vectorstore):
    """Run the RAG agent and get the response and sources."""
    response = agent_executor.invoke({
        "input": input_text, 
        "chat_history": chat_history
    })

    # Ekstrak sumber dari langkah-langkah perantara jika ada
    unique_sources = []
    if "intermediate_steps" in response and vectorstore:
        retriever = vectorstore.as_retriever()
        for step in response["intermediate_steps"]:
            if step[0].tool == "document_retriever":
                # Ambil dokumen sumber berdasarkan input ke tool
                docs = retriever.get_relevant_documents(step[0].tool_input)
                unique_sources.extend(docs)
    
    # Hapus duplikat berdasarkan konten halaman
    unique_sources = list({doc.page_content: doc for doc in unique_sources}.values())

    return response['output'], unique_sources

def convert_chat_history(chat_history):
    converted_history = []
    for message in chat_history:
        if message["role"] == "user":
            converted_history.append(HumanMessage(content=message["content"]))
        else:
            converted_history.append(AIMessage(content=message["content"]))
    return converted_history