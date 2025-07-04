import os
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub

def create_ollama_rag_agent(llm, vectorstore, tavily_api_key):
    tools = []
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        @tool
        def document_retriever(query: str) -> str:
            """Searches and returns information from the local documents."""
            docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in docs])
        tools.append(document_retriever)

    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        search_tool = TavilySearchResults(max_results=3)
        tools.append(search_tool)
    
    agent_prompt = hub.pull("hwchase17/react-chat")
    
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False,
        handle_parsing_errors=True
    )
    return agent_executor

def run_ollama_rag_agent(agent_executor, prompt, chat_history, vectorstore):
    response_container = st.empty()
    full_response = ""
    sources = []
    
    stream_output = agent_executor.stream({
        "input": prompt,
        "chat_history": chat_history
    })

    for chunk in stream_output:
        if "output" in chunk:
            full_response += chunk["output"]
            response_container.markdown(full_response + "▌")
        
        if "intermediate_steps" in chunk:
            for step in chunk["intermediate_steps"]:
                action, observation = step
                if action.tool == "document_retriever":
                     retrieved_docs = vectorstore.as_retriever().invoke(action.tool_input)
                     sources.extend(retrieved_docs)

    response_container.markdown(full_response)
    
    unique_sources = list({doc.page_content: doc for doc in sources}.values())
    return full_response, unique_sources

def convert_chat_history(messages):
    chat_history = []
    for msg in messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history