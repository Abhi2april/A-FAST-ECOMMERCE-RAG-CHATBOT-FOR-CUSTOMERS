from langchain.chat_models import init_chat_model
from time import time
import os
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def setup_model():
#    llm = init_chat_model("meta-llama/llama-4-scout-17b-16e-instruct", model_provider="groq")
    llm = ChatGroq(model_name="compound-beta-mini")
    return llm



def setup_workflow():

    model=setup_model()

    workflow = StateGraph(state_schema=MessagesState)


    # Define the function that calls the model
    def call_model(state: MessagesState):
        system_prompt = (
            "You are an E-Commerce bot, and assigned to reply to customers, "
            "you are strictly prohibited to reply any other irrelevant questions."
            "Answer all questions to the best of your ability using the provided context."
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = model.invoke(messages)
        return {"messages": response}


    # Define the node and edge
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # Add simple in-memory checkpointer
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


def test_model(llm, prompt_to_test):
    time_1 = time()
    response = llm.invoke(prompt_to_test)
    time_2 = time()
    print(f"Test inference: {round(time_2-time_1, 3)} sec.")
    content = response.content if hasattr(response, 'content') else str(response)
    print(f"Result: {content}")
    return content 

####------####
