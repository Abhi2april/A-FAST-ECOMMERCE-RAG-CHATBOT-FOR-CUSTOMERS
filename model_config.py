from langchain.chat_models import init_chat_model
from time import time

def setup_model():
    """
    Initialize the Groq model
    Returns:
        llm: The initialized language model
    """
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    return llm

def test_model(llm, prompt_to_test):
    """
    Perform a query and print the result
    Args:
        llm: The language model
        prompt_to_test: the prompt
    Returns:
        str: The generated response content
    """
    time_1 = time()
    response = llm.invoke(prompt_to_test)
    time_2 = time()
    print(f"Test inference: {round(time_2-time_1, 3)} sec.")
    # Extract just the content from the response
    content = response.content if hasattr(response, 'content') else str(response)
    print(f"Result: {content}")
    return content 