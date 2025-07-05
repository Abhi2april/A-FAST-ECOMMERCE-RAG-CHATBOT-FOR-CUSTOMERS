import streamlit as st
import os
import sqlite3
from dotenv import load_dotenv
from main import create_documents, build_vectorstore, setup_database, execute_sql_query, format_sql_results
from model_config import setup_workflow, setup_model
from data_loader import load_data
from query_analyzer import QueryAnalyzer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

st.set_page_config(
    page_title="E-Commerce Chatbot",
)

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vectordb = None
    st.session_state.llm = None
    st.session_state.query_analyzer = None
    st.session_state.app = None



def initialize_system():
    with st.spinner("Initializing the system..."):

        st.session_state.llm = setup_model()
        st.session_state.query_analyzer = QueryAnalyzer(llm=st.session_state.llm)
        st.session_state.app = setup_workflow()

        order_df, product_df = load_data()
        docs = create_documents((order_df, product_df))
        st.session_state.vectordb = build_vectorstore(docs)

        st.session_state.initialized = True




with st.sidebar:
    st.title("RAG System")
    
    if st.button("Initialize System"):
        initialize_system()

# Main content
st.title("E-Commerce Interface")

if not st.session_state.initialized:
    st.info("click 'Initialize System' in the sidebar to start.")
    st.stop()

if 'thread_counter' not in st.session_state:
    st.session_state.thread_counter = 1

query = st.text_input(
    "Enter your query:",
    placeholder="What would you like to know?",
    key="query_input"
)
config = {"configurable": {"thread_id": str(st.session_state.thread_counter)}}

if query:
    with st.spinner("Processing your query..."):
        if query.lower() == 'new':
            st.session_state.thread_counter += 1
            config = {"configurable": {"thread_id": str(st.session_state.thread_counter)}}
            st.warning("Started new conversation thread!")
            st.stop()

        sql_result = st.session_state.query_analyzer.generate_sql_query(query)
        if not sql_result or 'sql_query' not in sql_result:
            st.warning("Could not generate SQL query from your request. Please try rephrasing.")
            st.stop()
        sql_query = sql_result['sql_query']

        if not st.session_state.query_analyzer.validate_sql(sql_query):
            st.warning("Generated SQL query appears to be invalid or unsafe. Please try a different query.")
            st.stop()

        # a new connection for this query to avoid threading issues
        conn = sqlite3.connect('ecommerce.db')
        result_df = execute_sql_query(conn, sql_query)
        conn.close()

        used_vectorstore = False
        if result_df.empty:
            # vectorstore to retrieve relevant documents
            retriever = st.session_state.vectordb.as_retriever(
                search_kwargs={
                    "k": 10
                    }
                )
            try:
                relevant_docs = retriever.invoke(query)
            except Exception as e:
                st.warning(f"Vectorstore retrieval failed: {e}")
                relevant_docs = []
            if not relevant_docs:
                st.warning("No results found for your query in SQL or vectorstore. Try rephrasing or using different keywords.")
                
            formatted_results = "\n".join([
                f"Document {i+1}:\n{doc.page_content}\nMetadata: {doc.metadata}"
                for i, doc in enumerate(relevant_docs)
            ])
            used_vectorstore = True
        else:
     # format results for LLM context
            formatted_results = format_sql_results(result_df, query)
        # --- should end fallback logic ---
        
        if used_vectorstore:
            prompt = f"""
            Original Query: {query}
            
            (No SQL results, using vectorstore retrieval)
            
            Retrieved Documents:
            {formatted_results}

            Instructions: 
            1. Analyze the retrieved documents carefully
            2. Answer the original query using the information from the documents only if customer provided necessary details.
            3. If the documents don't fully answer the question, explain what information is available.
            4. Be specific and include only relevant details from the documents.
            5. Format your response as a helpful ecommerce assistant would.
            6. If there are multiple documents, summarize key insights.
            7. Remember our previous conversation and provide contextual responses when relevant.
            8. IF CUSTOMER DOESN'T PROVIDE NECESSARY DETAILS LIKE CUSTOMER ID THEN DON'T TELL THEM ABOUT ANY RETRIEVED DOCUMENT.

            Please provide a clear, helpful response based on the data above.
            """
        else:
            prompt = f"""
            Original Query: {query}
            
            SQL Query Executed: {sql_query}
            
            Query Results:
            {formatted_results}

            Instructions: 
            1. Analyze the query results carefully
            2. Answer the original query using the data from the SQL results only if customer provided necessary details.
            3. If the data doesn't fully answer the question, explain what information is available.
            4. Be specific and include relevant details from the results.
            5. Format your response as a helpful ecommerce assistant would.
            6. If there are multiple results, summarize key insights.
            7. Remember our previous conversation and provide contextual responses when relevant.
            8. IF CUSTOMER DOESN'T PROVIDE NECESSARY DETAILS LIKE CUSTOMER ID THEN DON'T TELL THEM ABOUT ANY RETRIEVED DOCUMENT.

            Please provide a clear, helpful response based on the data above.
            """
        
        input_messages = [HumanMessage(content=prompt)]
        result = st.session_state.app.invoke({"messages": input_messages}, config)
        
        response_content = result["messages"][-1].content
        st.success(f"Result: {response_content}") 