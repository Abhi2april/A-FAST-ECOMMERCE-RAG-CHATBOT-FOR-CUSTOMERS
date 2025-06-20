from imports import *

logging.basicConfig(level=logging.INFO, filename="main.log", filemode="w", 
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )

logger=logging.getLogger(__name__)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler=logging.FileHandler('test.log', mode='w')
handler.setFormatter(formatter)

consolehandler=logging.StreamHandler(sys.stdout)
consolehandler.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(consolehandler)

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def setup_database():
    """
    Create SQLite database and load CSV data into tables
    """
    conn = sqlite3.connect('ecommerce.db')
    cursor = conn.cursor()
    
    # Load data
    order_df, product_df = load_data()
    
    # Create tables and insert data
    order_df.to_sql('orders', conn, if_exists='replace', index=False)
    product_df.to_sql('products', conn, if_exists='replace', index=False)
    
    logger.info(f"Database created with {len(order_df)} order records and {len(product_df)} product records")
    
    return conn

def execute_sql_query(conn, sql_query: str) -> pd.DataFrame:
    try:
        result_df = pd.read_sql_query(sql_query, conn)
        logger.info(f"SQL query executed successfully, returned {len(result_df)} rows")
        return result_df
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        return pd.DataFrame()

def format_sql_results(df: pd.DataFrame, query: str) -> str:
    if df.empty:
        return "No results found for the given query."
    
    #the number of rows to display (for context length)
    max_rows = 10
    if len(df) > max_rows:
        df_display = df.head(max_rows)
        truncated_msg = f"\n... (showing first {max_rows} of {len(df)} total results)"
    else:
        df_display = df
        truncated_msg = ""
    
    formatted_context = f"""
                    Query Results Summary:
                    - Total rows found: {len(df)}
                    - Columns: {', '.join(df.columns.tolist())}

                    Data:
                    {df_display.to_string(index=False)}
                    {truncated_msg}
                                    """
    
    return formatted_context

def main():
    llm = setup_model()
    app = setup_workflow()
    thread_counter = 1
    config = {"configurable": {"thread_id": str(thread_counter)}}
    query_analyzer = QueryAnalyzer(llm=llm)

    logger.info("Setting up database...")
    conn = setup_database()
    logger.info("Database setup completed")

    # vectorstore for fallback
    logger.info("Loading data for vectorstore...")
    order_df, product_df = load_data()
    docs = create_documents((order_df, product_df))
    logger.info(f"Created {len(docs)} documents for vectorstore")
    vectordb = build_vectorstore(docs)
    logger.info("Vectorstore built for fallback retrieval")
    #

    # checking the llm
    test_prompt = "Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words."
    test_model(llm, test_prompt)

    # for continuous asking
    while True:    
        query = input("Type 'new' if you want a new conversation. "
                      "OR "
                      "Ask your query (or type 'exit' / 'quit' to quit): ")
        
        if query.lower() in ['exit', 'quit']:
            logger.info("Exiting the assistant.")
            conn.close()
            break

        # Start a new conversation
        if query.lower() == 'new':
            thread_counter += 1
            config = {"configurable": {"thread_id": str(thread_counter)}}
            logger.info("Started new conversation thread!")
            continue

        logger.info(f"\nProcessing query: {query}")
        
        # Generate SQL query from natural language
        sql_result = query_analyzer.generate_sql_query(query)
        
        if not sql_result or 'sql_query' not in sql_result:
            logger.info("Could not generate SQL query from your request. Please try rephrasing.")
            continue
        
        sql_query = sql_result['sql_query']
        logger.info(f"Generated SQL: {sql_query}")
        
        # Validate SQL query
        if not query_analyzer.validate_sql(sql_query):
            logger.info("Generated SQL query appears to be invalid or unsafe. Please try a different query.")
            continue
        
        # Execute SQL query
        result_df = execute_sql_query(conn, sql_query)
        
        # fallback to vectorstore if SQL fails or returns no results
        used_vectorstore = False
        if result_df.empty:
            logger.info("SQL returned no results or failed. Falling back to vectorstore retrieval.")
            # Use vectorstore to retrieve relevant documents
            retriever = vectordb.as_retriever(
                search_kwargs={
                    "k": 5
                    }
                )
            try:
                relevant_docs = retriever.invoke(query)
            except Exception as e:
                logger.error(f"Vectorstore retrieval failed: {e}")
                relevant_docs = []
            if not relevant_docs:
                logger.info("No results found for your query in SQL or vectorstore. Try rephrasing or using different keywords.")
                continue
            # Format vectorstore results
            formatted_results = "\n".join([
                f"Document {i+1}:\n{doc.page_content}\nMetadata: {doc.metadata}"
                for i, doc in enumerate(relevant_docs)
            ])
            used_vectorstore = True
        else:
            # Format results for LLM context
            formatted_results = format_sql_results(result_df, query)
        # --- End fallback logic ---
        
        # Create prompt for LLM with results
        if used_vectorstore:
            prompt = f"""
            Original Query: {query}
            
            (No SQL results, using vectorstore retrieval)
            
            Retrieved Documents:
            {formatted_results}

            Instructions: 
            1. Analyze the retrieved documents carefully
            2. Answer the original query using the information from the documents
            3. If the documents don't fully answer the question, explain what information is available
            4. Be specific and include relevant details from the documents
            5. Format your response as a helpful ecommerce assistant would
            6. If there are multiple documents, summarize key insights
            7. Remember our previous conversation and provide contextual responses when relevant

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
            2. Answer the original query using the data from the SQL results
            3. If the data doesn't fully answer the question, explain what information is available
            4. Be specific and include relevant details from the results
            5. Format your response as a helpful ecommerce assistant would
            6. If there are multiple results, summarize key insights
            7. Remember our previous conversation and provide contextual responses when relevant

            Please provide a clear, helpful response based on the data above.
            """
        
        # Use LangGraph for memory management
        input_messages = [HumanMessage(content=prompt)]
        result = app.invoke({"messages": input_messages}, config)
        
        logger.info("Generating response...")
        logger.info("=" * 50)
        
        # Extract and display the response
        response_content = result["messages"][-1].content
        logger.info(f"Result: {response_content}")
        
        logger.info("Assistant Response:")
        logger.info("-" * 20)
        logger.info(response_content)
        logger.info("=" * 50)
        
        # Optional: Show the SQL query that was executed
        if used_vectorstore:
            logger.info(f"\n(Vectorstore retrieval used for: {query})")
        else:
            logger.info(f"\n(SQL Query used: {sql_query})")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()