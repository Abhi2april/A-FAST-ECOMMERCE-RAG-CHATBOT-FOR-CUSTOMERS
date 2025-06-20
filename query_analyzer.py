from typing import Dict, Any, Optional
import json
import re
from data_loader import get_dataset_type

class QueryAnalyzer:
    def __init__(self, llm=None):
        self.llm = llm
        # Define database schema for both tables
        self.order_schema = {
            'table_name': 'orders',
            'columns': {
                'Customer_Id': 'INTEGER - The customer ID number',
                'Order_Date': 'DATE - The date of the order in YYYY-MM-DD format',
                'Product': 'TEXT - The name of the product',
                'Product_Category': 'TEXT - The category of the product',
                'Time': 'TIME - The time in HH:MM format',
                'Order_Priority': 'TEXT - The priority level of the order',
                'Payment_method': 'TEXT - The method of payment used'
            }
        }
        
        self.product_schema = {
            'table_name': 'products',
            'columns': {
                'main_category': 'TEXT - The main category of the product',
                'title': 'TEXT - The title/name of the product',
                'average_rating': 'REAL - The average rating of the product',
                'rating_number': 'INTEGER - The number of ratings for the product',
                'features': 'TEXT - The features of the product',
                'description': 'TEXT - The product description',
                'price': 'REAL - The price of the product',
                'store': 'TEXT - The store selling the product',
                'categories': 'TEXT - The categories the product belongs to',
                'details': 'TEXT - Additional product details',
                'parent_asin': 'TEXT - The parent ASIN of the product'
            }
        }

    def generate_sql_query(self, query: str) -> Dict[str, Any]:
        """
        Generate SQL query from natural language query using LLM
        """
        dataset_type = get_dataset_type(query)
        schema = self.product_schema if dataset_type == 'product' else self.order_schema
        
        # Create schema description for the LLM
        schema_description = f"""
        Table: {schema['table_name']}
        Columns:
        """
        for col, desc in schema['columns'].items():
            schema_description += f"- {col}: {desc}\n"
        
        prompt = f"""
        You are a SQL query generator. Convert the following natural language query into a SQL SELECT statement.

        Database Schema:
        {schema_description}

        Natural Language Query: "{query}"

        Rules:
        1. Generate ONLY a valid SQL SELECT statement
        2. Use appropriate WHERE clauses for filtering
        3. Use LIKE operator for text searches (case-insensitive with %)
        4. Use proper SQL syntax for dates, numbers, and text
        5. For extracting the year, month, or day from a date column, use SQLite syntax: strftime('%Y', column) for year, strftime('%m', column) for month, strftime('%d', column) for day. For example, to filter year 2018: strftime('%Y', Order_Date) = '2018'
        6. The SQL must be valid for SQLite database engine
        7. Limit results to 50 rows maximum
        8. Return only the SQL query, no explanations

        SQL Query:
        """

        if not self.llm:
            raise ValueError("LLM is required for SQL query generation")
            
        response = self.llm.invoke(prompt)
        sql_query = response.content if hasattr(response, 'content') else str(response)
        sql_query = sql_query.strip()
        
        # Clean up the SQL query
        if sql_query.startswith('```sql'):
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        elif sql_query.startswith('```'):
            sql_query = sql_query.replace('```', '').strip()
        
        # Remove any trailing semicolon and add it back
        sql_query = sql_query.rstrip(';') + ';'
        
        print(f"Generated SQL: {sql_query}")
        return {
            'sql_query': sql_query,
            'table_name': schema['table_name'],
            'dataset_type': dataset_type
        }


#check if it is a valid sql or not
    def validate_sql(self, sql_query: str) -> bool:
        
        # Remove comments and extra whitespace
        sql_clean = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE).strip()
        
        # if it starts with SELECT
        if not sql_clean.upper().startswith('SELECT'):
            return False
        
        # for balanced parentheses
        if sql_clean.count('(') != sql_clean.count(')'):
            return False
        
        # basic SQL injection patterns (simple check)
        dangerous_patterns = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        for pattern in dangerous_patterns:
            if pattern in sql_clean.upper():
                return False
        
        return True