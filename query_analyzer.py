from typing import Dict, Any, Optional
import json
import re

class QueryAnalyzer:
    def __init__(self, llm=None):
        self.llm = llm
        # Define the fields we want to extract
        self.fields = {
            'Customer_Id': 'The customer ID number',
            'Order_Date': 'The date of the order in YYYY-MM-DD format',
            'Product': 'The name of the product',
            'Product_Category': 'The category of the product',
            'Time': 'The time in HH:MM format',
            'Order_Priority': 'The priority level of the order',
            'Payment_method': 'The method of payment used'
        }
        # Keep regex patterns as fallback
        self.filter_patterns = {
            'Customer_Id': r'(?:customer\s+(?:id|ID)\s*|ID\s+)(\d+)',
            'Order_Date': r'(?:on|date|ordered)\s+(\d{4}-\d{2}-\d{2})',
            'Product': r'product\s+(?:named|called|is)\s+([\w\s]+)',
            'Product_Category': r'category\s+(?:named|called|is)\s+([\w\s]+)',
            'Time': r'at\s+time\s+(\d{2}:\d{2})',
            'Order_Priority': r'priority\s+(?:is|was)\s+([\w\s]+)',
            'Payment_method': r'payment\s+(?:method|type)\s+(?:is|was)\s+([\w\s]+)'
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query using both LLM and regex patterns to extract relevant filters.
        
        Args:
            query (str): The user's query
            
        Returns:
            Dict[str, Any]: Dictionary containing the filters to apply
        """
        filters = {}
        
        # First try regex patterns
        for field, pattern in self.filter_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                filters[field] = match.group(1).strip()
                print(f"Regex matched {field} with value: {filters[field]}")
        
        # If we have an LLM and no filters were found, try LLM
        if self.llm and not filters:
            prompt = f"""
            Extract relevant information from the following query into a structured format.
            Query: "{query}"

            Please extract ONLY the following fields if they are present in the query:
            {json.dumps(self.fields, indent=2)}

            Return the result in valid JSON format with ONLY the fields that are explicitly mentioned in the query.
            If a field is not mentioned in the query, do not include it in the JSON.
            
            Example format:
            {{
                "Customer_Id": "12345",
                "Product": "Laptop"
            }}
            
            Response (in JSON):
            """

            try:
                response = self.llm.invoke(prompt)
                # Extract JSON from response
                json_str = response.content if hasattr(response, 'content') else str(response)
                json_str = json_str.strip()
                
                # Try to find JSON in the response
                if '{' in json_str and '}' in json_str:
                    start = json_str.find('{')
                    end = json_str.rfind('}') + 1
                    json_str = json_str[start:end]
                    
                    # Clean up the JSON string
                    json_str = json_str.replace('\n', ' ').replace('\r', '')
                    json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)  # Remove comments
                    
                    try:
                        llm_filters = json.loads(json_str)
                        print(f"LLM extracted filters: {llm_filters}")
                        filters.update(llm_filters)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing LLM JSON: {e}")
                        print(f"Raw JSON string: {json_str}")
            except Exception as e:
                print(f"Error in LLM processing: {e}")
        
        print(f"Final filters: {filters}")
        return filters

    def create_filter_function(self, filters: Dict[str, Any]) -> Optional[callable]:
        """
        Create a filter function based on the extracted filters.
        
        Args:
            filters (Dict[str, Any]): Dictionary of filters to apply
            
        Returns:
            Optional[callable]: A function that can be used to filter documents
        """
        if not filters:
            return None
            
        def filter_func(doc):
            for field, value in filters.items():
                # Convert both values to string and lowercase for comparison
                doc_value = str(doc.metadata.get(field, '')).lower()
                filter_value = str(value).lower()
                
                # Check if the field exists and values match
                if field not in doc.metadata or doc_value != filter_value:
                    return False
            return True
            
        return filter_func