from typing import Dict, Any, Optional
import json
import re
from data_loader import get_dataset_type

class QueryAnalyzer:
    def __init__(self, llm=None):
        self.llm = llm
        # Define fields for both order and product datasets
        self.order_fields = {
            'Customer_Id': 'The customer ID number',
            'Order_Date': 'The date of the order in YYYY-MM-DD format',
            'Product': 'The name of the product',
            'Product_Category': 'The category of the product',
            'Time': 'The time in HH:MM format',
            'Order_Priority': 'The priority level of the order',
            'Payment_method': 'The method of payment used'
        }
        
        self.product_fields = {
            'main_category': 'The main category of the product',
            'title': 'The title/name of the product',
            'average_rating': 'The average rating of the product',
            'rating_number': 'The number of ratings for the product',
            'features': 'The features of the product',
            'description': 'The product description',
            'price': 'The price of the product',
            'store': 'The store selling the product',
            'categories': 'The categories the product belongs to',
            'details': 'Additional product details',
            'parent_asin': 'The parent ASIN of the product'
        }
        
        # Keep regex patterns as fallback for both types
        self.order_patterns = {
            'Customer_Id': r'(?:customer\s+(?:id|ID)\s*|ID\s+)(\d+)',
            'Order_Date': r'(?:on|date|ordered)\s+(\d{4}-\d{2}-\d{2})',
            'Product': r'product\s+(?:named|called|is)\s+([\w\s]+)',
            'Product_Category': r'category\s+(?:named|called|is)\s+([\w\s]+)',
            'Time': r'at\s+time\s+(\d{2}:\d{2})',
            'Order_Priority': r'priority\s+(?:is|was)\s+([\w\s]+)',
            'Payment_method': r'payment\s+(?:method|type)\s+(?:is|was)\s+([\w\s]+)'
        }
#main_category,title,average_rating,rating_number,features,description,price,store,categories,details,parent_asin
        
        self.product_patterns = {
            'main_category': r'main\s+category\s+(?:is|was)\s+([\w\s]+)',
            'title': r'(?:product|item)\s+(?:named|called|title|is)\s+([\w\s]+)',
            'average_rating': r'(?:average\s+)?rating\s+(?:is|was|of)\s+(\d+(?:\.\d+)?)',
            'rating_number': r'(?:number\s+of\s+)?ratings?\s+(?:is|was|of)\s+(\d+)',
            'features': r'features?\s+(?:are|is|include)\s+([\w\s,]+)',
            'description': r'description\s+(?:is|was|contains)\s+([\w\s]+)',
            'price': r'price\s+(?:is|was|of)\s+(\d+(?:\.\d{2})?)',
            'store': r'store\s+(?:is|was|named)\s+([\w\s]+)',
            'categories': r'categories?\s+(?:are|is|include)\s+([\w\s,]+)',
            'details': r'details?\s+(?:are|is|include)\s+([\w\s]+)',
            'parent_asin': r'(?:parent\s+)?asin\s+(?:is|was)\s+([A-Z0-9]+)'
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        
        dataset_type = get_dataset_type(query)
        filters = {}
        
        fields = self.product_fields if dataset_type == 'product' else self.order_fields
        patterns = self.product_patterns if dataset_type == 'product' else self.order_patterns
        
        for field, pattern in patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                filters[field] = match.group(1).strip()
                print(f"Regex matched {field} with value: {filters[field]}")
        
        if self.llm and not filters:
            prompt = f"""
            Extract relevant information from the following query into a structured format.
            Query: "{query}"

            Please extract ONLY the following fields if they are present in the query:
            {json.dumps(fields, indent=2)}

            Return the result in valid JSON format with ONLY the fields that are explicitly mentioned in the query.
            If a field is not mentioned in the query, do not include it in the JSON.
            
            Example format:
            {{
                "title": "Laptop",
                "price": "999.99"
            }}
            
            Response (in JSON):
            """

            try:
                response = self.llm.invoke(prompt)
                json_str = response.content if hasattr(response, 'content') else str(response)
                json_str = json_str.strip()
                
                if '{' in json_str and '}' in json_str:
                    start = json_str.find('{')
                    end = json_str.rfind('}') + 1
                    json_str = json_str[start:end]
                    
                    json_str = json_str.replace('\n', ' ').replace('\r', '')
                    json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
                    
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
        return {'filters': filters}

    def create_filter_function(self, filters: Dict[str, Any]) -> Optional[callable]:
        """
        Create a filter function based on the extracted filters.
        
        Args:
            filters (Dict[str, Any]): Dictionary of filters to apply
            dataset_type (str): Type of dataset ('order' or 'product')
            
        Returns:
            Optional[callable]: A function that can be used to filter documents
        """
        if not filters:
            return None
            
        def filter_func(doc):
            for field, value in filters.items():
                # Convert both values to string and lowercase for comparison
                doc_value = str(doc.metadata.get(field, '')).lower().strip()
                filter_value = str(value).lower().strip()
                
                # Check if the field exists and values match
                if field not in doc.metadata or doc_value != filter_value:
                    return False
            return True
            
        return filter_func
    

#main_category,title,average_rating,rating_number,features,description,price,store,categories,details,parent_asin
