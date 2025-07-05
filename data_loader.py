import requests
import pandas as pd
#from config import NGROK_URL

def load_data():
    # loading both the provided datasets in form of csv first for testing purposes, 
    # so later it can be used via ngrok's api linkage.
#   headers = {"ngrok-skip-browser-warning": "true"}
#   response = requests.get(NGROK_URL, headers=headers)
    order_df = pd.read_csv(r'C:\Users\ASUS\Desktop\A-FAST-ECOMMERCE-RAG-CHATBOT-FOR-CUSTOMERS\data\Order_Data_Dataset.csv')
    product_df = pd.read_csv(r'C:\Users\ASUS\Desktop\A-FAST-ECOMMERCE-RAG-CHATBOT-FOR-CUSTOMERS\data\Product_Information_Dataset.csv')
    return order_df, product_df

#this will tell weather the query is about product dataset or order dataset 

def get_dataset_type(query: str) -> str:
    
    product_keywords = ['product', 'item', 'price', 'inventory', 'stock', 'description', 'specification']
    order_keywords = ['order', 'purchase', 'customer', 'date', 'payment', 'shipping']
    
    query_lower = query.lower()
    
    product_matches = sum(1 for keyword in product_keywords if keyword in query_lower)
    order_matches = sum(1 for keyword in order_keywords if keyword in query_lower)
    
    return 'product' if product_matches>order_matches else 'order'
