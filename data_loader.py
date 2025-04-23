import requests
import pandas as pd
from config import NGROK_URL

def load_data():
#    headers = {"ngrok-skip-browser-warning": "true"}
#    response = requests.get(NGROK_URL, headers=headers)
    df=pd.read_csv('C:/Users/ASUS/Desktop/RAG/Order_Data_Dataset.csv')
#    df = pd.DataFrame(response.json())
#    df=df[:1000]
    return df
