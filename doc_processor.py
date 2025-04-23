from langchain.schema import Document

def create_documents(df):
    docs = []
    for _, row in df.iterrows():
        metadata = {
            'Order_Date': str(row['Order_Date']),
            'Time': str(row['Time']),
            'Customer_Id': str(row['Customer_Id']),
            'Product': str(row['Product']),
            'Product_Category': str(row['Product_Category'])
        }
        
        content = f"""
Order_Date: {row['Order_Date']}
Time: {row['Time']}
Aging: {row['Aging']}
Customer_Id: {row['Customer_Id']}
Gender: {row['Gender']}
Device_Type: {row['Device_Type']}
Customer_Login_type: {row['Customer_Login_type']}
Product_Category: {row['Product_Category']}
Product: {row['Product']}
Quantity: {row['Quantity']}
Discount: {row['Discount']}
Profit: {row['Profit']}
Sales: {row['Sales']}
Shipping_Cost: {row['Shipping_Cost']}
Order_Priority: {row['Order_Priority']}
Payment_method: {row['Payment_method']}
"""
        docs.append(Document(page_content=content, metadata=metadata))
    return docs