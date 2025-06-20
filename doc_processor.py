from langchain.schema import Document
import pandas as pd

def create_documents(dataframes):
    """
    Create documents from both order and product dataframes.
    
    Args:
        dataframes (tuple): A tuple containing (order_df, product_df)
        
    Returns:
        list: List of Document objects for both datasets
    """
    order_df, product_df = dataframes
    docs = []
    
    # Process order data
    for _, row in order_df.iterrows():
        metadata = {
            'Order_Date': str(row['Order_Date']),
            'Time': str(row['Time']),
            'Customer_Id': str(row['Customer_Id']),
            'Product': str(row['Product']),
            'Product_Category': str(row['Product_Category']),
            'dataset_type': 'order'  # Add dataset type to metadata
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

#main_category,title,average_rating,rating_number,features,description,price,store,categories,details,parent_asin

    # Process product data
    for _, row in product_df.iterrows():
        metadata = {
            'title': str(row['title']),
            'main_category': str(row['main_category']),
            'price': str(row['price']),
            'average_rating': str(row['average_rating']),
            'rating_number': str(row['rating_number']),
            'dataset_type': 'product'
        }
        
        # Create more comprehensive content for better searchability
        content = f"Product Information:\n"
        content += f"Title: {row['title']}\n"
        content += f"Category: {row['main_category']}\n"
        content += f"Price: ${row['price']}\n"
        content += f"Rating: {row['average_rating']} ({row['rating_number']} ratings)\n"
        
        #description if available and not too long
        if pd.notna(row['description']) and len(str(row['description'])) < 500:
            content += f"Description: {row['description']}\n"
        
        #features if available and not too long
        if pd.notna(row['features']) and len(str(row['features'])) < 500:
            content += f"Features: {row['features']}\n"
        
        #additional categories if available
        if pd.notna(row['categories']) and str(row['categories']) != 'nan':
            content += f"Additional Categories: {row['categories']}\n"

        #store information if available
        if pd.notna(row['store']) and str(row['store']) != 'nan':
            content += f"Store: {row['store']}\n"
        
        #etails if available and not too long
        if pd.notna(row['details']) and len(str(row['details'])) < 500:
            content += f"Details: {row['details']}\n"
        
        docs.append(Document(page_content=content, metadata=metadata))
    
    print(f"Created {len(docs)} documents ({len(order_df)} orders, {len(product_df)} products)")
    
    return docs
