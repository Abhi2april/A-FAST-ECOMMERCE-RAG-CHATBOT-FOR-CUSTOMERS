Soome change was made in the `api.py` API file to support real-time data access and processing.
- **Ngrok** is used to expose the dataset, enabling remote access through a public URL.

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. Open a new terminal and paste the localhost URL into ngrok to generate a public URL:
   ```bash
   ngrok http 8000
   ```

3. Copy the ngrok-generated public URL and update it in the `config.py` file.

4. In `data_loader.py`, modify the data loading section:
   - **Comment out** the `pd.read_csv(...)` line.
   - **Uncomment** the `GetResponse` lines to fetch data from the hosted ngrok link.



FOR TESTING THE RAG YOU CAN JUST MANUALLY ADD YOUR BOTH ORDER AND PRODUCT'S DATASET'S CSV FILE PATH IN THE data_loader.py file inside the load_data() function