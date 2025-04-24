# 🛒 E-commerce Assistant Bot

This project is an implementation of a Retrieval-Augmented Generation (RAG) based assistant designed for e-commerce platforms. While I couldn’t fully grasp every aspect initially, I managed to complete most of the core functionalities and included several custom enhancements to improve performance and response quality.

---

## 🔧 Modifications & Setup

- Soome change was made in the `api.py` API file to support real-time data access and processing.
- **Ngrok** is used to expose the dataset, enabling remote access through a public URL.
- To address hallucination issues caused by a large dataset (~54,000 documents), a **custom search and retrieval** technique was developed.
- The retriever fetches top-k relevant documents by first filtering through metadata using either **regex** or **LLM-based query interpretation** and then adding in the filter dictionary in retriever.

---

## 🚀 Running the API

Follow these steps to run the project:

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

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

---

## ⚠️ Issues Faced

- **Dataset Merging**:  
  Difficulty in merging the product description and order details datasets due to inconsistent keys or structural differences.

- **Delivery Timing Integration**:  
  Uncertainty about how and where to display delivery timing information in the model output, and how it should be saved in the final CSV structure.

- **Agentic Memory**:  
  Challenges in enabling the RAG system to remember previous queries and context, as demonstrated in multi-turn interaction examples.

---

## 💡 Custom Features Implemented

To enhance both search speed and delivery precision, a **custom dual-LLM mechanism** was integrated:

### 1. LLM-1 (Query Analyzer)
- Interprets the user query to infer a filter key (e.g., `Customer_Id`).
- This inferred filter is used for metadata-based document filtering.
- **Fallback Strategy**: If regex-based extraction fails, it defaults to using the LLM for filter prediction.

### 2. LLM-2 (RAG Responder)
- Acts as a standard RAG model.
- It takes the original query and the documents retrieved by LLM-1 and generates a context-aware response.

