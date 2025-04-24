##### install this also after pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


------------------------------------------------------------------------------------------------------------------------------------------
Custom Features Implemented:
(to enhace faster searching and deliverability, i added a custom double llm feature)
LLM-1 (Query Analyzer):
Interprets the user query and predicts a filter key (e.g., Customer_Id). This value is then used for document filtering before full retrieval.
Fallback: ONly if regex-based filtering fails, it defaults to using the llm.

LLM-2 (RAG Responder):
Functions like a standard Retrieval-Augmented Generation model. It takes the original query and the filtered documents to generate a final response.
