# RAG_Ollama
Public Repository

This is a Chatbot project created using a public dataset of Car Dealership company(provided in repository) which contains sold car details and customer reviews. I have used Python, LangChain,Ollama, ChromaDB (For RAG) and Streamlit (Frontend) for developing this project. There would be more features added whenever possible.

I have used llama3.1 LLM in this project. However, one can use advanced models based on their compute infrastructure.

Must do points:
1. Install Ollama to run LLM locally.
2. Set CHROMA_PATH = Provide ChromaDB Path and DATASET_PATH = Provide Dataset Path in .env file
3. Provide embedding function of your choice and a name for your collection in "gen_context.py" which will be create embeddings and collection. 
4. Run "gen_context.py" program to create embeddings and collection. This will create a directory with SQLite and Collection that is used for RAG. 
5. After creating embeddings and collection, you can run streamlit program using "streamlit run chat_app.py" in command prompt.

Below is the demo of output.
![alt text](image.png)