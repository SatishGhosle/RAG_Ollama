from chroma_db_embeddings import Rag
from dotenv import load_dotenv, dotenv_values
import os, sys
import pathlib

# load_dotenv(dotenv_path= path of .env file)
load_dotenv(pathlib.Path(__file__).parent/".env")

# Load environment variables
os.environ["DATASET_PATH"] = os.getenv("DATASET_PATH")
os.environ["CHROMA_PATH"] = os.getenv("CHROMA_PATH")

''' Instantiated Driver Program...'''
rag_o = Rag(folder = os.environ["DATASET_PATH"])

data_dict = rag_o.prepare_chrm_dict()
collection_instance = rag_o.create_collection(chroma_path = os.environ["CHROMA_PATH"],
                        embedding_fun = "all-MiniLM-L6-v2",
                        collection_name = "car_dealer",
                        data_dict = data_dict,
                        dist_metric = "cosine"
                        )
