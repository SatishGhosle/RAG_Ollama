import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import os, sys
import polars as pl
import pathlib
import time
class Rag:
    def __init__(self, folder):
        self.folder_path = folder
    
    def prepare_chrm_dict(self):
        paths_list = list(pathlib.Path(self.folder_path).rglob("*"))
        print("Total Files", len(paths_list))

        # sequence must be same as csv columns
        schema = {  "": pl.Int64,
                  "Review_Date": pl.Utf8,
                  "Author_Name": pl.Utf8,
                  "Vehicle_Title": pl.Utf8,
                  "Review_Title": pl.Utf8,
                  "Review": pl.Utf8,
                  "Rating": pl.Float64}

        car_review_df = pl.DataFrame()

        for file in paths_list:
            temp_df = pl.read_csv(file, schema = schema, ignore_errors = True)
            car_review_df = pl.concat((car_review_df, temp_df))

        car_review_df = car_review_df.with_columns(
            pl.col("Vehicle_Title").str.slice(0,4).alias("Launch_Year"),
            pl.col("Vehicle_Title").str.split(" ").list.get(1).alias("Vehicle_Model"),
        )

        car_review_df = car_review_df.fill_nan("None")
        car_review_df = car_review_df.fill_null("None")
        print("Shape = ", car_review_df.shape)
        print("\nCompleted the data pre-process")

        # Prepare ID
        print("\nCreating IDs...")
        doc_id = [ f"reviews_{i}" for i in range(len(car_review_df))]

        # Prepare Documents
        print("\nCreating Documents...")
        docs = car_review_df.select(pl.col("Review")).to_series().to_list()


        # Prepare Metadata
        print("\nCreating Metadata...")
        meta_list = []
        for row in car_review_df.drop(["", "Review"]).iter_rows():
            meta_list.append({"Review_Date": row[0],
                              "Author_Name": row[1],
                              "Vehicle_Title": row[2],
                              "Review_Title": row[3],
                              "Rating": row[4],
                              "Launch_Year": row[5],
                              "Vehicle_Model": row[6],
                              })
        return {"doc_ids": doc_id, "review_docs":docs, "metadata":meta_list}
    
    def create_collection(self, chroma_path:str, 
                          embedding_fun:str,
                          collection_name:str,
                          data_dict:dict,
                          dist_metric:str = "cosine",
                          ):
        chroma_dict = data_dict

        print("Chroma SQLite Client is Created")
        chroma_db_client = chromadb.PersistentClient(chroma_path)

        print("Max batch size of chromadb =", chroma_db_client.get_max_batch_size())

        print("\nInitializing Embedding Function...")
        use_embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name = embedding_fun
        )

        # Create Collection
        print("\nCreating Collection Instance...")
        
        collection = chroma_db_client.get_or_create_collection(name = collection_name,
                                                               embedding_function = use_embed_func,
                                                               metadata = {"hnsw:space":dist_metric})
        
        print("\nAdding Collection...@", time.strftime("%Y-%m-%d %I:%M:%S"), time.localtime())
        
        # Add Collection
        collection.add(ids = data_dict.get("doc_ids"),
                       documents = data_dict.get("review_docs"),
                       metadatas = data_dict.get("metadata"))
        
        print("\n DB is created and Ready to query..@", time.strftime("%Y-%m-%d %I:%M:%S"), time.localtime())

        
        
        


    
