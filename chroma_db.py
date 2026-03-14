import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings

import pandas as pd
import numpy as np

#database 
COLLECTION_NAME = "my_collection" #database name 
EMBED_MODEL = "intfloat/multilingual-e5-large"
CHROMA_DB_PATH = "chroma_db" #databas path 

#load dataframe 
df = pd.read_csv("chromadb-vector_database-\query_job_dataset.csv") #load data from csv file
print(df.columns) #print first 5 rows of the dataframe to check if data is loaded correctly

# Clean missing values  
df["ID_JOB"] = df["ID_JOB"].astype(str)
df["JOB_DESCRIPTION"] = df["JOB_DESCRIPTION"].fillna("").astype(str)
df["POSITION_NAME"] = df["POSITION_NAME"].fillna("").astype(str)
df["JOB_QUALIFICATION"] = df["JOB_QUALIFICATION"].fillna("").astype(str)

# Prepare data
ids = df["ID_JOB"].tolist()
documents = [
    f"passage: ตำแหน่งงาน {p}. รายละเอียดงาน {d}. คุณสมบัติ {q}"
    for p, d, q in zip(
        df["POSITION_NAME"],
        df["JOB_DESCRIPTION"],
        df["JOB_QUALIFICATION"]
    )
]
position_name = df["POSITION_NAME"].tolist()
job_qualification = df["JOB_QUALIFICATION"].tolist()

#meatdatas 
metadatas = [
    {
        "position_name": n,
        "job_qualification": j
    }
    for n, j in zip(position_name, job_qualification)
]

#Create client(Database)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

#create embedding model 
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

#Create collection(table)

collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=embedding_func
)

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

#print number of documents in the collection
print(f"Number of documents in the collection: {collection.count()}")

#print first 5 documents in the collection to check if data is added correctly
print(collection.peek())

#query data from the database
def query_collection(query:str):
    top_k = 5

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    return results

text = "ตำแหน่งอาจารย์สอนวิชาเบเกอรี่และการจัดการเรียนการสอน"

query = query_collection(query=text)
print(query)

