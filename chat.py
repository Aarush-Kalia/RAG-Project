from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


load_dotenv()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

llm = GoogleGenAI(
    model="gemini-3.5-flash",
)

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("my_notes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("what are the policy objectives of the central bank?")
print(response)

response = query_engine.query("How much does the Gray Whale cost")
print(response)