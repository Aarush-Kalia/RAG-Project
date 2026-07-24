from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate


load_dotenv()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

llm = GoogleGenAI(
    model="gemini-3.6-flash",
)

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("my_notes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

qa_template = PromptTemplate("Answer the question using only the notes given. \n \n Notes: \n {context_str} \n Question: \n {query_str}")

query_engine = index.as_query_engine(llm=llm, text_qa_template=qa_template)

while True:
    question = input("Ask a question(Type 'end' to exit the program): ")
    if question == "end":
        break
    response = query_engine.query(question)
    print(response)
    for node in response.source_nodes:
        print("from " + node.metadata["file_name"] + ", page " + node.metadata.get("page_label", "N/A") + " (score = " + str(node.score) + ")")