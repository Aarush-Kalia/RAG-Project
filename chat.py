from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI


load_dotenv()

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

llm = GoogleGenAI(
    model="gemini-3.5-flash",
)

resp = llm.complete("what are the policy objectives of the central bank?")
print(resp)