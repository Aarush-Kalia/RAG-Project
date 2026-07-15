from dotenv import load_dotenv
from google import genai

load_dotenv()

with open("notes.txt", "r") as file:
    content = file.read()

question = "How much does the Gray Whale cost?"
prompt = f"Answer the question using only the notes below.\n\nNOTES:\n{content}\n\nQUESTION: {question}"

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input=prompt
)
print(interaction.output_text)