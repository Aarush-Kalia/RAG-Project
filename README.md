# Notes RAG

A retrieval-augmented generation system that answers questions from a private document collection rather than from a model's general knowledge. It retrieves the passages most semantically relevant to a question, constrains the model to answer from only those passages, and cites the file and page each one came from.

## What it does

The system runs in two stages. Ingestion processes a folder of PDFs and text files once, converting them into a searchable vector index stored on disk. Querying happens in a chat loop, and each question retrieves only the passages relevant to it, so the cost of a question stays flat regardless of how large the collection grows. When the retrieved passages don't contain the answer, the model is instructed to say so rather than fill the gap with its own knowledge, which is the failure mode that makes an ungrounded chatbot unreliable on private material.

I built this to understand the retrieval architecture directly, since it's the pattern underlying most production tools that answer questions over a user's own documents.

## How it works

There are two scripts because ingestion and querying happen at different times and cost very different amounts of work.

**ingest.py** builds the index, and you only run it once, or again whenever you add new documents. It opens a persistent ChromaDB client, reads every file in the data folder, splits all that text into chunks of about 1024 tokens each, runs every chunk through an embedding model, and stores every chunk along with its embedding in the vector database. An embedding is a list of numbers, 768 of them for the model I used, that represents the meaning of a piece of text, and the model is trained so that text with similar meanings gets similar vectors. My three lecture PDFs plus a text file came out to 24 chunks.

**chat.py** handles the queries, and it runs every time you ask something. It embeds your question with the same embedding model, and it has to be the same one or the question and the chunks end up in different vector spaces and the search is meaningless. Then it does a similarity search, which is where the vector database finds the stored chunks whose vectors are nearest to the question's vector. That's the retrieval step. Then it takes the text of those retrieved chunks and inserts them into a prompt template along with the question, which is the augmentation step, and sends that prompt to Gemini for generation. It also prints the source file, the page, and the similarity score for every chunk it used.

The reason for building all of this instead of just putting every document into the prompt, which is called stuffing, is cost, latency, and the context window. Stuffing means you pay input tokens for your entire document set on every single question, the model has to process all of it before answering, and past a certain size it physically will not fit in the context window and the request gets rejected. Retrieval only sends the few chunks that matter, so the tokens per question stay roughly constant no matter how large the document set gets.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
# Create a file named .env containing:
#   GOOGLE_API_KEY=your_key_here
# (free key at https://aistudio.google.com)

# 4. Put your PDFs and text files into a folder named data/

# 5. Build the index (run again whenever documents change)
python ingest.py

# 6. Ask questions
python chat.py
```

The .env file holds the API key so it never ends up in the code, and it's listed in .gitignore along with the venv and the database folder so none of it gets committed.

## Tech stack

- **Python**
- **LlamaIndex** as the RAG framework, handling the chunking, the embedding pipeline, and the query engine. It's modular, so the LLM and the vector store are adapters that plug into the core and can be swapped without changing anything else.
- **ChromaDB** as the vector database. It stores the chunks and their embeddings and does the nearest neighbor search, which a normal database can't do because it only handles exact matches, not "what is closest to this point."
- **Google Gemini** as the LLM that generates the answers, called through the LlamaIndex adapter rather than Google's SDK directly.
- **HuggingFace bge-base-en-v1.5** as the embedding model. It runs locally, so embeddings cost nothing and need no API key. The only thing that needs a key is generation.

## Notes from building

**Three different API errors, and they all meant different things.** I got a 404 first, which turned out to be a deprecated model name that I had copied from a docs page written before it was discontinued. Then I got 503s, which meant the model was overloaded on Google's side and there was nothing wrong with my request at all, and the traceback showed the retry library attempting it a few times before giving up. Then I got a 429, RESOURCE_EXHAUSTED, which meant I had hit the free tier quota, and the error metadata showed the limit was per model per day, so switching models gave me a separate bucket. The useful thing I took from this is that 4xx errors mean my request was wrong and I need to fix something, while 5xx errors mean their servers had the problem and I should retry or route around it. I also learned that one .query() call can make more than one request to the model, because the response synthesizer refines its answer across the retrieved chunks, so my quota went faster than the number of questions I asked.

**Text files don't have page numbers.** My source citations worked until a query retrieved a chunk from a .txt file instead of a PDF, and then it crashed with a KeyError on page_label, because the metadata dictionary for a text chunk doesn't have that key. I fixed it by using .get() with a fallback instead of a direct key lookup. It worked on the data I happened to test with and broke on the data I didn't, which seems like a pretty common way for things to break.

**The similarity scores carry real information.** For questions my documents could answer, the retrieved chunks scored around 0.4 to 0.5. For questions they couldn't, like the capital of a country, the top scores dropped to around 0.25 to 0.31 every time. Retrieval always returns the nearest chunks even when nothing is actually relevant, so the model only refused because of the prompt template, which explicitly tells it to answer using the context and to say when the answer isn't there. But the score gap is a second, separate signal that the documents have nothing relevant, and it's consistent enough that a threshold could catch that case before the model is ever called.