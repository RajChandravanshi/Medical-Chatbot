from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# LangChain imports
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from src.helper import download_embeddings_model
from src.prompt import system_prompt

# ----------------------------------
# ENV SETUP
# ----------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ----------------------------------
# FLASK APP
# ----------------------------------
app = Flask(__name__)

# ----------------------------------
# EMBEDDINGS & VECTOR STORE
# ----------------------------------
embeddings = download_embeddings_model()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ----------------------------------
# LLM
# ----------------------------------
chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# ----------------------------------
# PROMPT
# ----------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

# ----------------------------------
# MEMORY
# ----------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ----------------------------------
# RAG + MEMORY CHAIN
# ----------------------------------
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False
)

# ----------------------------------
# ROUTES
# ----------------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]
    print("User:", user_message)

    response = rag_chain.invoke({"question": user_message})

    bot_response = response["answer"]
    print("Bot:", bot_response)

    return bot_response


# ----------------------------------
# RUN APP
# ----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
