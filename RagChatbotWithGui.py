import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer

# -----------------
# Setup embeddings
# -----------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit config
st.set_page_config(page_title="🎥 YouTube RAG Chatbot", layout="centered")
st.title("🎬 YouTube Video Chatbot with RAG (LangChain + Ollama)")

# Custom CSS for style
st.markdown(
    """
    <style>
        .stChatMessage {
            background-color: #f0f2f6;
            color: #000000;
            border-radius: 12px;
            padding: 10px;
        }
        .stButton button {
            background-color: #3b82f6;
            color: white;
            border-radius: 10px;
        }
        .main .block-container {
            max-width: 800px;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------
# Video input
# -----------------
video_id = st.text_input("Enter YouTube Video ID", value="Gfr50f6ZBvo", help="Paste only the video ID (not full URL)")

if "vector_store" not in st.session_state or st.session_state.get("current_video_id") != video_id:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        transcript = transcript.replace("\n", " ").replace("  ", " ")
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        st.stop()

    # Recursive splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Vector store
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=f"./chroma_db_{video_id}", collection_name=f"youtube_{video_id}")

    st.session_state.vector_store = vector_store
    st.session_state.retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.5})
    st.session_state.current_video_id = video_id

# -----------------
# Memory (multi-turn)
# -----------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# LLM
llm = ChatOllama(
    model="tinyllama:latest",
    model_kwargs={"num_predict": 500}
)

# ✅ Fixed professional prompt
prompt = PromptTemplate(
    template="""
You are an expert AI assistant helping viewers understand and summarize YouTube content clearly and professionally.

Use only the provided transcript context to answer the user's question precisely.

If summarizing, write a concise, structured summary in your own words (never copy verbatim). Use proper grammar and avoid run-on or ambiguous sentences.

Be specific and cite chunk IDs if possible.

---
Transcript context:
{context}
---
Question:
{question}

Answer:
""",
    input_variables=['context', 'question']
)


# -----------------
# Runnable chain
# -----------------
def format_docs(docs):
    joined = "\n\n".join(doc.page_content for doc in docs)
    ids = ", ".join([doc.metadata.get("source", "N/A") for doc in docs])
    return f"{joined}\n\n[Cited chunks: {ids}]"

parallel_chain = RunnableParallel({
    'context': st.session_state.retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# -----------------
# Chat UI
# -----------------
st.subheader("Ask a question about the video")
user_question = st.text_input("Your question", placeholder="E.g., Can you summarize the main ideas?")

if st.button("Ask and Polish"):
    if user_question.strip() != "":
        answer = main_chain.invoke(user_question)
        st.session_state.memory.chat_memory.add_user_message(user_question)
        st.session_state.memory.chat_memory.add_ai_message(answer)
        st.success("Answer (Polished):")
        st.markdown(f"<div class='stChatMessage'>{answer}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a question before clicking 'Ask and Polish'.")

# Show chat history
if st.session_state.memory.chat_memory.messages:
    st.subheader("Conversation history")
    for msg in st.session_state.memory.chat_memory.messages:
        if msg.type == "human":
            st.markdown(f"**You:** {msg.content}")
        else:
            st.markdown(f"**Bot:** {msg.content}")
