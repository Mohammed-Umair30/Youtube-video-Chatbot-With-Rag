# Youtube-video-Chatbot-With-Rag
Youtube video Chatbot with Rag
# 🎬 YouTube Video Chatbot using RAG (LangChain + Ollama + Streamlit)

This is a Retrieval-Augmented Generation (RAG)-based chatbot application that allows users to interactively ask questions about any YouTube video using only its **video ID**. Built using **LangChain**, **Ollama's TinyLlama**, **ChromaDB**, and **Streamlit**, this app demonstrates a scalable and intelligent way to summarize and understand video content.

---

## 🚀 Features

✅ Accepts any YouTube video ID dynamically  
✅ Uses `YouTubeTranscriptAPI` to fetch transcript (if captions are enabled)  
✅ Semantic chunking for context-aware document splitting  
✅ MMR (Maximal Marginal Relevance) based retrieval  
✅ Embedding via `sentence-transformers/all-MiniLM-L6-v2`  
✅ Local LLM via `Ollama` using `tinyllama`  
✅ Professional prompt formatting  
✅ Multi-turn conversation memory  
✅ Clean, modern **Streamlit UI** for real-time interaction  
✅ Answers include **citation context** from the source chunks

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/) with [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)

---

## 🧠 How it Works

1. User enters a YouTube video ID.
2. Transcript is fetched (if captions are available).
3. Transcript is semantically chunked using `SemanticChunker`.
4. Each chunk is embedded and stored in **ChromaDB**.
5. A question is asked via UI → Relevant chunks are retrieved via MMR.
6. Prompt is dynamically generated using context and question.
7. Answer is generated using `TinyLlama` model (via Ollama).
8. The answer is displayed along with source citations.
9. Multi-turn history is maintained using LangChain memory.

---

## 💻 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/youtube-rag-chatbot.git
cd youtube-rag-chatbot
