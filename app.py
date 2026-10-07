from pathlib import Path

import streamlit as st
from youtube_transcript_api import (
    YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="YouTube Chatbot", page_icon="▶️", layout="wide")


# ---------- Styling ----------
st.markdown(
"""
<style>
.block-container {padding-top: 2rem; max-width: 900px;}
.hero {display:flex; align-items:center; gap:18px; padding:20px 24px; border-radius:20px; background:linear-gradient(135deg, rgba(255,0,0,0.20), rgba(255,0,0,0.04)); border:1px solid rgba(255,0,0,0.30); margin-bottom:20px;}
.hero h1 {margin:0; padding:0; font-size:2.1rem; background:linear-gradient(90deg,#ff3b3b,#ff9a9a); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p {margin:4px 0 0 0; opacity:0.75; font-size:0.95rem;}
[data-testid="stChatMessage"] {border-radius:16px; padding:12px 16px; border:1px solid rgba(128,128,128,0.22); background:rgba(128,128,128,0.08); margin-bottom:10px;}
section[data-testid="stSidebar"] {border-right:1px solid rgba(255,0,0,0.30);}
div.stButton > button {border-radius:12px;}
.side-brand {display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.1rem; margin-bottom:6px;}
</style>
""",
unsafe_allow_html=True,
)

LOGO = """<svg width="{w}" height="{h}" viewBox="0 0 48 34" xmlns="http://www.w3.org/2000/svg"><rect width="48" height="34" rx="9" fill="#FF0000"/><polygon points="19,10 19,24 32,17" fill="#FFFFFF"/></svg>"""


# ---------- Models (loaded only once) ----------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    # Multilingual model: Hindi transcript + English question dono handle karta hai
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return llm, embeddings


llm, embeddings = load_models()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

template = """You are a helpful assistant. Answer the user's question strictly using the YouTube
video transcript context below. The transcript may be in Hindi or another language, and the
question may be in a different language. Understand the context properly and answer in the
same language as the user's question.
If the answer is truly not present in the context, do not guess or use outside knowledge.
Simply reply with: "I'm sorry, but the video does not contain information about this."
context:{context}
user_query:{user_query}
Answer: """

prompt = PromptTemplate.from_template(template)


# ---------- Helpers ----------
def get_video_id(url):
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    return None


@st.cache_resource(show_spinner=False)
def get_retriever(video_id, language):
    # One folder per video and language, so different videos never mix
    # "_multi" = multilingual embeddings wala index
    db_dir = BASE_DIR / "chroma_db" / f"{video_id}_{language}_multi"

    if db_dir.exists():
        # Already embedded before, so reuse it
        db = Chroma(persist_directory=str(db_dir), embedding_function=embeddings)
    else:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=[language])
        text = " ".join(i.text for i in transcript)
        chunks = text_splitter.split_text(text)
        db = Chroma.from_texts(
            chunks, embedding=embeddings, persist_directory=str(db_dir)
        )

    base_retriever = db.as_retriever(search_kwargs={"k": 5})
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = {}     # key -> {url, video_id, language, messages}
if "current" not in st.session_state:
    st.session_state.current = None   # which video is currently open


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        f'<div class="side-brand">{LOGO.format(w=36, h=26)} YouTube Chatbot</div>',
        unsafe_allow_html=True,
    )

    st.subheader("➕ New video")
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    language = st.text_input("Language code", value="en", help="e.g. en, hi, es, fr")
    load_clicked = st.button("▶ Load video", type="primary", use_container_width=True)

    if load_clicked:
        video_id = get_video_id(url.strip())
        lang = language.strip()
        if not video_id:
            st.error("Invalid URL!")
        elif not lang:
            st.error("Please enter the language code.")
        else:
            try:
                with st.spinner("Fetching transcript and building the index..."):
                    get_retriever(video_id, lang)
                key = f"{video_id}_{lang}"
                if key not in st.session_state.history:
                    st.session_state.history[key] = {
                        "url": url.strip(),
                        "video_id": video_id,
                        "language": lang,
                        "messages": [],
                    }
                st.session_state.current = key
            except (TranscriptsDisabled, NoTranscriptFound):
                st.error("No transcript available for this video in that language.")
            except Exception as e:
                st.error(f"Could not fetch transcript (check your internet connection): {e}")

    st.divider()
    st.subheader("🕘 History")

    if not st.session_state.history:
        st.caption("No videos loaded yet.")

    for key, item in reversed(list(st.session_state.history.items())):
        questions = [m["content"] for m in item["messages"] if m["role"] == "user"]
        title = questions[0] if questions else item["video_id"]
        if len(title) > 28:
            title = title[:28] + "…"
        marker = "🔴" if key == st.session_state.current else "⚪"
        if st.button(f"{marker} {title}", key=f"hist_{key}", use_container_width=True):
            st.session_state.current = key
            st.rerun()
        st.caption(f"{item['video_id']} · {len(questions)} question(s)")

    if st.session_state.history:
        if st.button("🗑 Clear history", use_container_width=True):
            st.session_state.history = {}
            st.session_state.current = None
            st.rerun()


# ---------- Header ----------
st.markdown(
f"""
<div class="hero">
{LOGO.format(w=70, h=50)}
<div>
<h1>YouTube Chatbot</h1>
<p>Ask questions about any YouTube video and get answers straight from its transcript.</p>
</div>
</div>
""",
unsafe_allow_html=True,
)


# ---------- Welcome screen ----------
if st.session_state.current is None:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("### 1️⃣ Paste a link\nEnter a YouTube URL in the sidebar.")
    with c2:
        with st.container(border=True):
            st.markdown("### 2️⃣ Load it\nType the language code and click **Load video**.")
    with c3:
        with st.container(border=True):
            st.markdown("### 3️⃣ Ask away\nType your question in the chat box below.")
    st.stop()


# ---------- Chat ----------
item = st.session_state.history[st.session_state.current]

try:
    retriever = get_retriever(item["video_id"], item["language"])
except Exception as e:
    st.error(f"Could not load this video: {e}")
    st.stop()

with st.expander("📺 Watch video"):
    st.video(item["url"])

messages = item["messages"]

if not messages:
    st.info("Your video is ready! Ask your first question below 👇")

for message in messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "▶️"):
        st.markdown(message["content"])

user_query = st.chat_input("What do you want to know about this video?")

if user_query:
    messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="▶️"):
        try:
            with st.spinner("Thinking..."):
                fetch = retriever.invoke(user_query)

                context = ""
                for i in fetch:
                    context = context + i.page_content + "\n\n"

                final_prompt = prompt.format(context=context, user_query=user_query)
                answer = llm.invoke(final_prompt).content
        except Exception as e:
            answer = f"Something went wrong: {e}"

        st.markdown(answer)

    messages.append({"role": "assistant", "content": answer})
    st.rerun()