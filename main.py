from pathlib import Path

from youtube_transcript_api import (YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).parent

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
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


while True:

    url = input("\nPaste your Youtube url here (or type 'exit' to quit): ").strip()

    if url.lower() in ["exit", "quit"]:
        print("Exiting the program.")
        break

    if "watch?v=" in url:
        link = url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        link = url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    else:
        print("Invalid url!")
        continue

    video_language = input("Enter the language of the video (e.g. en, hi, es, fr): ").strip()
    if not video_language:
        print("Please enter the language code.")
        continue

    # Har video + language ka alag folder (app.py wala hi naam)
    db_dir = BASE_DIR / "chroma_db" / f"{link}_{video_language}_multi"

    try:
        if db_dir.exists():
            print("Pehle se embedded hai, purana index use kar raha hu...")
            db = Chroma(persist_directory=str(db_dir), embedding_function=embeddings)
        else:
            print("Fetching transcript and building the index...")
            transcript = YouTubeTranscriptApi().fetch(link, languages=[video_language])

            url_list = []
            for i in transcript:
                url_list.append(i.text)
            text = " ".join(url_list)

            chunks = text_splitter.split_text(text)

            db = Chroma.from_texts(
                chunks, embedding=embeddings, persist_directory=str(db_dir)
            )
    except (TranscriptsDisabled, NoTranscriptFound):
        print("No transcript available for this video in that language.")
        continue
    except Exception as e:
        print(f"Could not fetch transcript (check your internet connection): {e}")
        continue

    mmr_retriever = db.as_retriever(search_kwargs={"k": 5})
    retriever = MultiQueryRetriever.from_llm(
        retriever=mmr_retriever,
        llm=llm,
    )

    print("\nVideo ready!")

    while True:

        user_query = input("\nWhat do you want to know about this video? ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("Is video ke sawaal khatam. Naya video load karo ya 'exit' likho.")
            break

        if not user_query:
            continue

        try:
            fetch = retriever.invoke(user_query)

            context = ""
            for i in fetch:
                context = context + i.page_content + "\n\n"

            final_prompt = prompt.format(context=context, user_query=user_query)

            response = llm.invoke(final_prompt)
            print(response.content)
        except Exception as e:
            print(f"Something went wrong: {e}")

        print(">>>>>>>>>>Enter 'exit' or 'quit' to stop asking questions about this video.<<<<<<<<<")