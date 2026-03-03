import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from chatbot_utility import get_chapter_list
from get_yt_video import get_yt_video_link


# ---------------- ENV SETUP ---------------- #

load_dotenv()
DEVICE = os.getenv("DEVICE", "cpu")

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

subjects_list = ["Biology", "Physics", "Chemistry"]


# ---------------- VECTOR DB PATH ---------------- #

def get_vector_db_path(chapter, subject):
    if chapter == "All Chapters":
        return f"{parent_dir}/vector_db/class_12_{subject.lower()}_vector_db"
    return f"{parent_dir}/chapters_vector_db/{chapter}"


def get_vectorstore(selected_chapter, selected_subject):
    vector_db_path = get_vector_db_path(chapter=selected_chapter, subject=selected_subject)

    embeddings = HuggingFaceEmbeddings(
        model_kwargs={"device": DEVICE}
    )

    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )

    return vectorstore


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(
    page_title="StudyPal",
    page_icon="🌀",
    layout="centered"
)

st.title("📚 STUDY NEST")


# ---------------- SESSION STATE ---------------- #

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "video_history" not in st.session_state:
    st.session_state.video_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ---------------- SUBJECT SELECTION ---------------- #

selected_subject = st.selectbox(
    label="Select a Subject from class 12",
    options=subjects_list,
    index=None
)

if selected_subject:
    chapter_list = get_chapter_list(selected_subject) + ["All Chapters"]

    selected_chapter = st.selectbox(
        label=f"Select a Chapter from class 12 - {selected_subject}",
        options=chapter_list,
        index=0
    )

    if selected_chapter:
        if st.session_state.get("selected_chapter") != selected_chapter:
            st.session_state.vectorstore = get_vectorstore(
                selected_chapter,
                selected_subject
            )

        st.session_state.selected_chapter = selected_chapter


# ---------------- DISPLAY OLD MESSAGES ---------------- #

for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and idx < len(st.session_state.video_history):
            video_refs = st.session_state.video_history[idx]
            if video_refs:
                st.subheader("Video Reference")
                for title, link in video_refs:
                    st.info(f"{title}\n\nLink: {link}")


# ---------------- CHAT INPUT ---------------- #

user_input = st.chat_input("Ask AI")

if user_input and st.session_state.vectorstore:

    # Save user message
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    st.session_state.video_history.append(None)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        # 🔎 Retrieve Documents
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )

        docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 🧠 Build Prompt
        prompt = f"""
You are a helpful Class 12 study assistant.

Use the following context to answer clearly and concisely.

Context:
{context}

Question:
{user_input}

Answer:
"""

        # 🤖 Streaming LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            streaming=True
        )

        # 🔥 Slower Smooth Streaming
        for chunk in llm.stream(prompt):
            token = chunk.content
            if token:
                full_response += token
                placeholder.markdown(full_response + "▌")
                time.sleep(0.02)   # 🔥 CONTROL SPEED HERE (0.02–0.05 best range)

        # Final render (remove cursor)
        placeholder.markdown(full_response)

        final_answer = full_response

        # 🎥 Video References
        search_query = ", ".join([
            item["content"]
            for item in st.session_state.chat_history
            if item["role"] == "user"
        ])

        video_titles, video_links = get_yt_video_link(search_query)

        st.subheader("Video Reference")

        video_refs = []

        for i in range(min(3, len(video_titles))):
            st.info(f"{video_titles[i]}\n\nLink: {video_links[i]}")
            video_refs.append((video_titles[i], video_links[i]))

        # Save assistant message
        st.session_state.chat_history.append(
            {"role": "assistant", "content": final_answer}
        )
        st.session_state.video_history.append(video_refs)