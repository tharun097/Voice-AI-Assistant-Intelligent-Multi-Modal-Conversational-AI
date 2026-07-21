# app.py
import os
import uuid
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from groq import Groq

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# -----------------------------
# Environment
# -----------------------------
load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# -----------------------------
# Embeddings & tools
# -----------------------------
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = st.session_state.embeddings

tavily = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"), max_results=5)

base_tools = [tavily]
tool_dict = {t.name: t for t in base_tools}

# -----------------------------
# LLM
# -----------------------------
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        model="meta-llama/llama-prompt-guard-2-86m",
        api_key=os.getenv("GROQ_API_KEY"),
        streaming=True,
    )

llm = st.session_state.llm

# -----------------------------
# Session state
# -----------------------------
if "store" not in st.session_state:
    st.session_state.store = {}
if "chroma_indexes" not in st.session_state:
    st.session_state.chroma_indexes = {}
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = str(uuid.uuid4())
    st.session_state.messages[st.session_state.active_chat] = []

# -----------------------------
# Helpers
# -----------------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def create_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"context_name": "New chat"}
    st.session_state.messages[chat_id] = []
    return chat_id

def add_message(chat_id, role, content):
    st.session_state.messages[chat_id].append({"role": role, "content": content})
    hist = get_session_history(chat_id)
    if role == "user":
        hist.add_user_message(content)
    else:
        hist.add_ai_message(content)

def index_uploaded_document(chat_id, uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    vectordb.add_documents(chunks)
    st.session_state.chroma_indexes[chat_id] = vectordb
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    tool_name = f"docs_retriever_{chat_id[:8]}"
    docs_tool = create_retriever_tool(retriever, name=tool_name, description="Retriever for uploaded docs")
    tool_dict[docs_tool.name] = docs_tool
    return tool_name

def initiate_chat(chat_id, user_input):
    if "document" in user_input.lower() or "file" in user_input.lower():
        if chat_id in st.session_state.chroma_indexes:
            retriever = st.session_state.chroma_indexes[chat_id].as_retriever(search_kwargs={"k": 4})
            results = retriever.invoke(user_input)
            if results:
                return "\n\n".join([r.page_content for r in results])
            else:
                return "[No relevant content found in the uploaded document.]"
        else:
            return "[No document indexed yet. Please upload and index a file first.]"

    tools_for_chat = list(base_tools)
    for name, tool in tool_dict.items():
        if name.startswith("docs_retriever_") and chat_id[:8] in name:
            tools_for_chat.append(tool)

    llm_with_tools = llm.bind_tools(tools_for_chat)
    chatbot = RunnableWithMessageHistory(llm_with_tools, get_session_history)

    try:
        response = chatbot.invoke([HumanMessage(content=user_input)],
                                  config={"configurable": {"session_id": chat_id}})
    except Exception as e:
        return f"[Error calling LLM: {e}]"

    if not getattr(response, "tool_calls", None):
        return response.content

    tool_messages = []
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        # 🔧 Normalize include_images to boolean
        if "include_images" in tool_args:
            val = tool_args["include_images"]
            if isinstance(val, str):
                tool_args["include_images"] = val.lower() == "true"
            elif val is None:
                tool_args["include_images"] = False

        try:
            result = tool_dict[tool_name].invoke(tool_args)
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        except Exception as e:
            tool_messages.append(ToolMessage(content=f"[Tool error: {e}]", tool_call_id=tool_call["id"]))



    final_response = chatbot.invoke([HumanMessage(content=user_input), response, *tool_messages],
                                    config={"configurable": {"session_id": chat_id}})
    return final_response.content

# gTTS helper
def speak_text(text, selected_lang_code):
    tts = gTTS(text=text, lang=selected_lang_code)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name, format="audio/mp3")

# SpeechRecognition helper
def transcribe_live(audio_bytes, language="en"):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        audio_path = f.name

    with open(audio_path, "rb") as audio_file:

        transcription = groq_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="verbose_json",
            language=language
        )

    return transcription.text


# Name check helper
def check_for_name_question(user_input):
    lowered = user_input.strip().lower()
    if "your name" in lowered or "who are you" in lowered or "what is your name" in lowered:
        return "I am Nandu. And may I know your sweet name please?"
    elif lowered in ["hi", "hello", "hey", "hey nandu", "hello nandu"]:
        return "Hey, this is Nandu. May I know how can I help you today?"
    return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎙️ Voice Assistant Chatbot")

if "active_chat" not in st.session_state:
    st.session_state.active_chat = create_chat()

chat_id = st.session_state.active_chat

# Generate dynamic welcome message once
if "welcome_message" not in st.session_state:
    try:
        response = llm.invoke([HumanMessage(content="Generate a short, friendly welcome message for a chatbot named Nandu. ONLY return the message.")])
        st.session_state.welcome_message = response.content
    except Exception:
        st.session_state.welcome_message = "👋 Welcome! I’m Nandu, your assistant."

# Clear welcome once any message exists
if st.session_state.messages[chat_id]:
    st.session_state.welcome_message = None
    
# Show welcome message only if present AND no user messages yet
if not st.session_state.messages[chat_id]:
    st.markdown(
        f"""
        <div style="
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            height: 55vh;
        ">
            <h1 style="margin-bottom:10px;">👋 Welcome</h1>
            <p style="
                font-size: 22px;
                color: #666;
                max-width: 700px;
            ">
                {st.session_state.welcome_message}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Show past messages
for m in st.session_state.messages[chat_id]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Sidebar: voice + upload
# Full name → language code mapping
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Korean": "ko"
}

selected_lang_name = st.sidebar.selectbox(
    "Choose language",
    list(LANGUAGES.keys())
)
selected_lang_code = LANGUAGES[selected_lang_name]

# Sidebar: Voice Input
st.sidebar.header("🎤 Voice Input")

with st.sidebar:
    audio = mic_recorder(
        start_prompt="🎤 Speak",
        stop_prompt="⏹ Stop Recording",
        just_once=True,
        use_container_width=True,
        key="voice_input"
    )

if audio:

    with st.spinner("Transcribing voice..."):
        transcript = transcribe_live(
            audio["bytes"],
            selected_lang_code
        )

    if transcript:

        add_message(chat_id, "user", transcript)

        with st.chat_message("user"):
            st.markdown(transcript)

        special_reply = check_for_name_question(transcript)

        if special_reply:
            answer = special_reply
        else:
            answer = initiate_chat(chat_id, transcript)

        add_message(chat_id, "assistant", answer)

        with st.chat_message("assistant"):
            st.markdown(answer)

        speak_text(answer, selected_lang_code)


st.sidebar.header("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
if uploaded_file and st.sidebar.button("Index Document"):
    st.session_state.welcome_message = None  # clear welcome
    tool_name = index_uploaded_document(chat_id, uploaded_file)
    st.sidebar.success(f"Indexed {uploaded_file.name} as tool {tool_name}")

prompt = st.chat_input("Type your message")

if prompt:
    add_message(chat_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    special_reply = check_for_name_question(prompt)
    if special_reply:
        answer = special_reply
    else:
        answer = initiate_chat(chat_id, prompt)

    add_message(chat_id, "assistant", answer)
    with st.chat_message("assistant"):
        st.markdown(answer)
    speak_text(answer, selected_lang_code)
