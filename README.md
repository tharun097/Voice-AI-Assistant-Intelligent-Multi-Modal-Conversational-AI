# Voice-AI-Assistant-Intelligent-Multi-Modal-Conversational-AI

## 📖 Project Description

Developed a production-inspired AI assistant capable of natural
conversations, document question answering, real-time web search, and
multilingual voice interactions.

The application combines **Groq Llama-4**, **LangChain Tool Calling**,
**Retrieval-Augmented Generation (RAG)**, **Chroma Vector Database**,
and **Speech Recognition** to deliver context-aware responses from both
uploaded documents and live internet sources.

The assistant supports dynamic document indexing, conversational memory,
multiple chat sessions, multilingual speech-to-text/text-to-speech, and
intelligent routing between Large Language Models, document retrieval,
and web search tools.

------------------------------------------------------------------------

# 🎯 Key Features

## 🤖 Conversational AI

-   Groq Llama-4 Scout
-   Multi-turn conversation
-   Session memory
-   Multiple chat sessions
-   Personalized greeting

## 📄 Retrieval-Augmented Generation (RAG)

-   PDF upload
-   TXT upload
-   Automatic document chunking
-   HuggingFace Embeddings (all-MiniLM-L6-v2)
-   Chroma Vector Database
-   Semantic similarity search

## 🌐 Tool Calling

-   Tavily Search integration
-   LangChain Tool Calling
-   Dynamic Retriever Tool creation
-   Automatic tool execution

## 🎤 Voice AI

-   Speech Recognition
-   Google Speech API
-   gTTS (Text-to-Speech)
-   Multilingual support
-   Voice conversation

## 💬 Memory

-   RunnableWithMessageHistory
-   Conversation history
-   Independent chat sessions

## 🎨 Streamlit UI

-   Modern Chat Interface
-   Voice Button
-   PDF Upload
-   Chat History
-   Sidebar Controls

------------------------------------------------------------------------

# 🏗 Project Architecture

``` text
                    User
                      │
                      ▼
              Streamlit Web UI
                      │
     ┌────────────────┼────────────────┐
     │                │                │
     ▼                ▼                ▼
 Voice Input     Text Input      File Upload
     │                │                │
     ▼                ▼                ▼
SpeechRecognition     │       PDF/TXT Loader
     │                │                │
     └──────────────┬─┴───────────────┘
                    ▼
             Query Processing
                    │
                    ▼
          Intent / Tool Routing
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
 Normal Chat     RAG Search    Web Search
      │             │             │
      ▼             ▼             ▼
 Groq LLM     Chroma DB     Tavily Search
      │             │             │
      └─────────────┼─────────────┘
                    ▼
             Final AI Response
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
   Streamlit UI             Voice Output
                                 │
                               gTTS
```

------------------------------------------------------------------------

# 🏛 High-Level Architecture

``` text
                ┌──────────────────────────┐
                │      Streamlit UI        │
                └─────────────┬────────────┘
                              │
               User Question / Voice Input
                              │
                              ▼
                 ┌────────────────────┐
                 │   Chat Controller   │
                 └─────────┬───────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    Session Memory      Tool Router      Greeting
          │                │
          │                ▼
          │     ┌──────────┴──────────┐
          │     ▼                     ▼
          │   Tavily             Retriever
          │                         │
          │                    Chroma DB
          │                         │
          └──────────────┬──────────┘
                         ▼
                    Groq LLM
                         │
                         ▼
                  Assistant Reply
                         │
              ┌──────────┴─────────┐
              ▼                    ▼
         Streamlit UI          Voice Output
```

------------------------------------------------------------------------

# ⚙ Technology Stack

  Category               Technology
  ---------------------- -------------------------
  LLM                    Groq Llama-4 Scout
  Framework              LangChain
  RAG                    ChromaDB
  Embeddings             all-MiniLM-L6-v2
  Search                 Tavily Search
  UI                     Streamlit
  Voice                  SpeechRecognition, gTTS
  Vector Database        Chroma
  Document Loader        PyPDFLoader
  Programming Language   Python

------------------------------------------------------------------------

# 💡 Skills Demonstrated

-   Agentic AI
-   Large Language Models (LLMs)
-   Retrieval-Augmented Generation (RAG)
-   Semantic Search
-   Vector Databases
-   Prompt Engineering
-   LangChain
-   Tool Calling
-   Voice AI
-   Generative AI
-   Conversational AI
-   Streamlit
-   Session Management

------------------------------------------------------------------------

# 🚀 Future Enhancements

-   Multi-agent workflows
-   Image understanding
-   OCR support
-   Authentication
-   Persistent cloud database
-   Advanced memory management
