%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import plotly.express as px

# -------------------------------
# 🔐 LOGIN SYSTEM
# -------------------------------
USERS = {"admin": "1234", "user": "abcd"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------------------
# 🎨 UI
# -------------------------------
st.set_page_config(layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:#00C9A7;'>📡 ZENDS AI Intelligence System</h1>", unsafe_allow_html=True)

# -------------------------------
# 📊 LOAD DATA (ROBUST)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("sample_Zends_synthetic_dataset.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

# Detect columns dynamically
sentiment_col = [c for c in df.columns if "sentiment" in c][0]
category_col = [c for c in df.columns if "service" in c or "category" in c][0]

# -------------------------------
# 🤖 SENTIMENT MODEL
# -------------------------------
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_model = load_sentiment()

def get_sentiment(text):
    label = sentiment_model(text)[0]['label']
    return {"LABEL_2":"Positive","LABEL_1":"Neutral","LABEL_0":"Negative"}[label]

# -------------------------------
# 🧠 TOPIC CLASSIFICATION
# -------------------------------
def classify_topic(text):
    text = text.lower()
    if "internet" in text:
        return "Broadband Service"
    elif "bill" in text:
        return "Billing & Payments"
    elif "network" in text:
        return "Mobile Network"
    elif "app" in text:
        return "Mobile App Issues"
    elif "sim" in text or "activation" in text:
        return "Service Activation"
    else:
        return "Customer Support"

# -------------------------------
# ⚡ FAST RAG
# -------------------------------
@st.cache_resource
def load_rag():
    reader = PdfReader("zends_communications_telecom_knowledge_base.pdf")
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t
    
    chunks = [text[i:i+300] for i in range(0, len(text), 300)]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb = model.encode(chunks)
    
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))
    
    return model, index, chunks

embed_model, index, chunks = load_rag()

def retrieve(query, k=2):
    q = embed_model.encode([query])
    D, I = index.search(q, k)
    return [chunks[i] for i in I[0]]

# -------------------------------
# 🤖 LLM
# -------------------------------
@st.cache_resource
def load_llm():
    return pipeline("text-generation",
                    model="google/flan-t5-small",
                    max_length=200,
                    do_sample=True,
                    temperature=0.7)

llm = load_llm()

def generate_response(query):
    context = " ".join(retrieve(query))[:500]

    prompt = f"""
    You are a telecom support assistant.

    Customer Issue:
    {query}

    Context:
    {context}

    Provide:
    - A polite acknowledgment
    - Possible cause
    - 2-3 steps solution
    - Closing message
    """

    return llm(prompt)[0]['generated_text']

# -------------------------------
# 💬 CHAT INTERFACE
# -------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Enter your issue...")

if user_input:
    sentiment = get_sentiment(user_input)
    category = classify_topic(user_input)
    response = generate_response(user_input)

    full_response = f"""
📊 Sentiment: {sentiment}
📂 Category: {category}

🤖 Response:
{response}
"""

    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("bot", full_response))

for role, msg in st.session_state.chat:
    st.chat_message("user" if role=="user" else "assistant").write(msg)

# -------------------------------
# 📊 DASHBOARD (FULLY FIXED)
# -------------------------------
st.sidebar.title("📊 Analytics Dashboard")

# Sentiment chart
sent_counts = df[sentiment_col].value_counts().reset_index()
sent_counts.columns = ['sentiment', 'count']

sent_fig = px.bar(
    sent_counts,
    x='sentiment',
    y='count',
    color='sentiment',
    title="Sentiment Distribution"
)

# Category chart
cat_fig = px.pie(
    df,
    names=category_col,
    title="Service Categories"
)

st.sidebar.plotly_chart(sent_fig)
st.sidebar.plotly_chart(cat_fig)