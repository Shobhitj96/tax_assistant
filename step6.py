import streamlit as st
import json
import os
import time
import tempfile
import base64
from dotenv import load_dotenv
import google.generativeai as genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Setup ---
st.set_page_config(page_title="Indian Income Tax Chatbot", page_icon="🧾", layout="wide")
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
TAX_DATA_PATH = os.getenv("TAX_JSON_PATH")
CHAT_HISTORY_PATH = "chat_history.json"
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
TOP_N_RESULTS = 5

standard_safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
]

# --- Initialize Gemini ---
if not API_KEY:
    st.error("GEMINI_API_KEY not found.")
    st.stop()

try:
    genai.configure(api_key=API_KEY)
    genai.GenerativeModel('gemini-1.5-flash').generate_content("Hello", safety_settings=standard_safety_settings)
except Exception as e:
    st.error(f"Failed to configure Gemini: {e}")
    st.stop()

# --- Load Tax Data ---
tax_data, bm25, sentence_model, corpus_embeddings = [], None, None, None
if TAX_DATA_PATH and os.path.exists(TAX_DATA_PATH):
    with open(TAX_DATA_PATH, 'r', encoding='utf-8') as f:
        tax_data = json.load(f)
    corpus = [item.get("raw_text", "") for item in tax_data]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    @st.cache_resource
    def load_model(model_name):
        return SentenceTransformer(model_name)

    sentence_model = load_model(SENTENCE_TRANSFORMER_MODEL)

    if sentence_model:
        @st.cache_resource
        def encode_corpus(corpus_data, _model):
            return _model.encode(corpus_data)

        corpus_embeddings = encode_corpus(corpus, sentence_model)

# --- Helper Functions ---

def search_tax_data_advanced(query, data, bm25_model, sentence_model, corpus_embeddings, top_n=TOP_N_RESULTS):
    if not data:
        return ""
    query_lower = query.lower()
    bm25_scores = bm25_model.get_scores(query_lower.split()) if bm25_model else np.zeros(len(data))
    semantic_scores = np.zeros(len(data))
    if sentence_model and corpus_embeddings is not None:
        query_embedding = sentence_model.encode(query_lower)
        semantic_scores = np.dot(query_embedding, corpus_embeddings.T)
    combined_scores = 0.5 * bm25_scores + 0.5 * semantic_scores
    top_indices = np.argsort(combined_scores)[-top_n:]
    relevant_sections = [data[i] for i in reversed(top_indices)]
    return json.dumps(relevant_sections, indent=2)

def get_gemini_response(prompt, tax_context="", language="English"):
    model = genai.GenerativeModel('gemini-1.5-flash')

    full_prompt = f"""
You are an AI assistant specializing in Indian Income Tax law. Your primary goal is to provide clear, precise, and easy-to-understand information based on the provided context or your knowledge, while strictly adhering to the following rules:

**Core Instructions:**
- Regardless of the source (provided context or your knowledge), all fetched content must go through you to:
    - Refine sentence structure.
    - Present in a clear, precise, and easy-to-understand format. Don't just give users the answer to their query in terms of sections, give them results.
    - Maintain tone, compliance, and legal accuracy regarding Indian taxation.
    - In case user asked any generic advice regarding to Income tax eg: I am freelancer how to save tax, i have capital gains on sale of second residential property and where to invest the capital gain amount to save the tax etc. use your knowledge.
    - In case you feel like, json knowledge base does not fully asnwer user query you are free to add your knowledge/sections to the same. For eg: User ask deduction for home loan and json knowledge base only giving reference to Sec 24(b), you please add Sec 80C,Sec 80EE, Sec 80EEA to make it comprehensive and complete.  
- Answer in {language} as requested by the user. If the user's query does not clearly indicate a language, default to English.
- When providing information about a topic, synthesize the relevant details from all provided tax context sections. If multiple sections are relevant (e.g., different deductions for the same topic), mention and explain each relevant section based on the context provided.

**Clarification Protocol:**
- If the user's query is vague or incomplete, first attempt to provide a general answer based on the most likely interpretation or common information related to the topic.
- ONLY if a more precise answer is needed or the query is highly ambiguous, suggest what additional details the user could provide for a more accurate response. Frame these as helpful suggestions, not demands.

**Content Boundaries & Safety:**
- Never answer questions unrelated to Indian taxation.
- If asked about politics, personal matters, or other domains, respond: "This is outside my domain."
- If a user asks something illegal or unethical, respond: "I cannot assist with this request."

**Structured Answer Format (use this where applicable):**
- **Section Number**: ...
- **Section Explanation**: ...
- **Deduction Limit / Tax Rate**: ...
- **Qualifying Investments / Items**: ...
- **Exclusions**: ...
- **Examples**: ...
- **Claim Process**: ...
- **Additional Points**: ...
- **Disclaimer**: "This is an AI-generated answer, not tax or legal advice. Kindly consult a tax advisor or Chartered Accountant expert in direct tax law matters before making any decisions."

---

**Provided Tax Data Context (if available):**
{tax_context if tax_context else "No specific tax data context found."}

---

**User Query:**
{prompt}
    """

    try:
        response = model.generate_content(full_prompt, safety_settings=standard_safety_settings)
        return response.text
    except Exception as e:
        return f"Error contacting AI: {e}"

def save_history():
    with open(CHAT_HISTORY_PATH, "w", encoding='utf-8') as f:
        json.dump(st.session_state.chat_history, f)

def load_history():
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, "r", encoding='utf-8') as f:
            st.session_state.chat_history = json.load(f)
    else:
        st.session_state.chat_history = []

# --- Streamlit App ---

st.title("🧾 Indian Income Tax Chatbot (Enhanced)")
st.caption("Built with 💬 Gemini + BM25 + Sentence Transformers")

# Language Selector
language = st.selectbox("Choose Language", options=["English", "Hindi"], index=0)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
else:
    # ✅ Show chat history within same session
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

# Text Input
text_input = st.chat_input("Ask me about Income Tax...")
user_input = None

if text_input:
    user_input = text_input

# Handle User Query
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    relevant_data = search_tax_data_advanced(user_input, tax_data, bm25, sentence_model, corpus_embeddings)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = get_gemini_response(user_input, relevant_data, language)
        response_sentences = response_text.split('. ')
        full_response = ""
        response_placeholder = st.empty()
        for sentence in response_sentences:
            full_response += sentence.strip() + ". "
            response_placeholder.markdown(full_response + "▌")  # Typing cursor
            time.sleep(0.1)
        response_placeholder.markdown(full_response)  # Final clean output

    st.session_state.chat_history.append(("assistant", full_response))
    save_history()
