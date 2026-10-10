import os
import time
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sherlock Smart Compose — AI Next-Word Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Styling: Modern Detective Terminal & Smart Compose
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    .app-title {
        font-family: 'Courier New', monospace;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #f0883e;
        margin-bottom: 0.2rem;
        border-bottom: 2px solid #30363d;
        padding-bottom: 8px;
    }

    .app-sub {
        font-family: 'Courier New', monospace;
        font-size: 1.0rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }

    .writing-canvas {
        background: #161b22;
        border: 2px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        font-family: 'Georgia', serif;
        font-size: 1.35rem;
        line-height: 2.0;
        color: #f0f6fc;
        min-height: 110px;
        margin-bottom: 1.2rem;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.4);
    }

    .typed-text {
        color: #e6edf3;
    }

    .cursor-blink {
        display: inline-block;
        width: 3px;
        height: 1.2rem;
        background-color: #f0883e;
        margin-left: 4px;
        animation: blink 1s infinite;
        vertical-align: middle;
    }

    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }

    .metric-badge {
        font-family: 'Courier New', monospace;
        background: #21262d;
        color: #79c0ff;
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 0.88rem;
        display: inline-block;
        margin-right: 8px;
        border: 1px solid #30363d;
    }

    .suggestion-label {
        font-family: 'Courier New', monospace;
        color: #f0883e;
        font-weight: bold;
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Trained Model and Tokenizer
# ---------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "next_word_model.keras")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.pkl")
MAX_SEQUENCE_LEN = 16

@st.cache_resource(show_spinner=False)
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None, None
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)
    index_to_word = tokenizer.index_word
    return model, tokenizer, index_to_word

with st.spinner("🕵️ Loading Sherlock Holmes Neural Engine..."):
    model, tokenizer, index_to_word = load_resources()

# ---------------------------------------------------------
# Core Next-Word Prediction Logic (Top-K Probabilities)
# ---------------------------------------------------------
def predict_next_words(text, top_k=5):
    """
    Given a seed text, computes the probability distribution over all 6,305 words
    and returns the top-K highest-probability next words with percentages.
    """
    if not text.strip():
        return []
    
    tokens = tokenizer.texts_to_sequences([text.strip()])[0]
    if not tokens:
        tokens = [0]
    
    tokens = pad_sequences([tokens], maxlen=MAX_SEQUENCE_LEN, padding="pre")
    start_time = time.time()
    predictions = model(tokens, training=False).numpy()[0]
    latency_ms = (time.time() - start_time) * 1000

    top_indices = np.argsort(predictions)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        word = index_to_word.get(idx, "")
        prob = float(predictions[idx]) * 100
        if word:
            results.append((word, prob))
            
    return results, latency_ms

def complete_phrase(text, num_words=4):
    """
    Automatically completes the current clause (3-5 words) using greedy top-1 prediction.
    """
    current_text = text.strip()
    generated = []
    
    for _ in range(num_words):
        tokens = tokenizer.texts_to_sequences([current_text])[0]
        if not tokens:
            break
        tokens = pad_sequences([tokens], maxlen=MAX_SEQUENCE_LEN, padding="pre")
        predictions = model(tokens, training=False).numpy()[0]
        best_idx = int(np.argmax(predictions))
        word = index_to_word.get(best_idx, "")
        if not word:
            break
        generated.append(word)
        current_text += " " + word
        
    return current_text, " ".join(generated)

# ---------------------------------------------------------
# App State & Interface
# ---------------------------------------------------------
st.markdown('<div class="app-title">🔍 Sherlock Smart Compose</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Deep Learning Next-Word Prediction & Smart Autocomplete Engine (Trained on Sherlock Holmes)</div>', unsafe_allow_html=True)

if model is None or tokenizer is None:
    st.error("⚠️ Model or Tokenizer files missing! Please ensure `next_word_model.keras` and `tokenizer.pkl` exist.")
    st.stop()

# Helper callbacks for updating state cleanly before widget instantiation
def update_sentence(new_sentence):
    st.session_state["editor_text"] = new_sentence

def add_suggested_word(word):
    current = st.session_state.get("editor_text", "").strip()
    st.session_state["editor_text"] = (current + " " + word).strip()

def run_smart_complete():
    current = st.session_state.get("editor_text", "").strip()
    new_text, _ = complete_phrase(current, num_words=4)
    st.session_state["editor_text"] = new_text

def clear_sentence():
    st.session_state["editor_text"] = ""

if "editor_text" not in st.session_state:
    st.session_state["editor_text"] = "Holmes looked at the"

# Quick Starter Chips
st.markdown("##### 📁 Quick Starters (Click to load):")
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.button("🚪 'The detective entered the'", on_click=update_sentence, args=("The detective entered the",), use_container_width=True)
with sc2:
    st.button("🔍 'Holmes looked at the'", on_click=update_sentence, args=("Holmes looked at the",), use_container_width=True)
with sc3:
    st.button("📜 'Watson and i examined the'", on_click=update_sentence, args=("Watson and i examined the",), use_container_width=True)
with sc4:
    st.button("🕯️ 'It was a very dark'", on_click=update_sentence, args=("It was a very dark",), use_container_width=True)

# Single Line Responsive Input with instant Predict button
col_in, col_btn = st.columns([4, 1])
with col_in:
    user_text = st.text_input(
        "✍️ Type your sentence (Press Enter or click Predict):",
        key="editor_text",
        placeholder="Type a detective sentence...",
        help="Press Enter to update suggestions instantly!"
    )
with col_btn:
    st.write("")
    st.write("")
    st.button("⚡ Predict", type="primary", use_container_width=True)

st.caption("💡 **Tip:** Type your sentence and press **Enter** (or click **⚡ Predict**) to see instant predictions!")

# Fetch Predictions for the current text
suggestions, latency = predict_next_words(st.session_state.get("editor_text", ""), top_k=5)

# ---------------------------------------------------------
# Smart Autocomplete Word Chips (Click to Append!)
# ---------------------------------------------------------
if suggestions:
    st.markdown('<div class="suggestion-label">✨ AI Top Predicted Next Words (Click any word to add it):</div>', unsafe_allow_html=True)
    cols = st.columns(len(suggestions))
    for i, (word, prob) in enumerate(suggestions):
        with cols[i]:
            st.button(f"➕ {word}\n\n({prob:.1f}%)", key=f"sug_btn_{i}", on_click=add_suggested_word, args=(word,), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Action Buttons: Complete Current Phrase & Reset
# ---------------------------------------------------------
col_action1, col_action2, col_clear = st.columns([2, 2, 1])

with col_action1:
    st.button("⚡ Smart Complete (Next 4 Words)", type="primary", on_click=run_smart_complete, use_container_width=True)

with col_action2:
    st.download_button(
        label="📥 Download Completed Text (.txt)",
        data=st.session_state.get("editor_text", ""),
        file_name="completed_sentence.txt",
        mime="text/plain",
        use_container_width=True
    )

with col_clear:
    st.button("🧹 Clear", on_click=clear_sentence, use_container_width=True)

# ---------------------------------------------------------
# Softmax Probability Distribution (Visual Explainability)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Neural Network Softmax Distribution")
st.caption("How the model's final Dense Softmax layer calculated probabilities across its 6,305-word vocabulary:")

if suggestions:
    for word, prob in suggestions:
        col_name, col_bar = st.columns([1, 4])
        with col_name:
            st.markdown(f"**`{word}`**")
        with col_bar:
            st.progress(prob / 100.0, text=f"{prob:.2f}% probability")

st.markdown(f"""
<div style="margin-top: 15px;">
    <span class="metric-badge">⚡ Latency: {latency:.1f} ms</span>
    <span class="metric-badge">📚 Vocabulary: 6,305 tokens</span>
    <span class="metric-badge">🎯 Training Accuracy: 90.72%</span>
    <span class="metric-badge">🧠 Architecture: Embedding + LSTM (150) + Softmax</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Project & Academic Evaluation Documentation
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📑 Project Report & Viva Notes (Why Next-Word Autocomplete is the Right Task)"):
    st.markdown("""
    ### 1. Problem Formulation
    - **Original Objective:** Next-Word Prediction on *The Adventures of Sherlock Holmes* using LSTM.
    - **Why Storytelling Failed:** Long text generation with basic LSTMs causes **Exposure Bias** (errors accumulate over long sequences).
    - **Why Smart Compose Succeeds:** It restricts the model to its true capability: **predicting the immediate next 1–3 words** given prior context.

    ### 2. Model Architecture
    - **Input Layer:** Sequence of up to 16 word IDs (`padding='pre'`).
    - **Embedding Layer:** 100-dimensional continuous dense vector representation.
    - **Recurrent Layer:** LSTM with 150 hidden units capturing sequence dependencies.
    - **Output Layer:** Dense with Softmax activation across 6,305 unique vocabulary tokens.

    ### 3. Key Metrics
    - **Training Accuracy:** `90.72%`
    - **Training Loss:** `0.3295`
    - **Inference Time:** `< 35 ms` per prediction step.
    """)
