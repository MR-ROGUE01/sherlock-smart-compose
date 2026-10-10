# 🔍 Sherlock Smart Compose — Deep Learning Next-Word Predictor

An interactive, deep learning-powered next-word autocomplete system built using a **Long Short-Term Memory (LSTM)** neural network and **Streamlit**. Trained on Sir Arthur Conan Doyle's *The Adventures of Sherlock Holmes*, this application mimics Google Smart Compose and mobile predictive keyboards for Victorian detective literature.

---

## 📸 Key Features

- **⚡ Real-Time Next-Word Prediction**: Calculates Softmax probability distributions across a 6,305-word vocabulary in under 35 milliseconds.
- **🎯 Click-to-Insert Word Chips**: Displays Top-5 predicted candidate words with confidence percentages. Clicking any word automatically inserts it into the sentence.
- **🚀 Smart Clause Completion**: Predicts and appends the next 3–4 words to finish the current phrase or clause cleanly.
- **📊 Explainable AI (Softmax Visualization)**: Live probability breakdown bars show how the neural network evaluated alternative next-word candidates.
- **📁 Detective Scene Starters**: One-click quick-start prompts from classic Sherlock mystery cases.
- **📥 One-Click Export**: Download completed sentences and notes as a `.txt` file.

---

## 🧠 Model Architecture & Specifications

```text
[Input Text (up to 16 tokens)]
           │
           ▼
[Tokenizer (6,305 Vocabulary)]
           │
           ▼
[Pre-Padding (maxlen = 16)]
           │
           ▼
[Embedding Layer (100 Dimensions)]
           │
           ▼
[LSTM Layer (150 Hidden Units)]
           │
           ▼
[Dense Output Layer + Softmax (6,305 Classes)]
           │
           ▼
[Top-5 Probabilities & Next-Word Predictions]
```

### Technical Summary
| Metric / Component | Specification |
|---|---|
| **Architecture** | Sequential (Embedding → LSTM → Dense Softmax) |
| **LSTM Units** | 150 |
| **Embedding Dimension** | 100 |
| **Sequence Length** | 16 tokens (`padding='pre'`) |
| **Vocabulary Size** | 6,305 unique tokens |
| **Training Dataset** | *The Adventures of Sherlock Holmes* |
| **Training Accuracy** | **90.72%** |
| **Training Loss** | **0.3295** |
| **Inference Latency** | ~25–35 ms per step |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/sherlock-smart-compose.git
cd sherlock-smart-compose
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
├── app.py                                        # Streamlit web application
├── next_word_model.keras                         # Trained Keras LSTM model (5.2M params)
├── tokenizer.pkl                                 # Fitted Keras Tokenizer
├── next_word_predictor.ipynb                     # Model training & experimentation notebook
├── sherlock-holm.es_stories_plain-text_advs.txt   # Training text corpus
├── requirements.txt                              # Python package dependencies
├── .gitignore                                    # Git ignore rules
└── README.md                                     # Project documentation
```

---

## 🛠️ Tech Stack
- **Deep Learning Framework**: TensorFlow / Keras 2.x
- **Web Interface**: Streamlit
- **Preprocessing**: Keras Tokenizer & Sequence Padding
- **Language**: Python 3.x
