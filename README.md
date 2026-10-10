# 🧠 Next-Word Prediction using LSTM

A Deep Learning project that implements a **Next-Word Predictor** using a **Long Short-Term Memory (LSTM)** neural network trained on Sir Arthur Conan Doyle's *The Adventures of Sherlock Holmes*.

Given an input sequence of words, the model calculates a probability distribution across its vocabulary and predicts the most likely next word using a Dense Softmax layer.

---

## 📌 Project Overview

Next-word prediction is a foundational Natural Language Processing (NLP) task utilized in search engine autocompletion, predictive text keyboards, and assistive writing tools. This project demonstrates an end-to-end deep learning pipeline:
1. **Text Preprocessing & Tokenization**: Splitting text into tokens, mapping to unique integer IDs, and creating n-gram sequences.
2. **Sequence Padding**: Padding input sequences to a uniform length of 16 tokens (`padding='pre'`).
3. **Neural Network Architecture**: Word embeddings combined with an LSTM layer for sequence modeling and a Dense output layer for multi-class classification.
4. **Model Export**: Saving trained weights (`.keras`) and tokenizer mapping (`.pkl`) for direct inference.

---

## 🏗️ Model Architecture

```text
Input Sequence (up to 16 tokens)
           │
           ▼
Embedding Layer (Input Dim: 6305, Output Dim: 100)
           │
           ▼
LSTM Layer (150 Units)
           │
           ▼
Dense Output Layer (6305 Units, Softmax Activation)
           │
           ▼
Predicted Next Word ID (Argmax / Probability Distribution)
```

### Architectural Breakdown
| Layer | Type | Output Shape | Parameters | Description |
|---|---|---|---|---|
| **Layer 1** | Embedding | `(None, 16, 100)` | 630,500 | Maps word indices to 100-dimensional continuous dense vectors |
| **Layer 2** | LSTM | `(None, 150)` | 150,600 | Captures sequential context and temporal dependencies |
| **Layer 3** | Dense | `(None, 6305)` | 952,055 | Computes class probabilities via Softmax over 6,305 words |

- **Total Parameters**: **5,199,467** (~19.83 MB)
- **Trainable Parameters**: **1,733,155**

---

## 📊 Training Performance & Metrics

| Metric | Value |
|---|---|
| **Training Accuracy** | **90.72%** |
| **Training Loss** | **0.3295** |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Vocabulary Size** | 6,305 unique tokens |
| **Context Window** | 16 tokens (`maxlen = 16`, `padding='pre'`) |
| **Training Corpus** | *The Adventures of Sherlock Holmes* |

---

## 📁 Repository Structure

```text
├── next_word_predictor.ipynb                     # Jupyter notebook: preprocessing, training & evaluation
├── next_word_model.keras                         # Saved trained LSTM model file
├── tokenizer.pkl                                 # Fitted Keras Tokenizer
├── sherlock-holm.es_stories_plain-text_advs.txt   # Training text corpus
├── requirements.txt                              # Required dependencies
├── .gitignore                                    # Git ignore rules
└── README.md                                     # Project documentation
```

---

## 🚀 How to Use / Run Inference

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Predict the Next Word in Python
You can load the saved model and tokenizer directly without retraining:

```python
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model('next_word_model.keras')

# Input prompt
seed_text = "Holmes examined the locked door and noticed"

# Preprocess
tokens = tokenizer.texts_to_sequences([seed_text])[0]
padded_tokens = pad_sequences([tokens], maxlen=16, padding='pre')

# Predict next word
predictions = model.predict(padded_tokens, verbose=0)
next_word_id = np.argmax(predictions)
next_word = tokenizer.index_word.get(next_word_id, "")

print(f"Input: {seed_text}")
print(f"Predicted next word: {next_word}")
```

### 3. Training & Experimentation
Open and run the notebook:
```bash
jupyter notebook next_word_predictor.ipynb
```

---

## 🛠️ Technologies Used
- **Language**: Python 3.x
- **Deep Learning Framework**: TensorFlow / Keras 2.x
- **Data Manipulation**: NumPy
- **Interactive Development**: Jupyter Notebook
