# 🪶 Scroll — Emotion-Weighted News Summarizer  
### *An AI-powered, fuzzy-logic-based, emotion-aware news summarization and personalization system*

---

## 🌍 Overview  

**Scroll** is an advanced **AI-powered news summarization platform** that combines **transformer-based summarization, emotion detection, fuzzy logic weighting, and topic inference** to deliver highly contextual and emotionally intelligent news summaries.  

Unlike traditional summarizers that treat all stories equally, **Scroll** evaluates the *emotional intensity* and *contextual relevance* of each news cluster — surfacing stories that are not just recent, but impactful. It integrates **DeBERTa**, **DistilBART**, **VADER**, and **SentenceTransformer embeddings**, forming a comprehensive, adaptive news analysis pipeline.

---

## 🚀 Key Features  

- 🧠 **Emotion-Weighted Summarization:**  
  Uses fuzzy logic to compute the emotional weight of each cluster, highlighting socially or sentimentally significant news.  

- 🔍 **Transformer-Based Summaries:**  
  Employs `sshleifer/distilbart-cnn-12-6` for high-quality abstractive summarization.  

- 💬 **Emotion & Sentiment Detection:**  
  Combines `j-hartmann/emotion-english-distilroberta-base` for emotion distribution with `VADER` for sentiment polarity.  

- 📡 **AI-Driven Clustering:**  
  Groups related stories using semantic embeddings (`all-MiniLM-L6-v2`) and hierarchical clustering.  

- 🧮 **Fuzzy Logic Prioritization:**  
  Evaluates news importance using fuzzy membership functions for low, medium, and high emotional intensities.  

- 🗂 **Topic Classification:**  
  Infers topics such as Politics, Science, Health, and Technology using **DeBERTa-v3-large** and semantic similarity.  

- 🔊 **Text-to-Speech Summaries:**  
  Converts summaries into natural audio briefings using **Google Text-to-Speech (gTTS)**.  

- 🖥 **Interactive Gradio Interface:**  
  Allows users to select preferred topics, listen to summaries, and explore emotionally ranked news clusters.  

---

## 🧠 System Architecture  

RSS Feeds ──► Text Extraction (Newspaper3k)
│
▼
Preprocessing & Cleaning
│
▼
Sentence Embedding (MiniLM-L6-v2)
│
▼
Clustering (Agglomerative)
│
▼
Transformer Summarization
│
▼
Emotion Detection (DistilRoBERTa + VADER)
│
▼
Fuzzy Logic Scoring & Weighting
│
▼
Topic Classification (DeBERTa / Embeddings)
│
▼
Personalized Ranking + Audio Generation
│
▼
Gradio Web Interface

markdown
Copy code

---

## 📊 Dataset & Sources  

- **Dynamic Data**: Live RSS feeds from major news outlets — CNN, BBC, Reuters, Al-Jazeera, The Guardian, and others.  
- **Extraction**: Each entry includes:
  - `title`, `url`, `domain`, `summary`, `published_at`, `raw_text`
- **Processing**:
  - Fetched using `feedparser`
  - Full content parsed with `newspaper3k`
  - Deduplication via `SentenceTransformer` cosine similarity
- **Sample Size**: Typically 300–400 live articles per batch.  

---

## 🧩 Methodology  

| Step | Process | Technique / Model | Library |
|------|----------|-------------------|----------|
| **Data Collection** | RSS + Full-text extraction | Feedparser, Newspaper3k | Python |
| **Preprocessing** | Deduplication, truncation | Embedding similarity | SentenceTransformer |
| **Summarization** | Abstractive summarization | DistilBART (`sshleifer/distilbart-cnn-12-6`) | Transformers |
| **Emotion Detection** | Multi-label classification | DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`) | Hugging Face |
| **Sentiment Scoring** | Polarity detection | VADER | VaderSentiment |
| **Fuzzy Logic Scoring** | Emotion membership evaluation | scikit-fuzzy | Control system |
| **Topic Detection** | Zero-shot + semantic embedding | DeBERTa-v3-large | Transformers |
| **Clustering** | Semantic grouping | Agglomerative Clustering | scikit-learn |
| **Audio Summary** | Text-to-Speech generation | gTTS | gtts |
| **Frontend** | Interactive visualization | Gradio | gradio |

---

## ⚙️ Experimental Setup  

- **Environment**: Python 3.10+, Colab GPU / Local (CUDA optional)  
- **Runtime Dependencies**:  
  - `transformers`, `sentence-transformers`, `scikit-fuzzy`, `newspaper3k`, `feedparser`, `gTTS`, `gradio`
- **Hyperparameter Configuration**:
  - Similarity threshold for clustering → `0.65–0.92`  
  - Fuzzy thresholds for low/medium/high emotion intensity  
  - Summarization length → `min=30`, `max=110` tokens  
  - Topic confidence threshold → `0.45`
- **Execution Time**:
  - ~2 minutes to fetch, summarize, and classify 100+ news stories.

---

## 🧪 Installation  

```bash
# Clone the repository
git clone https://github.com/tanvisingh18/Scroll---Emotion-Weighted-News-Summarizer.git
cd Scroll---Emotion-Weighted-News-Summarizer

# Install dependencies
pip install -r requirements.txt
▶️ Usage
Launch the Jupyter notebook or Gradio app:

bash
Copy code
jupyter notebook finalfinal.ipynb
Select your preferred topics (e.g., Technology, Politics, Health).

The model automatically:

Fetches recent news

Summarizes clusters

Evaluates emotional intensity

Ranks stories using fuzzy-logic-based weighting

View or listen to generated summaries with embedded audio via Gradio UI.

📈 Evaluation Metrics
Metric	Description
Summarization Accuracy	Evaluated through semantic coherence and compression ratio
Emotion Intensity Score	Weighted fuzzy score combining all emotional categories
Topic Classification Precision	Consistency between DeBERTa and semantic embedding classification
User Ranking Relevance	Measures personalized feed effectiveness
Sentiment Balance	Distribution of polarity via VADER scores

🔬 Comparative Baseline
Model	Approach	Observations
Baseline Summarizer	Plain DistilBART summarization	Neutral summaries lacking emotional depth
Proposed Scroll Model	Emotion-weighted fuzzy hybrid	Produces emotionally contextual, more relevant summaries
Topic-Only Variant	DeBERTa classification only	Accurate topic tagging, lacks emotion ranking

🔮 Future Scope
🌐 Multilingual summarization using mbart-large-50 for cross-language support

💬 Adaptive tone synthesis – modulate voice based on emotion intensity

🧭 User interaction learning – refine personalization from click data

📱 Web deployment as a Progressive Web App with offline audio cache

📚 References
Hartmann, J. Emotion English DistilRoBERTa Model, 2023.

Lewis, M. et al. BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, ACL 2020.

He, P. et al. DeBERTa: Decoding-Enhanced BERT with Disentangled Attention, 2021.

Pedregosa, F. et al. Scikit-learn: Machine Learning in Python, JMLR 2011.

Hugging Face Transformers, Model Repository, 2024.

Scikit-Fuzzy Documentation, Fuzzy Control Systems in Python, 2023.

🧑‍💻 Team
Project Title: Scroll — Emotion-Weighted News Summarizer
Course: BCSE306 Artificial Intelligence
Professor: Dr. Ilanthenral Kandasamy
Institution: VIT Vellore

Developer: Tanvi Singh
GitHub: @tanvisingh18

🤖 Acknowledgment of GenAI Use
This project utilized ChatGPT and Hugging Face Transformers for experimentation, documentation, and code generation under academic guidelines.

🧾 License
This project is released under the MIT License — allowing academic and personal use with attribution.

✨ Summary
Scroll represents the convergence of AI-driven NLP, fuzzy logic reasoning, and human-centered personalization.
It demonstrates how emotional context can enhance machine summarization and enable deeper engagement with real-world information streams.
