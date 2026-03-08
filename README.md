# 🌊 AuraFlow

> **Multimodal Video Intelligence** — Transcript · Speakers · Visual Analysis · Summary · Chapters · Translation · Q&A · Similarity Evaluation · Multimodal Sentiment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-orange)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 📖 Overview

AuraFlow is an end-to-end video understanding pipeline that combines **audio transcription**, **speaker diarization**, **visual frame analysis**, and **fine-tuned sentiment modeling** into a single Gradio-powered interface. Upload any video and get a rich multimodal analysis in minutes.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ **Transcription** | Whisper-powered speech-to-text with word-level timestamps |
| 👥 **Speaker Diarization** | pyannote.audio 3.1 — identifies and labels each speaker |
| 🎨 **Visual Analysis** | BLIP keyframe captioning — scene, action, and setting descriptions |
| 📋 **Summarization** | LLaMA 3.3 70B (via Groq) generates concise audio + visual summaries |
| 📑 **Chapters** | Auto-generated chapter markers with real timestamps |
| ⭐ **Key Moments** | Most important or surprising moments extracted from transcript |
| 🎭 **Multimodal Sentiment** | Fine-tuned RoBERTa (85%+ accuracy) fused with BLIP visual emotion scoring |
| 🌍 **Translation** | 14 languages via Google Translate (Hindi, Spanish, French, Arabic, Tamil, and more) |
| 💬 **Q&A Chat** | RAG pipeline — FAISS + sentence-transformers + LLaMA 3.3 |
| 🎯 **Similarity Evaluation** | Auto (LLaMA-generated reference) and Manual modes with cosine scoring |

---

## 🏗️ Architecture

```
Video Input
    │
    ├──► FFmpeg Audio Extraction
    │         │
    │         ├──► Whisper (transcription)
    │         └──► pyannote.audio (diarization)
    │                   │
    │                   └──► Segments [{text, start, end, speaker}]
    │
    ├──► OpenCV Keyframe Extraction
    │         │
    │         └──► BLIP (scene / action / setting captions)
    │                   │
    │                   └──► Visual Emotion Scoring
    │
    └──► Groq LLaMA 3.3 70B
              ├── Summary · Chapters · Key Moments
              ├── Combined Audio+Visual Summary
              └── Similarity Evaluation

Sentiment Pipeline:
  RoBERTa (fine-tuned on Twitter) ──┐
                                    ├──► Fused Score (70% text + 30% visual)
  BLIP Visual Emotion Score ────────┘
```

---

## 🚀 Quick Start

### 1. Open in Google Colab
Upload `AuraFlow.ipynb` to [Google Colab](https://colab.research.google.com) and run cells in order.

### 2. Set API Keys
In the keys cell, fill in:
```python
GROQ_API_KEY = "your_groq_key"   # https://console.groq.com
HF_TOKEN     = "your_hf_token"   # https://huggingface.co/settings/tokens
```
> ⚠️ Accept the pyannote speaker diarization terms at [hf.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

### 3. Upload the Twitter Dataset (Cell 13 only)
Cell 13 fine-tunes a sentiment model and requires `twitter_training.csv` from the [Kaggle Twitter Entity Sentiment dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

**Upload options:**

```python
# Option A — file picker dialog
from google.colab import files
uploaded = files.upload()

import zipfile
with zipfile.ZipFile("twitter_training.zip", "r") as z:
    z.extractall("/content/")
```

```python
# Option B — Kaggle API (auto-download)
!pip install kaggle -q
# upload kaggle.json first, then:
!kaggle datasets download -d jp797498e/twitter-entity-sentiment-analysis
!unzip twitter-entity-sentiment-analysis.zip -d /content/
```

### 4. Run All Cells → Launch Gradio
The final cell launches a public Gradio link (`share=True`). Open it and upload any video.

---

## 📦 Dependencies

All installed automatically in Cell 1 (Bootstrap):

```
torch >= 2.0          openai-whisper        transformers == 4.47.0
tokenizers == 0.21.0  accelerate >= 0.26    sentence-transformers >= 3.0
groq                  faiss-cpu             gradio
opencv-python         Pillow                deep-translator
pyannote.audio        scikit-learn          hdbscan
umap-learn            datasets              plotly
numpy == 2.2.2        scipy == 1.15.2       ffmpeg (apt)
```

> **Note:** Cell 1 force-pins `numpy==2.2.2` and `scipy==1.15.2` then kills the kernel. This is expected — restart and continue from Cell 2.

---

## 🧠 Models Used

| Model | Purpose | Source |
|---|---|---|
| `openai/whisper-base` | Speech-to-text | OpenAI |
| `pyannote/speaker-diarization-3.1` | Speaker identification | HuggingFace |
| `Salesforce/blip-image-captioning-base` | Visual frame description | HuggingFace |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment (fine-tuned) | Cardiff NLP |
| `all-MiniLM-L6-v2` | Sentence embeddings for RAG | SBERT |
| `llama-3.3-70b-versatile` | Summarization, Q&A, evaluation | Groq |

---

## 🎭 Sentiment Model Details

- **Base model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Fine-tuned on:** Kaggle Twitter Entity Sentiment dataset (~24,000 samples)
- **Classes:** Negative · Neutral · Positive
- **Target accuracy:** 85%+ per class
- **Fusion:** 70% text sentiment + 30% BLIP visual emotion score
- **Output:** Per-segment timeline with fused score, confidence, and dominant mood

---

## 🌍 Supported Translation Languages

Hindi · Spanish · French · German · Portuguese · Japanese · Korean · Chinese (Simplified) · Arabic · Italian · Tamil · Telugu · Malayalam · Kannada

---

## 📁 Project Structure

```
AuraFlow/
├── AuraFlow.ipynb          # Main notebook (all cells)
├── README.md               # This file
└── twitter_training.csv    # Required for Cell 13 (not included — download from Kaggle)
```

---

## ⚠️ Common Issues

| Issue | Fix |
|---|---|
| `FileNotFoundError: twitter_training.csv` | Upload the CSV/zip to Colab before running Cell 13 |
| Kernel crash after Cell 1 | Expected — numpy is being pinned. Restart and skip Cell 1 |
| `AttributeError: _blas_supports_fpe` | Run the numpy fix cell: install `numpy==1.26.4` and restart |
| GitHub save fails from Colab | Use `git clone` + manual push with a personal access token |
| Gradio link not opening | Ensure `share=True` in `demo.launch()` |
| pyannote 401 error | Accept model terms on HuggingFace and verify `HF_TOKEN` |

---

## 🔑 Getting API Keys

- **Groq API Key:** [console.groq.com](https://console.groq.com) → free tier available
- **HuggingFace Token:** [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → read access is sufficient
- **Kaggle API Key:** [kaggle.com/settings](https://www.kaggle.com/settings) → Account → API → Create New Token

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Salesforce BLIP](https://github.com/salesforce/BLIP)
- [Cardiff NLP](https://github.com/cardiffnlp/tweeteval)
- [Groq](https://groq.com) for fast LLaMA 3.3 inference
- [Gradio](https://gradio.app) for the UI
