# 🎬 AuraFlow
 
**An end-to-end multimodal intelligence pipeline for video — built on Whisper, pyannote, CLIP, and LLaMA.**
 
AuraFlow takes a single video file and turns it into a full analytical package: who said what, what the camera saw, how people felt, and what it all means — surfaced through one Gradio app.
 
---
 
## Why AuraFlow
 
Most video tools handle one modality at a time — either the audio gets transcribed, or the frames get tagged. AuraFlow fuses both tracks into a single timeline, so the emotional tone of a scene is read from *what was said* and *what was shown*, not just one or the other.
 
---
 
## Core Capabilities
 
- **Speech-to-text** with word-level timing (Whisper)
- **Speaker separation** across the conversation (pyannote diarization)
- **Scene narration** for each keyframe (LLaMA 3.2 Vision via Groq)
- **Facial/emotional cues** pulled from frames (CLIP zero-shot, 27 emotion classes)
- **Dual-channel sentiment timeline** — voice tone vs. visual cues, plotted together
- **Audio + visual fusion summary** of the entire video
- **Auto-generated chapters** and key-moment timestamps
- **Conversational Q&A** grounded in the transcript (RAG with FAISS)
- **14-language translation** of transcript and summary
- **Auto-edited highlight reel** built from motion-detected keyframes
- **Summary-quality scoring** — both against an AI-generated reference and against your own reference text
- **Location extraction** — places mentioned in speech, mapped automatically
- **Topic relationship graph** connecting transcript segments by meaning
---
 
## Model Stack
 
| Stage | Model / Library |
|---|---|
| Transcription | Whisper (base) |
| Diarization | pyannote `speaker-diarization-3.1` |
| Frame captioning | LLaMA 3.2 Vision — `llama-4-scout-17b-16e-instruct` (Groq) |
| Visual emotion | OpenCLIP ViT-L/14, zero-shot |
| Text sentiment | RoBERTa (`twitter-roberta-base-sentiment-latest`), fine-tuned |
| Summaries / chapters / Q&A reasoning | LLaMA 3.3 70B — `llama-3.3-70b-versatile` (Groq) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| Translation | Google Translate via `deep-translator` |
| Vector search | FAISS (flat IP index) |
 
---
 
## How a Video Flows Through the System
 
```
                         ┌────────────────────────┐
                         │      Video Upload       │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                                 │
        AUDIO TRACK                                       VIDEO TRACK
              │                                                 │
        Whisper ASR                                  Keyframe extraction
              │                                       (motion-diff / uniform)
        pyannote diarization                                    │
              │                                ┌─────────────────┴────────────────┐
   Transcript (speaker-tagged segments)    LLaMA 3.2 Vision               CLIP ViT-L/14
              │                          (scene descriptions)         (27-way emotion scoring)
   ┌─────┬────┴────┬─────────┐                  │                              │
LLaMA 3.3  FAISS+MiniLM   RoBERTa          FFmpeg stitching                    │
(summary,  (RAG index)   (per-segment             │                            │
chapters)                 sentiment)        Highlight reel                    │
              │                │                                              │
        deep-translator        └──────────────────┬───────────────────────────┘
        (14 languages)                             │
              │                              FUSION LAYER
              └─────────────────────────────────────┤
                                                     │
                          LLaMA 3.3 → combined narrative summary
                          RoBERTa + CLIP → fused sentiment curve
                          MiniLM cosine sim → quality/coverage scoring
                                                     │
                                          Gradio multi-tab UI
```
 
---
 
## Sentiment Engine
 
Sentiment is computed every 2 seconds across the video's full duration, from two independent sources that are then blended:
 
- **Voice channel** — every transcript segment is scored by a fine-tuned RoBERTa classifier (Negative / Neutral / Positive on a continuous 0–1 scale). It was trained for 5 epochs on ~24K balanced samples from the Kaggle Twitter Entity Sentiment dataset, landing around 85%+ per-class accuracy.
- **Visual channel** — CLIP compares each keyframe against 27 fine-grained emotional descriptors ("genuine smile," "anxiety," "pride," etc.) and rolls those into a Positive / Negative / Neutral score.
- **Fusion** — voice and visual contribute equally; the combined line is a straight average of both.
The resulting chart plots three series: voice (dashed), visual (dotted), and the fused result (solid).
 
---
 
## Measuring Summary Quality
 
AuraFlow doesn't just generate a summary — it checks itself:
 
**Automatic check** — LLaMA 3.3 drafts independent "ground truth" paragraphs from the raw transcript and from the visual narrative. MiniLM embeddings then measure cosine similarity between each generated summary and its reference (scores capped at 95% to avoid inflated numbers), and LLaMA writes a short note on what was captured well and what got missed.
 
**Manual check** — drop in your own reference paragraph, and AuraFlow scores the combined summary against it directly, surfacing the three transcript segments most relevant to your reference text.
 
For the auto-edited highlight reel specifically, ROUGE-1/2/L scoring compares the transcript text covered by the selected clips against the full transcript, producing a coverage percentage and a letter-style grade (Excellent → Poor).
 
---
 
## Semantic Network View
 
Transcript segments are embedded and connected whenever their similarity clears a configurable threshold, producing a navigable graph:
 
- 🟣 **Purple** — high-degree hub segments (the video's core topics)
- 🟠 **Amber** — low-degree but informative segments
- ❌ **Red** — filler content, flagged by LLaMA as low-value
- Peripheral segments link by dotted edge to their closest hub
A dedicated action feeds only the hub + informative segments back into LLaMA for a summary that ignores filler entirely.
 
---
 
## Getting Started
 
### 1. Install dependencies
 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper
pip install "transformers==4.47.0" "tokenizers==0.21.0" "accelerate>=0.26" "huggingface_hub==0.33.5"
pip install "sentence-transformers>=3.0"
pip install groq faiss-cpu gradio "pandas>=2.0,<3.0"
pip install opencv-python-headless Pillow deep-translator
pip install pyannote.audio
pip install "scikit-learn>=1.6.1" "hdbscan>=0.8.40" "umap-learn>=0.5.6"
pip install open-clip-torch rouge-score
apt-get install ffmpeg
```
 
A version pin matters for stability — apply it after the rest installs:
 
```bash
pip install "numpy==2.2.2" "scipy==1.15.2" --force-reinstall --no-deps
```
 
### 2. Add your API keys
 
```python
GROQ_API_KEY = ""   # console.groq.com
HF_TOKEN     = ""   # huggingface.co/settings/tokens
```
 
> Your Hugging Face token needs access to `pyannote/speaker-diarization-3.1` — accept that model's terms on the Hub before running diarization.
 
### 3. Run the notebook top to bottom
 
The cells have to run in order because of a kernel restart partway through:
 
1. Bootstrap / dependency pinning *(restarts the kernel — this is expected)*
2. Full package install *(restart the runtime afterward)*
3. torchvision upgrade
4. transformers / huggingface_hub pin
5. NumPy patch *(only needed if you hit a `_blas_supports_fpe` error)*
6. Sanity-check all imports
7. Set API keys
8. Load every model into memory
9. Audio-pipeline functions
10. Visual-pipeline functions
11. RAG, translation, and summarization functions
12. Similarity-scoring functions
13. Fine-tune the RoBERTa sentiment head (~8 min; checkpoint saved to Drive — reused on later runs)
14. Launch the Gradio app
---
 
## What You Get in the UI
 
| Tab | Contents |
|---|---|
| 📝 Transcript | Segment-by-segment viewer with ✓ / ✗ / ? accuracy tagging |
| 📊 Analysis | Summary, chapters, key moments |
| 🎨 Visual | Scene narrative, audio-visual sync view, keyframe gallery |
| 📋 Combined Summary | Fused audio + visual overview |
| 🎭 Multimodal Sentiment | Dual-channel timeline + emotion breakdown |
| 🌍 Translation | Transcript and summary across 14 languages |
| 💬 Q&A Chat | RAG chatbot grounded strictly in the transcript |
| 🎯 Similarity Evaluation | Auto and manual summary-quality scoring |
| 🎬 Summary Video | Auto-edited highlight reel + ROUGE accuracy report |
| 🕸️ Semantic Network | Segment-similarity graph + graph-driven summary |
 
---
 
## Languages Supported
 
Hindi · Spanish · French · German · Portuguese · Japanese · Korean · Chinese (Simplified) · Arabic · Italian · Tamil · Telugu · Malayalam · Kannada
 
---
 
## Input Types
 
| Input | What still works |
|---|---|
| Audio + Video | Everything |
| Audio only | Transcription and all text/NLP tabs; visual tabs report "no video" |
| Video only | Visual tabs and keyframes; transcript-dependent tabs report "no audio" |
 
---
 
## Known Limitations
 
- Built for Google Colab; a GPU is strongly recommended, CPU will be noticeably slower
- Whisper-base trades accuracy for speed — swap in `large` if precision matters more than runtime
- CLIP-based emotion detection leans on visible facial expression and can struggle without it
- Diarization quality drops with heavy speaker overlap or more than ~5 simultaneous voices
- Free-tier Google Translate may rate-limit very long transcripts
- The highlight-reel feature needs FFmpeg installed and enough free space under `/content/`
---
 
## Project Layout
 
```
auraflow_git.ipynb              # the full pipeline, meant to run in Colab
checkpoints/                    # Gradio checkpoints (created automatically)
results/                        # per-epoch sample outputs (created automatically)
/content/drive/MyDrive/
  ├── auraflow_sentiment_ckpt/  # fine-tuned RoBERTa weights
  └── summary_video.mp4         # generated highlight reel
```
 
---
 
## Built On
 
[OpenAI Whisper](https://github.com/openai/whisper) · [pyannote.audio](https://github.com/pyannote/pyannote-audio) · [OpenCLIP](https://github.com/mlfoundations/open_clip) · [Cardiff NLP RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) · [Groq](https://groq.com/) (LLaMA 3.2 Vision + LLaMA 3.3 70B) · [FAISS](https://github.com/facebookresearch/faiss) · [Gradio](https://www.gradio.app/)
