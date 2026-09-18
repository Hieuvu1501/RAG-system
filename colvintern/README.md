# 🖼 ColVintern-1B-v1 Demo (Visual Document Retrieval)

> Part of the [RAG demos](../README.md) collection — see also [basic-rag](../basic-rag), [hybrid-search-rag](../hybrid-search-rag), and [lightrag](../lightrag).

[ColVintern-1B-v1](https://huggingface.co/5CD-AI/ColVintern-1B-v1) is a ColPali-style visual retrieval model for Vietnamese (and English): it embeds whole **document images** (scans, screenshots, photographed pages, tables, posters) and text queries into the same multi-vector token space, then scores every `(query, image)` pair with **ColBERT-style MaxSim** ("late interaction") — no OCR step needed.

This is a different retrieval problem from the other three demos: instead of chunking and searching *text*, you're searching *images of documents* directly. It's also the base model that [`hybrid-search-rag`](../hybrid-search-rag)'s `vintern` embedding backend builds on — that one (`5CD-AI/Vintern-Embedding-1B`) extends this with text-document embedding on top of the same image retrieval capability.

Retrieval alone only returns *which image* best matches a question, not an answer in words. Optionally, when `GEMINI_API_KEY` is set, `main.py` takes the top-matched image and asks Gemini (multimodal) to read it and answer the question directly — the same "retrieve, then generate" pattern as the other three demos, just with an image as the retrieved context instead of a text chunk.

## 🏛 How it works

```
  Document images (ex1.jpg, ex2.jpg, ...)
              │
              ▼
   [ColVintern processor.process_images]
              │
              ▼
   Per-image multi-vector token embeddings
              │
   Text Query ──► [processor.process_queries] ──► query token embeddings
              │
              ▼
   [processor.score_multi_vector]  (ColBERT-style MaxSim)
              │
              ▼
       Ranked images per query
              │
              ▼ (optional, if GEMINI_API_KEY is set)
   [Gemini reads the top-matched image] ──► Answer in words
```

Retrieval is 100% local — no LLM call, no API key needed for that part. The answer-generation step is optional and only runs when `GEMINI_API_KEY` is configured.

## 📂 Project Structure

```
colvintern/
├── .env.example       # Environment configuration template (GEMINI_API_KEY optional)
├── requirements.txt   # torch, transformers==4.48.2, timm, einops, decord/eva-decord, Pillow, python-dotenv, google-genai (optional)
├── data/
│   ├── ex1.jpg         # Sample document image (from the official model card)
│   └── ex2.jpg         # Sample document image (from the official model card)
├── main.py             # Loads the model, indexes data/*.jpg, and scores queries
└── README.md
```

## ⚡ Quickstart

```bash
cd colvintern
pip install -r requirements.txt
.venv/bin/python main.py --query "Phí giao hàng bao nhiêu ?"
```

First run downloads the ~0.9B-param model from Hugging Face. It auto-picks CUDA → MPS → CPU; CPU works but is slow (falls back to float32 since bf16 isn't well-supported there).

With `GEMINI_API_KEY` set in `.env`, the same command also prints a generated answer, grounded in the matched image:
```
Query > Phí giao hàng bao nhiêu ?
--------------------------------------------------------------------
  1. [score=62.8333] ex2.jpg
--------------------------------------------------------------------

💡 Answer (grounded in ex2.jpg):
--------------------------------------------------------------------
Thông tin về phí giao hàng không được hiển thị cụ thể trên hóa đơn/phiếu
giao hàng này (chỉ có mục "Tiền thu Người nhận" là 182,700 VND).
--------------------------------------------------------------------
```
Without a key, only the retrieval scores are shown — the script never requires `GEMINI_API_KEY` to run.

Run every sample query against every image at once and print the full score table:
```bash
.venv/bin/python main.py --all
```
```
Query                                         |  ex1.jpg |  ex2.jpg
-------------------------------------------------------------------
Chuyện gì xảy ra với quốc lộ 5 TP Hải Phòng ? | 64.5932 * |  61.8635
Phí giao hàng bao nhiêu ?                     |  60.7226 | 62.8208 *
```

Use your own queries with `--all` by repeating `--query`:
```bash
.venv/bin/python main.py --all --query "Câu hỏi 1" --query "Câu hỏi 2"
```

Or run interactively:
```bash
.venv/bin/python main.py
```

Drop your own document images into `data/` (`.jpg`/`.png`) to search over them instead.

## ⚙️ Configuration (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `COLVINTERN_MODEL` | `5CD-AI/ColVintern-1B-v1` | HuggingFace model id |
| `COLVINTERN_DATA_DIR` | `data` | Directory of images to index |
| `COLVINTERN_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `GEMINI_API_KEY` | *(optional)* | Enables the answer-generation step ([get one free](https://aistudio.google.com/)); retrieval works without it |
| `LLM_MODEL` | `gemini-3.6-flash` | Model used to read the matched image and answer |

## Notes

- `attention_mask` is cast to the model's compute dtype (`bfloat16`/`float32`), not left as int — this model consumes it as a multiplicative float weight rather than a boolean mask, per the official quickstart. Getting this wrong silently produces bad scores rather than an error.
- ColVintern-1B-v1 only embeds **images** and **queries** (`process_images` / `process_queries`). It has no `process_docs` for plain text — that capability was added in the newer [`Vintern-Embedding-1B`](https://huggingface.co/5CD-AI/Vintern-Embedding-1B), which this repo already uses for text retrieval in `hybrid-search-rag`.
- `decord`, `timm`, and `einops` are hard dependencies of the model's own remote code (not optional, despite how such packages are often listed for other HF vision models — `transformers` refuses to load the model at all without them). `flash_attn` is the one genuinely optional dependency, and only helps on CUDA; the model falls back to eager attention without it (you'll see a harmless `FlashAttention is not installed.` log line).
- The official `decord` PyPI package has no macOS arm64 (Apple Silicon) wheels — `requirements.txt` installs the `eva-decord` fork instead on that platform, which provides the same `decord` import.
- The first run downloads the model from Hugging Face's public CDN, which rate-limits unauthenticated requests under load (`429 Too Many Requests`, "maximum queue size reached"). If a download gets interrupted mid-way, `transformers` can be left with a **corrupted local module cache**: the file exists fully in `~/.cache/huggingface/hub/models--5CD-AI--ColVintern-1B-v1/snapshots/<revision>/`, but is missing from `~/.cache/huggingface/modules/transformers_modules/5CD-AI/ColVintern-1B-v1/<revision>/`, so retries keep failing with a `FileNotFoundError` for some `.py` file even though your connection is fine. Fix: delete both cache directories for this model and let it re-download from scratch:
  ```bash
  rm -rf ~/.cache/huggingface/modules/transformers_modules/5CD-AI/ColVintern-1B-v1
  rm -rf ~/.cache/huggingface/hub/models--5CD-AI--ColVintern-1B-v1
  ```
  Setting `HF_TOKEN` in your environment (a free HF account) raises the rate limit and makes this much less likely to happen in the first place.
