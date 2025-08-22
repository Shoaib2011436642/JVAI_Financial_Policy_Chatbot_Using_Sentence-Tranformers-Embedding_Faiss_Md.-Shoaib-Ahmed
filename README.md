# JVAI Financial Policy Chatbot

This project began as a **Colab notebook**, then I built a **Streamlit app** that reproduces the exact pipeline and answers so recruiters can test it quickly.

---

## 📓 Notebook (created first)

The notebook implements the complete retrieval pipeline step‑by‑step and was my primary development environment. It includes:
- **PDF extraction** with `pdfplumber`
- **Chunking** into ~600‑character passages (long paragraphs are sentence‑batched)
- **Embeddings** using `sentence-transformers/all-MiniLM-L6-v2`
- **FAISS** vector index (`IndexFlatIP` with L2‑normalized vectors → cosine similarity)
- **Hybrid search**: semantic similarity + light keyword bonus for domain terms
- **Strict post‑processing** tuned for policy Q&A (e.g., taxation vs GSP, strategic priorities, superannuation)
- **Minimal memory** for follow‑ups (last‑topic heuristic)

👉 **Open in Colab:** https://colab.research.google.com/drive/1oRzmf_fhhIQgWMAXfgFcrjV6iKzLA3kE?usp=sharing

**Notebook path (in this repo):**
```
notebook/JVAI_Financial_Policy_CHATBOT.ipynb
```

---

## 💻 Streamlit App (same pipeline, interactive)

To make evaluation easier, I ported the exact notebook logic into a **Streamlit app**. It uses the same:
- extraction → chunking → embeddings → FAISS → hybrid search → strict post‑processing
- minimal conversational memory
- **static PDF** located at `data/For Task - Policy file.pdf` (no upload step needed)

When you run the app, it indexes the bundled PDF and lets you ask questions in a chat UI, returning answers **with page citations** like `Sources: p.12, p.13`.

---

## 📦 Project Structure

```
.
├── app.py                       # Streamlit app (notebook-faithful pipeline)
├── requirements.txt             # Pinned deps (CPU-friendly)
├── README.md                    # This file
├── data/
│   └── For Task - Policy file.pdf   # Bundled policy PDF used by the app
└── notebook/
    └── JVAI_Financial_Policy_CHATBOT.ipynb  # Original notebook
```

> The app uses the **static PDF** in `/data/` by default, so no upload step is required.

---

## ✅ Requirements

This repo targets CPU setups and typical Windows/macOS/Linux dev machines.

```
streamlit==1.37.0
pdfplumber==0.11.4
faiss-cpu==1.8.0.post1
sentence-transformers==2.7.0
numpy>=1.23
torch>=2.1.0
nltk>=3.8
rich>=13.7
```

**Windows note:** If `faiss-cpu` is problematic via `pip`, consider Conda:

```bash
conda install -c conda-forge faiss-cpu=1.8.0
```

If PyTorch doesn’t install automatically:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## ▶️ Run the Notebook

```bash
pip install -r requirements.txt
jupyter notebook notebook/JVAI_Financial_Policy_CHATBOT.ipynb
```
*(Or open the notebook directly in Colab using the badge/link above.)*

---

## ▶️ Run the Streamlit App

```bash
# Optional: create & activate a virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# Install deps
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# Start the app
streamlit run app.py
```

Then open the browser URL (typically `http://localhost:8501`).  
You should see a green “Indexed and ready” status; type questions into the chat box.

**Ensure the PDF exists at:**
```
data/For Task - Policy file.pdf
```

---

## 🧪 Sample Questions (copy/paste to test)

Start with the four core prompts:

1) What are the strategic priorities?  
2) What about debt?  
3) What does the Budget say about taxation vs GSP?  
4) What’s the superannuation funding target?

More prompts to exercise retrieval + strict filtering + memory:

5) Define “net interest” and state the target trend mentioned.  
6) What does the document say about maintaining a balanced budget?  
7) Is the Territory’s credit rating discussed? Summarize it in one line.  
8) What are the key points on infrastructure or capital works?  
9) What does “net assets” refer to in this document?  
10) List any mentions of “operating result” or “surplus.”  
11) Quote the line that includes “Taxation as a % of GSP.”  
12) Give me all percentage figures you can find related to taxation vs GSP.  
13) Extract any dates/years tied to the superannuation funding goal.  
14) Are there targets that mention remaining negative/less than zero? What are they about?  
15) Does the policy mention “property, plant and equipment”? Summarize that part.  
16) What’s said about own-source revenue (if mentioned)?  
17) Is there anything about cash reserves or liquidity?  
18) Summarize any risks or constraints mentioned around debt or interest.  
19) Provide the six bullets under “Strategic priorities” as a single line (keep order).  
20) Compare the discussion of taxation with the discussion of GSP in two sentences.  
21) What performance measures are used around superannuation funding?  
22) Are there tables referenced for these topics? Mention them with page numbers.  
23) Paraphrase the section that discusses credit rating requirements.  
24) What is said about funding percentages and timelines (years) for superannuation?  
25) Summarize infrastructure in exactly 20 words.  
26) Give me two sentences that jointly cover debt AND operating result.  
27) What about it?  ← (ask after #26 to test memory/anaphora)  
28) And what about that?  ← (follow-up to test memory carryover)

---

## 🧠 Design Notes (why answers look clean)

- **Chunking:** ~600‑char chunks; long paragraphs are sentence‑batched to keep meaning intact.  
- **Embeddings:** `all-MiniLM-L6-v2` from `sentence-transformers`.  
- **Vector index:** FAISS `IndexFlatIP` with L2 normalization → cosine similarity.  
- **Hybrid search:** semantic similarity with a small **keyword bonus** for domain terms (debt, taxation, GSP, etc.).  
- **Strict post‑processing:** extracts sentences/lines that include query tokens or **canonical aliases**; special polish for:
  - **Strategic priorities:** stitches 6 bullets even if line‑wrapped.  
  - **Taxation vs GSP:** prefers lines that mention both terms or show the “% of GSP” series.  
  - **Superannuation:** surfaces the funding % and timeline; filters out assets/liabilities noise.  
- **Memory:** minimal, last‑topic heuristic for follow‑ups like “What about it?”

---

## 🛠️ Troubleshooting

- **`ModuleNotFoundError: torch`** → install CPU wheel:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- **FAISS install error on Windows** → prefer conda (see above).  
- **File not found** → ensure `PDF_PATH` in `app.py` points to `data/For Task - Policy file.pdf`.  
- **Port in use** → `streamlit run app.py --server.port 8502`.  
- **Slow first run** → models download on first launch; subsequent runs are faster.

---

## 📄 License / Use

For assessment purposes only. Please contact me before reusing code or data.
