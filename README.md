# Case Study 13 — Document Summarization
### PySpark + NLP + GPT Chatbot + STRIDE Security Analysis

---

## 📋 Prerequisites

Before you begin, make sure you have the following installed:

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.9 – 3.12 | https://www.python.org/downloads/ |
| Java (JDK) | 8 or 11 | https://adoptium.net/ |
| Git | Latest | https://git-scm.com/ |

> ⚠️ **Java is required for PySpark to work.** After installing, set the `JAVA_HOME` environment variable.

---

## 🚀 Getting Started (after downloading from GitHub)

### Step 1 — Clone the repository
```bash
git clone https://github.com/Nidhi2004-create/Document-Summarization.git
cd Document-Summarization
```

### Step 2 — Create a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Enables GPT-based chatbot responses |
| `S3_BUCKET` | Use AWS S3 to load/save data |

**Windows:**
```powershell
$env:OPENAI_API_KEY = "your-key-here"
```
**Mac/Linux:**
```bash
export OPENAI_API_KEY="your-key-here"
```

---

## ▶️ Run Order

Run the scripts in this order:

```bash
python 01_load_data.py              # Load & build data/reuters_raw.csv
python 02_pyspark_preprocessing.py  # Clean, tokenize, extract features → Parquet
python 03_eda.py                    # EDA figures → outputs/
python 04_summarizer.py             # Run models M1–M4 + evaluation
python 06_make_figures.py           # Generate STRIDE diagram & screenshots (optional)
python 05_chatbot.py                # Launch the interactive chatbot
```

Or run everything at once (Mac/Linux):
```bash
bash run_all.sh
```

---

## 💬 Chatbot Commands

Once `05_chatbot.py` is running, use these commands:

| Command | Description |
|---------|-------------|
| `/doc <id>` | Load a document by ID |
| `/paste` | Paste your own text |
| `/summarize` | Summarize the loaded document |
| `/style paragraph\|bullets\|tldr\|eli5\|executive` | Change summary style |
| `/lang <language>` | Translate summary to another language |
| `/length short\|medium\|long` | Set summary length |
| `/ask <question>` | Ask a question about the document |
| `/quit` | Exit the chatbot |

---

## 📁 Project Structure

```
Document-Summarization/
├── 01_load_data.py               # Data loading
├── 02_pyspark_preprocessing.py   # PySpark preprocessing
├── 03_eda.py                     # Exploratory Data Analysis
├── 04_summarizer.py              # Summarization models
├── 04b_abstractive_bart_optional.py  # BART model (optional)
├── 05_chatbot.py                 # Interactive chatbot
├── 06_make_figures.py            # Figure generation
├── common.py                     # Shared utilities
├── summarizer_lib.py             # Summarizer library
├── requirements.txt              # Python dependencies
├── run_all.sh                    # Run all scripts (Linux/Mac)
├── data/                         # Dataset files
└── outputs/                      # Generated figures & logs
```

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `JAVA_HOME not set` | Install Java 8/11 and set JAVA_HOME env variable |
| `PySpark not found` | Run `pip install pyspark` |
| `ModuleNotFoundError` | Make sure venv is activated and `pip install -r requirements.txt` was run |
| GPT responses not working | Set `OPENAI_API_KEY` environment variable |
| Windows Spark error | The `hadoop_home/bin/winutils.exe` is included — set `HADOOP_HOME` to the `hadoop_home` folder |

