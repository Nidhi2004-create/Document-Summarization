# Case Study 13 - Document Summarization (PySpark + NLP + GPT chatbot + STRIDE)

## Run order
    pip install -r requirements.txt        # Java 8+ must be installed for Spark
    python 01_load_data.py                 # builds data/reuters_raw.csv (set S3_BUCKET to also use AWS S3)
    python 02_pyspark_preprocessing.py     # cleaning, tokenising, features -> Parquet
    python 03_eda.py                       # EDA figures -> outputs/
    python 04_summarizer.py                # models M1-M4 + evaluation
    python 06_make_figures.py              # terminal-style screenshots + STRIDE diagram (optional)
    python 05_chatbot.py                   # chatbot (set OPENAI_API_KEY for GPT + multilingual)
    (or:  bash run_all.sh)

## Chatbot commands
/doc <doc_id> | /paste | /summarize | /style paragraph|bullets|tldr|eli5|executive
/lang <language> | /length short|medium|long | /ask <question> | /prompt | /quit

## Before you submit
* Take your own screenshots of steps 02, 04 and 05 (the PNGs in outputs/ are console captures rendered from the real logs).
* Run the chatbot once with OPENAI_API_KEY set and, if you like, paste your real GPT answers into Table 6 of the report.
* Fill in your name / roll number on page 1.
