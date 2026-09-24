#!/usr/bin/env bash
# Runs the complete pipeline. Usage:  bash run_all.sh
set -e
python 01_load_data.py
python 02_pyspark_preprocessing.py
python 03_eda.py
python 04_summarizer.py
echo "Now run the chatbot:  python 05_chatbot.py"
