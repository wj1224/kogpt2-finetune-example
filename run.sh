#!/bin/bash

uvicorn run_fastapi:app --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &

streamlit run run_streamlit.py --server.port 8501