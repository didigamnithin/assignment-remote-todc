## DoorDash Performance Dashboard

Streamlit dashboard for Marketing, Operations, Sales, and Payouts built from the provided CSVs and notebooks.

### Quickstart
1) Clone the repo
2) From project root, run:
   ./run.sh
   # or manual:
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   streamlit run script.py

App runs at http://localhost:8501

### Notes
- `.env` and local virtual envs are ignored via `.gitignore`.
- Optional heavy libs (Prophet, statsmodels, sklearn) are pinned in `requirements.txt`.
- Data files expected at project root:
  - doordash powers marketing.csv
  - doordash powers operations.csv
  - doordash powers payouts.csv
  - doordash powers sales.csv
