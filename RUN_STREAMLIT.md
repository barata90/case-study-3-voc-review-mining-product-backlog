# Run the Streamlit Mock Dashboard

## 1) Install dependencies
```bash
pip install -r requirements.txt
pip install streamlit
```

## 2) Run
From the project directory containing the files:

- `streamlit_app.py`
- `airline_reviews_skytrax.csv`

Run:
```bash
streamlit run streamlit_app.py
```

## 3) Notes
- The first run trains a lightweight TF‑IDF + Logistic Regression sentiment model (cached by Streamlit).
- Sidebar controls adjust the recent window, rolling averages, and alert thresholds.
- Hotspots are computed from available dataset fields (airline, route, cabin, aircraft, author_country).
