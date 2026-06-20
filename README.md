# SpendSense

AI-powered spending monitor that auto-categorizes transactions and predicts budget overruns *before* they happen.

Built for the Decoding Data Science (DDS) AI Application Building Challenge — 8 days, building in public.

## What it does
- Captures bank SMS alerts (simulated via Google Sheets for this prototype)
- Auto-categorizes spending (food, shopping, travel, daily) using ML
- Forecasts budget threshold breaches using trend analysis
- Sends real-time alerts via Telegram
- Generates weekly AI-written spending summaries (Gemini Flash-Lite)

## Tech stack
- Google Colab (Python)
- gspread (Google Sheets integration)
- scikit-learn (classification)
- pandas (forecasting)
- Gemini Flash-Lite API (summarization)
- Telegram Bot API (alerts)

## Status
🚧 Day 2 — Environment setup complete, core logic in progress

## How to run
1. Open `notebooks/spendsense.ipynb` in Google Colab
2. Run the setup cells (installs dependencies)
3. Paste your Gemini API key when prompted
4. Run all cells
