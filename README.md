# 💸 SpendSense

AI-powered spending monitor that auto-categorizes transactions, forecasts budget overruns before they happen, and sends real-time alerts — built for the Decoding Data Science (DDS) AI Application Building Challenge.

## 🌐 Live App

🔗 [Try SpendSense →](https://your-app-name.streamlit.app) <!-- update once deployed -->

## 🌟 What is SpendSense?

SpendSense watches your spending (via simulated bank SMS alerts) and tells you — before the week is over — whether you're on track to blow a budget. It auto-categorizes every transaction, projects your weekly spend at the current pace, fires a Telegram alert the moment a category looks risky, and writes a short AI-generated read on how your week is going.

Stop finding out you're over budget after the damage is done.

## ✨ Key Features

- 🤖 **AI-powered categorization** — every transaction is classified (food, shopping, travel, daily) using a trained ML model, with a confidence threshold so uncertain transactions are flagged as "uncategorized" instead of silently guessed wrong
- 📈 **Budget breach forecasting** — projects each category's full-week spend from the current daily burn rate, so you see a breach coming before it happens
- 🔔 **Real-time Telegram alerts** — fires the moment a category is projected to go over budget
- 🌙 **Automated evening digest** — a scheduled GitHub Action sends a consolidated daily summary every evening, independent of the dashboard
- 📝 **Structured AI summary** — Gemini Flash-Lite returns a fixed-format read (tone, headline, tip) instead of a free-text blob, so the UI can color-code it and the Telegram digest can format it consistently
- 📜 **Run history** — every check is logged to a `History` sheet tab, so past digests aren't lost — visible in the dashboard sidebar
- 📊 **Live dashboard** — custom-built dark fintech UI with a spend/budget bar chart, category donut chart, and filterable transaction feed
- 🛡️ **Resilient by design** — retries on transient Gemini API errors with exponential backoff, skips malformed transaction rows instead of crashing, and normalizes spreadsheet header whitespace

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, custom CSS |
| Categorization | scikit-learn (TF-IDF + Logistic Regression) |
| AI Summaries | Gemini Flash-Lite (structured JSON output) |
| Data source | Google Sheets (gspread) — simulates live bank SMS feed |
| Alerts | Telegram Bot API |
| Scheduling | GitHub Actions (daily evening digest) |

## 🚀 How to Run

### Option A — Streamlit dashboard (interactive)

1. Clone the repo
   ```
   git clone https://github.com/Ahmad-hub-bot/spendsense-ai.git
   cd spendsense-ai
   ```
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Add your secrets locally in `.streamlit/secrets.toml` (or via the Streamlit Cloud dashboard once deployed):
   ```toml
   GEMINI_API_KEY = "your-gemini-key"
   TELEGRAM_BOT_TOKEN = "your-telegram-bot-token"
   TELEGRAM_CHAT_ID = "your-chat-id"
   GSHEET_NAME = "SpendSense Live Feed"

   [gcp_service_account]
   type = "service_account"
   project_id = "..."
   private_key_id = "..."
   private_key = "..."
   client_email = "..."
   client_id = "..."
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "..."
   ```
4. Run the app
   ```
   streamlit run streamlit.py
   ```

### Option B — Scheduled evening digest (automated)

Runs independently via GitHub Actions (`.github/workflows/`). Set these as **repo secrets** (Settings → Secrets and variables → Actions):

```
GEMINI_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
GSHEET_NAME            (optional — defaults to "SpendSense Live Feed")
GCP_SERVICE_ACCOUNT_JSON   (full service account JSON, as a single-line string)
```

The workflow runs `daily_summary.py` on schedule and sends a Telegram digest regardless of whether any category has breached its budget.

## 📁 Project Structure

```
spendsense-ai/
├── .github/workflows/    # GitHub Actions — scheduled daily digest
├── notebooks/            # Original prototyping notebook
├── streamlit.py          # Main Streamlit dashboard (entry point)
├── daily_summary.py      # Standalone scheduled digest script
├── requirements.txt      # Dependencies
└── README.md
```

## ⚠️ Notes

- Gemini free tier has a daily token/request limit — if you hit it, the app falls back gracefully (empty/placeholder AI summary) rather than crashing.
- The classifier is trained on a small built-in sample set; transactions below a 40% confidence threshold are labeled "uncategorized" rather than guessed.
- `daily_summary.py` and `streamlit.py` share the same training data and budget config — keep them in sync if you change either.
- The Google Sheet acts as a simulated live bank feed; a `History` tab is auto-created on first run to log past digests.

## 🏆 Challenge Info

**Challenge:** Decoding Data Science 8-Day AI Application Building Challenge
**Builder:** Ahmad
