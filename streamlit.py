"""
SpendSense — AI-powered spending monitor with predictive budget alerts.
Built for the Decoding Data Science (DDS) AI Application Building Challenge.

Deploy on Streamlit Community Cloud:
1. Push this file to your GitHub repo as app.py
2. Go to share.streamlit.io -> New app -> select this repo/branch/app.py
3. Add secrets in the Streamlit dashboard (Settings -> Secrets) using the format below:

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

   (Copy these values directly from your service account JSON file —
    paste the private_key exactly as-is, including the \\n line breaks.)
"""

import re
import json
import time
import datetime
import requests
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ──────────────────────────────────────────────────────────────
# 1. Training data for the classifier
# ──────────────────────────────────────────────────────────────
SAMPLE_TRANSACTIONS = [
    # FOOD (10)
    "INR 450.00 debited from your account for SWIGGY on 12-06-2026",
    "Rs 220 paid to ZOMATO for food order",
    "You spent Rs 320 at MCDONALDS via UPI",
    "Rs 150 paid to STARBUCKS via UPI",
    "INR 180 paid to DOMINOS PIZZA via UPI",
    "You spent Rs 75 at local TEA STALL via UPI",
    "Rs 500 spent at PIZZA HUT via UPI",
    "INR 90 debited at CAFE COFFEE DAY",
    "Rs 280 paid to KFC via UPI",
    "INR 60 debited at BAKERY for snacks",
    # SHOPPING (10)
    "Rs 1200 spent at ZARA using your card ending 4521",
    "INR 2500.00 debited at AMAZON for online purchase",
    "INR 3200.00 debited at H&M for shopping",
    "You spent Rs 1800 at DECATHLON",
    "Rs 950 spent at LIFESTYLE for clothing",
    "INR 1500 debited at MYNTRA for online order",
    "Rs 4200 spent at RELIANCE TRENDS",
    "INR 700 debited at NIKE STORE",
    "Rs 1100 paid to FLIPKART for purchase",
    "INR 2000 debited at PANTALOONS",
    # TRAVEL (10)
    "INR 89.50 debited for UBER ride on 13-06-2026",
    "INR 600 debited for OLA cab booking",
    "INR 99 debited for METRO CARD recharge",
    "INR 50 debited for PARKING fee",
    "Rs 4500 spent at FLIGHT BOOKING - INDIGO",
    "INR 6000.00 debited at MAKEMYTRIP for hotel booking",
    "Rs 60 paid for BUS TICKET via UPI",
    "INR 3000 debited for SPICEJET flight booking",
    "Rs 80 paid for AUTO RICKSHAW",
    "INR 250 debited for RAPIDO bike ride",
    # DAILY (10)
    "You spent Rs 95 at LOCAL GROCERY STORE",
    "Rs 40 paid for MILK delivery via UPI",
    "INR 120 debited at PHARMACY for medicines",
    "Rs 65 paid to VEGETABLE VENDOR via UPI",
    "INR 200 debited at BIG BAZAAR for groceries",
    "Rs 30 paid for NEWSPAPER subscription",
    "INR 150 debited at DMART for household items",
    "Rs 55 paid to LAUNDRY SERVICE",
    "INR 80 debited for WATER CAN delivery",
    "Rs 100 paid at GENERAL STORE",
]

LABELS = (
    ["food"] * 10 + ["shopping"] * 10 + ["travel"] * 10 + ["daily"] * 10
)

BUDGETS = {
    "food": 15000,
    "shopping": 3000,
    "travel": 1000,
    "daily": 800,
    "miscellaneous": 500,  # catch-all for low-confidence classifications
}


# ──────────────────────────────────────────────────────────────
# 2. Cached resources — classifier, sheet client, gemini client
#    (cached so they aren't rebuilt on every button click)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    vectorizer = TfidfVectorizer(
        stop_words=[
            "inr", "rs", "debited", "spent", "paid", "for", "at", "via",
            "upi", "your", "account", "from", "on", "you", "using",
            "card", "ending",
        ]
    )
    X = vectorizer.fit_transform(SAMPLE_TRANSACTIONS)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, LABELS)
    return vectorizer, clf


@st.cache_resource
def load_spreadsheet():
    """Opens the spreadsheet itself (not just sheet1) so both the live
    feed tab and the History tab can be reached from one client."""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    sheet_name = st.secrets.get("GSHEET_NAME", "SpendSense Live Feed")
    return client.open(sheet_name)


def load_sheet_client():
    """Live transaction feed tab (sheet1) — same role as before."""
    return load_spreadsheet().sheet1


def load_history_sheet():
    """Opens (or creates) a 'History' worksheet tab, to store one row
    per check run so past digests aren't lost between sessions."""
    spreadsheet = load_spreadsheet()
    try:
        return spreadsheet.worksheet("History")
    except gspread.WorksheetNotFound:
        history_ws = spreadsheet.add_worksheet(title="History", rows=1000, cols=8)
        history_ws.append_row(
            ["timestamp", "total_spent", "total_budget", "categories_over",
             "tone", "headline", "tip"]
        )
        return history_ws


@st.cache_resource
def load_genai_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ──────────────────────────────────────────────────────────────
# 3. Core pipeline functions
# ──────────────────────────────────────────────────────────────
def parse_transaction(sms_text: str) -> dict:
    """Extracts amount and a rough merchant guess from raw SMS-style text."""
    amount_match = re.search(
        r"(?:INR|Rs\.?)\s?([\d,]+\.?\d*)", sms_text, re.IGNORECASE
    )
    amount = float(amount_match.group(1).replace(",", "")) if amount_match else None

    # Try capitalized merchant names first (SWIGGY, Subway, Coffee Day, H&M)
    merchant_match = re.findall(r"\b[A-Z][a-zA-Z&]{2,}(?:\s[A-Z][a-zA-Z&]{2,})*\b", sms_text)
    ignore_words = {
        "INR", "RS", "UPI", "CARD", "YOUR", "ACCOUNT", "FROM", "FOR",
        "YOU", "USING", "SPENT", "DEBITED", "PAID", "ENDING", "TO",
    }
    merchants = [m for m in merchant_match if m.upper() not in ignore_words]

    if merchants:
        merchant = merchants[0]
    else:
        # Fallback for lowercase generic phrases (e.g. "general store", "laundry service")
        fallback_match = re.search(
            r"(?:for|at|to)\s+([a-zA-Z\s]+?)(?:\s+via|\s+delivery|$)", sms_text, re.IGNORECASE
        )
        merchant = fallback_match.group(1).strip().title() if fallback_match else "UNKNOWN"

    return {"amount": amount, "merchant": merchant, "raw_text": sms_text}


def classify_transaction(sms_text: str, vectorizer, clf, threshold: float = 0.4) -> str:
    """Classifies with a confidence floor — anything below threshold is
    labeled 'miscellaneous' rather than silently guessing wrong.
    NOTE: with a small training set, even in-sample transactions can score
    under 40% confidence (verified during testing), so this may catch more
    transactions than intended. Consider lowering `threshold` (e.g. 0.25)
    or expanding SAMPLE_TRANSACTIONS if 'miscellaneous' fills up too fast.
    Keep this in sync with the matching function in daily_summary.py."""
    X_new = vectorizer.transform([sms_text])
    probs = clf.predict_proba(X_new)[0]
    best_idx = probs.argmax()
    confidence = probs[best_idx]

    if confidence < threshold:
        return "miscellaneous"
    return clf.classes_[best_idx]


def forecast_breach(df, category, weekly_budget, week_start, week_end, today=None):
    """Checks current spend pace and forecasts if it'll exceed budget by week end."""
    if today is None:
        today = df["date"].max()

    week_data = df[
        (df["category"] == category)
        & (df["date"] >= week_start)
        & (df["date"] <= week_end)
    ]

    current_spend = week_data["amount"].sum()
    days_elapsed = max((today.date() - week_start.date()).days + 1, 1)
    total_days = (week_end - week_start).days + 1

    daily_rate = current_spend / days_elapsed
    projected_total = daily_rate * total_days
    will_breach = projected_total > weekly_budget

    return {
        "category": category,
        "current_spend": round(current_spend, 2),
        "projected_total": round(projected_total, 2),
        "budget": weekly_budget,
        "will_breach": will_breach,
    }


def send_telegram_alert(message: str) -> dict:
    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
    chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload, timeout=10)
    return response.json()


def generate_summary(summary_df: pd.DataFrame) -> dict:
    """Uses Gemini Flash-Lite to generate a structured spending summary.
    Returns a dict: {"tone": "good"|"warning", "headline": str, "tip": str}
    instead of a free-text blob, so the UI can render it consistently and
    color-code it by risk level.
    Retries on transient server-side errors (e.g. 503 high-demand) before giving up."""
    client = load_genai_client()
    prompt = (
        "You are a friendly financial assistant. Based on this weekly spending "
        "data, respond with ONLY a JSON object (no markdown, no code fences) "
        "in exactly this shape:\n"
        '{"tone": "good" or "warning", "headline": "one short sentence on the '
        'overall state", "tip": "one practical, encouraging tip tied to the '
        'riskiest category"}\n\n'
        "Set tone to \"warning\" only if at least one category is projected "
        "to breach its budget. Be warm and encouraging, never alarming.\n\n"
        f"{summary_df.to_string(index=False)}"
    )

    fallback = {
        "tone": "warning",
        "headline": "Summary unavailable",
        "tip": "",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=prompt
            )
            raw = response.text.strip()
            # Strip accidental markdown code fences, just in case
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.replace("json\n", "", 1) if raw.startswith("json\n") else raw

            parsed = json.loads(raw)

            # Validate shape before trusting it — never let a malformed
            # response crash the UI layer downstream
            if not all(k in parsed for k in ("tone", "headline", "tip")):
                raise ValueError("Missing expected keys in Gemini JSON response")
            if parsed["tone"] not in ("good", "warning"):
                parsed["tone"] = "warning"

            return parsed

        except Exception as e:
            is_last_attempt = attempt == max_retries - 1
            if is_last_attempt:
                fallback["headline"] = f"Summary unavailable after {max_retries} attempts"
                fallback["tip"] = str(e)
                return fallback
            time.sleep(2 ** attempt)  # 1s, then 2s backoff before retrying

    return fallback


def log_run_to_history(raw_df: pd.DataFrame, ai_summary: dict) -> None:
    """Best-effort logging — a logging failure should never break the dashboard."""
    try:
        history_ws = load_history_sheet()
        total_spent = float(raw_df["spent"].sum())
        total_budget = float(raw_df["budget"].sum())
        categories_over = int(raw_df["will_breach"].sum())

        history_ws.append_row([
            datetime.datetime.now().isoformat(timespec="seconds"),
            round(total_spent, 2),
            round(total_budget, 2),
            categories_over,
            ai_summary.get("tone", ""),
            ai_summary.get("headline", ""),
            ai_summary.get("tip", ""),
        ])
    except Exception:
        pass  # logging is non-critical; never break the main flow over it


def load_run_history(limit: int = 10) -> pd.DataFrame:
    """Fetches the most recent N logged runs, newest first."""
    try:
        history_ws = load_history_sheet()
        records = history_ws.get_all_records()
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        return df.tail(limit).iloc[::-1].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# 4. Main pipeline runner
# ──────────────────────────────────────────────────────────────
def run_spendsense_check():
    vectorizer, clf = load_classifier()
    sheet = load_sheet_client()

    data = sheet.get_all_records()
    if not data:
        return "No transactions found in the sheet yet.", pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()

    # Normalize column names — strips stray whitespace from Sheet headers
    # (e.g. "sms_text " with a trailing space) so lookups don't silently fail
    data = [{k.strip(): v for k, v in row.items()} for row in data]

    processed = []
    skipped_rows = 0
    for row in data:
        sms_text = str(row.get("sms_text", "")).strip()
        timestamp = str(row.get("timestamp", "")).strip()

        if not sms_text or not timestamp:
            skipped_rows += 1
            continue  # skip incomplete rows instead of crashing the whole app

        parsed = parse_transaction(sms_text)
        category = classify_transaction(sms_text, vectorizer, clf)
        processed.append(
            {
                "date": timestamp,
                "merchant": parsed["merchant"],
                "amount": parsed["amount"],
                "category": category,
            }
        )

    if not processed:
        return (
            f"No valid transactions found ({skipped_rows} row(s) skipped due to missing data).",
            pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(),
        )

    live_df = pd.DataFrame(processed)
    live_df["date"] = pd.to_datetime(live_df["date"])

    simulated_today = live_df["date"].max()
    week_start = live_df["date"].min().normalize()
    week_end = week_start + pd.Timedelta(days=6)

    summary_rows = []
    raw_rows = []
    alerts_fired = []

    for cat, budget in BUDGETS.items():
        result = forecast_breach(live_df, cat, budget, week_start, week_end, today=simulated_today)
        status = "⚠️ Over pace" if result["will_breach"] else "✅ On track"

        summary_rows.append(
            {
                "Category": cat.capitalize(),
                "Spent so far": f"₹{result['current_spend']}",
                "Projected": f"₹{result['projected_total']}",
                "Budget": f"₹{result['budget']}",
                "Status": status,
            }
        )
        raw_rows.append(
            {
                "category": cat.capitalize(),
                "spent": result["current_spend"],
                "projected": result["projected_total"],
                "budget": result["budget"],
                "will_breach": result["will_breach"],
            }
        )

        if result["will_breach"]:
            alert_message = (
                f"⚠️ Budget Alert: {cat.upper()}\n"
                f"Spent so far: ₹{result['current_spend']}\n"
                f"Projected by week end: ₹{result['projected_total']}\n"
                f"Budget: ₹{result['budget']}"
            )
            send_telegram_alert(alert_message)
            alerts_fired.append(cat)

    summary_df = pd.DataFrame(summary_rows)
    raw_df = pd.DataFrame(raw_rows)

    if alerts_fired:
        status_message = f"🔔 Alerts sent for: {', '.join(alerts_fired)}. Check your Telegram!"
    else:
        status_message = "✅ All categories on track. No alerts needed."

    ai_summary_data = generate_summary(summary_df)
    log_run_to_history(raw_df, ai_summary_data)

    return status_message, summary_df, ai_summary_data, raw_df, live_df


# ──────────────────────────────────────────────────────────────
# 5. Streamlit UI — fintech dashboard
# ──────────────────────────────────────────────────────────────
import plotly.graph_objects as go

st.set_page_config(page_title="SpendSense", page_icon="💸", layout="wide")

DASH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:        #0D1117;
    --surface:   #161B26;
    --surface2:  #1C2333;
    --ink:       #F3F4F6;
    --muted:     #9CA3AF;
    --accent1:   #4F7CFF;
    --accent2:   #A855F7;
    --good:      #34D399;
    --warn:      #F87171;
    --hair:      #262C3D;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: var(--bg); color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background-color: var(--surface); border-right: 1px solid var(--hair); }
.block-container { padding-top: 1.8rem; }

/* sidebar brand */
.ss-brand {
    display: flex; align-items: center; gap: 10px; padding: 4px 4px 22px 4px;
    border-bottom: 1px solid var(--hair); margin-bottom: 18px;
}
.ss-brand-icon {
    width: 30px; height: 30px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 14px; color: white;
}
.ss-brand-text { font-weight: 700; font-size: 1.05rem; color: var(--ink); }
.ss-nav-label {
    color: var(--muted); font-size: 0.7rem; letter-spacing: 0.08em;
    text-transform: uppercase; margin: 4px 0 10px 4px;
}

/* stat cards */
.ss-stat {
    background: var(--surface); border: 1px solid var(--hair); border-radius: 14px;
    padding: 18px 20px; height: 100%;
}
.ss-stat-label {
    color: var(--muted); font-size: 0.8rem; margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: center;
}
.ss-stat-value { font-size: 1.6rem; font-weight: 700; color: var(--ink); margin-bottom: 6px; }
.ss-stat-sub { font-size: 0.76rem; }
.ss-stat-sub.good { color: var(--good); }
.ss-stat-sub.warn { color: var(--warn); }
.ss-pill {
    font-size: 0.68rem; padding: 2px 9px; border-radius: 20px; font-weight: 600;
}
.ss-pill.good { background: rgba(52,211,153,0.15); color: var(--good); }
.ss-pill.warn { background: rgba(248,113,113,0.15); color: var(--warn); }

/* hero balance card */
.ss-hero {
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    border-radius: 16px; padding: 22px 24px; color: white; height: 100%;
}
.ss-hero-label { font-size: 0.82rem; opacity: 0.85; margin-bottom: 6px; }
.ss-hero-value { font-size: 2rem; font-weight: 700; margin-bottom: 4px; }
.ss-hero-sub { font-size: 0.78rem; opacity: 0.85; }

/* panel wrapper for charts */
.ss-panel {
    background: var(--surface); border: 1px solid var(--hair);
    border-radius: 14px; padding: 18px 20px 6px 20px; margin-bottom: 18px;
}
.ss-panel-title { font-size: 0.95rem; font-weight: 600; color: var(--ink); margin-bottom: 2px; }
.ss-panel-sub { font-size: 0.76rem; color: var(--muted); margin-bottom: 6px; }

/* transactions table */
.ss-tx-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 11px 4px; border-bottom: 1px solid var(--hair); font-size: 0.85rem;
}
.ss-tx-row:last-child { border-bottom: none; }
.ss-tx-merchant { font-weight: 600; color: var(--ink); }
.ss-tx-cat { color: var(--muted); font-size: 0.76rem; }
.ss-tx-amount { font-weight: 600; color: var(--ink); }

/* ai summary callout */
.ss-summary {
    background: var(--surface2); border-left: 3px solid var(--accent1);
    border-radius: 0 12px 12px 0; padding: 16px 20px; font-size: 0.9rem;
    line-height: 1.6; color: var(--ink); margin-top: 4px;
}
.ss-summary-label {
    font-size: 0.7rem; color: var(--muted); letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 8px; display: block; font-weight: 600;
}

/* past digests (sidebar history) */
.ss-hist-row {
    font-size: 0.78rem; padding: 6px 0; border-bottom: 1px solid var(--hair);
}
.ss-hist-row:last-child { border-bottom: none; }
.ss-hist-time { color: var(--muted); }

div.stButton > button {
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    color: white; border: none; font-weight: 600; border-radius: 10px;
    padding: 0.55rem 1.3rem;
}
div.stButton > button:hover { opacity: 0.92; color: white; }

.ss-footer { color: var(--muted); font-size: 0.75rem; margin-top: 28px; }
</style>
"""

st.markdown(DASH_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <div class="ss-brand">
            <div class="ss-brand-icon">S</div>
            <div class="ss-brand-text">SpendSense</div>
        </div>
        <div class="ss-nav-label">menu</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("🏠 &nbsp; Dashboard", unsafe_allow_html=True)
    st.markdown("📊 &nbsp; Reports", unsafe_allow_html=True)
    st.markdown("⚙️ &nbsp; Settings", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📜 Past digests**")
    history_df = load_run_history(limit=10)
    if history_df.empty:
        st.caption("No history yet — run a check to start your log.")
    else:
        for _, row in history_df.iterrows():
            icon = "✅" if row.get("tone") == "good" else "⚠️"
            spent_val = row.get("total_spent", 0)
            try:
                spent_val = f"{float(spent_val):,.0f}"
            except (TypeError, ValueError):
                spent_val = str(spent_val)
            st.markdown(
                f"<div class='ss-hist-row'>{icon} <b>₹{spent_val}</b> — "
                f"{row.get('headline', '')}<br>"
                f"<span class='ss-hist-time'>{str(row.get('timestamp', ''))[:16]}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Built for the DDS AI Application Building Challenge")

st.markdown("## Dashboard")
st.caption("Live view of your spending, categorized and forecasted by AI.")

run_clicked = st.button("🔍  Run the check")

if run_clicked:
    with st.spinner("Pulling transactions · classifying · forecasting..."):
        try:
            status_message, summary_df, ai_summary_data, raw_df, live_df = run_spendsense_check()
        except Exception as e:
            status_message, summary_df, ai_summary_data, raw_df, live_df = (
                f"error: {e}", pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame()
            )

    if summary_df.empty:
        st.warning(status_message)
        st.caption(
            "Double-check Streamlit secrets are filled in (Gemini key, Telegram "
            "token/chat ID, Google service account, sheet name)."
        )
    else:
        total_spent = raw_df["spent"].sum()
        total_budget = raw_df["budget"].sum()
        pct_used = (total_spent / total_budget * 100) if total_budget else 0
        over_count = int(raw_df["will_breach"].sum())

        # ── top row: hero balance + stat cards ─────────────────
        col_hero, col1, col2 = st.columns([1.3, 1, 1])

        with col_hero:
            st.markdown(
                f"""
                <div class="ss-hero">
                    <div class="ss-hero-label">Total spent this week</div>
                    <div class="ss-hero-value">₹{total_spent:,.0f}</div>
                    <div class="ss-hero-sub">{pct_used:.0f}% of ₹{total_budget:,.0f} weekly budget</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col1:
            pill_class = "warn" if over_count > 0 else "good"
            pill_text = f"{over_count} over pace" if over_count > 0 else "all on track"
            st.markdown(
                f"""
                <div class="ss-stat">
                    <div class="ss-stat-label">Categories <span class="ss-pill {pill_class}">{pill_text}</span></div>
                    <div class="ss-stat-value">{len(raw_df)}</div>
                    <div class="ss-stat-sub">tracked this week</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            tx_count = len(live_df) if not live_df.empty else 0
            st.markdown(
                f"""
                <div class="ss-stat">
                    <div class="ss-stat-label">Transactions</div>
                    <div class="ss-stat-value">{tx_count}</div>
                    <div class="ss-stat-sub">captured from live feed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── charts row: bar (spent vs budget) + donut (category split) ──
        col_bar, col_donut = st.columns([1.4, 1])

        with col_bar:
            st.markdown(
                """
                <div class="ss-panel">
                    <div class="ss-panel-title">Spent vs budget</div>
                    <div class="ss-panel-sub">By category, this week</div>
                """,
                unsafe_allow_html=True,
            )
            bar_colors = ["#F87171" if b else "#34D399" for b in raw_df["will_breach"]]
            fig_bar = go.Figure()
            fig_bar.add_bar(
                x=raw_df["category"], y=raw_df["budget"],
                name="Budget", marker_color="#262C3D", width=0.45,
            )
            fig_bar.add_bar(
                x=raw_df["category"], y=raw_df["spent"],
                name="Spent", marker_color=bar_colors, width=0.45,
            )
            fig_bar.update_layout(
                barmode="overlay", height=280, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9CA3AF", size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                yaxis=dict(gridcolor="#262C3D"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col_donut:
            st.markdown(
                """
                <div class="ss-panel">
                    <div class="ss-panel-title">Where it went</div>
                    <div class="ss-panel-sub">Share of total spend</div>
                """,
                unsafe_allow_html=True,
            )
            donut_colors = ["#4F7CFF", "#A855F7", "#F472B6", "#34D399"]
            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=raw_df["category"], values=raw_df["spent"],
                        hole=0.62, marker=dict(colors=donut_colors),
                        textinfo="label+percent", textfont=dict(color="#F3F4F6", size=11),
                    )
                ]
            )
            fig_donut.update_layout(
                height=280, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                annotations=[
                    dict(
                        text=f"₹{total_spent:,.0f}", x=0.5, y=0.5,
                        font=dict(size=16, color="#F3F4F6"), showarrow=False,
                    )
                ],
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── recent transactions ──────────────────────────────────
        if not live_df.empty:
            st.markdown(
                """
                <div class="ss-panel">
                    <div class="ss-panel-title">Recent transactions</div>
                    <div class="ss-panel-sub">From your live feed</div>
                """,
                unsafe_allow_html=True,
            )

            tx_filter = st.selectbox(
                "Filter by category",
                options=["All"] + sorted(live_df["category"].str.capitalize().unique().tolist()),
                key="tx_filter",
                label_visibility="collapsed",
            )

            filtered_df = live_df if tx_filter == "All" else live_df[live_df["category"].str.capitalize() == tx_filter]

            rows_html = ""
            for _, row in filtered_df.sort_values("date", ascending=False).head(15).iterrows():
                rows_html += f"""
                <div class="ss-tx-row">
                    <div>
                        <div class="ss-tx-merchant">{row['merchant']}</div>
                        <div class="ss-tx-cat">{row['category'].capitalize()}</div>
                    </div>
                    <div class="ss-tx-amount">₹{row['amount']:,.0f}</div>
                </div>
                """
            st.markdown(rows_html if rows_html else "<p style='color:var(--muted);font-size:0.85rem;'>No transactions in this category.</p>", unsafe_allow_html=True)
            st.markdown(
                f'<p style="color:var(--muted);font-size:0.75rem;margin-top:8px;">Showing {min(len(filtered_df), 15)} of {len(filtered_df)} transactions</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── status + AI summary ─────────────────────────────────
        status_color = "var(--warn)" if "🔔" in status_message else "var(--good)"
        st.markdown(
            f'<div style="color:{status_color};font-size:0.85rem;font-weight:600;margin-bottom:10px;">{status_message}</div>',
            unsafe_allow_html=True,
        )

        if ai_summary_data and ai_summary_data.get("headline"):
            tone = ai_summary_data.get("tone", "warning")
            border_color = "var(--good)" if tone == "good" else "var(--warn)"
            tone_icon = "✅" if tone == "good" else "⚠️"

            st.markdown(
                f"""
                <div class="ss-summary" style="border-left-color: {border_color};">
                    <span class="ss-summary-label">AI weekly read</span>
                    <div style="font-weight:600; margin-bottom:6px;">{tone_icon} {ai_summary_data['headline']}</div>
                    <div style="color:var(--muted); font-size:0.85rem;">{ai_summary_data.get('tip', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("Click **Run the check** to pull your latest transactions and see your dashboard.")

st.markdown(
    '<div class="ss-footer">SpendSense · DDS AI Application Building Challenge</div>',
    unsafe_allow_html=True,
)
