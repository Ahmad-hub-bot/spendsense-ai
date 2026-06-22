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
import requests
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
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
    "food": 1500,
    "shopping": 3000,
    "travel": 1000,
    "daily": 800,
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
def load_sheet_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    sheet_name = st.secrets.get("GSHEET_NAME", "SpendSense Live Feed")
    return client.open(sheet_name).sheet1


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

    merchant_match = re.findall(r"\b[A-Z]{3,}(?:\s[A-Z]{2,})*\b", sms_text)
    ignore_words = {"INR", "RS", "UPI", "CARD"}
    merchants = [m for m in merchant_match if m not in ignore_words]
    merchant = merchants[0] if merchants else "UNKNOWN"

    return {"amount": amount, "merchant": merchant, "raw_text": sms_text}


def classify_transaction(sms_text: str, vectorizer, clf) -> str:
    X_new = vectorizer.transform([sms_text])
    return clf.predict(X_new)[0]


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


def generate_summary(summary_df: pd.DataFrame) -> str:
    """Uses Gemini Flash-Lite to generate a short natural-language spending summary."""
    try:
        client = load_genai_client()
        prompt = (
            "You are a friendly financial assistant. Based on this weekly spending "
            "data, write a 2-3 sentence summary highlighting any categories at risk "
            "and one practical tip. Be encouraging, not alarming.\n\n"
            f"{summary_df.to_string(index=False)}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt
        )
        return response.text
    except Exception as e:
        return f"(Summary unavailable: {e})"


# ──────────────────────────────────────────────────────────────
# 4. Main pipeline runner
# ──────────────────────────────────────────────────────────────
def run_spendsense_check():
    vectorizer, clf = load_classifier()
    sheet = load_sheet_client()

    data = sheet.get_all_records()
    if not data:
        return "No transactions found in the sheet yet.", pd.DataFrame(), ""

    processed = []
    for row in data:
        sms_text = row["sms_text"]
        parsed = parse_transaction(sms_text)
        category = classify_transaction(sms_text, vectorizer, clf)
        processed.append(
            {
                "date": row["timestamp"],
                "merchant": parsed["merchant"],
                "amount": parsed["amount"],
                "category": category,
            }
        )

    live_df = pd.DataFrame(processed)
    live_df["date"] = pd.to_datetime(live_df["date"])

    simulated_today = live_df["date"].max()
    week_start = live_df["date"].min().normalize()
    week_end = week_start + pd.Timedelta(days=6)

    summary_rows = []
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

    if alerts_fired:
        status_message = f"🔔 Alerts sent for: {', '.join(alerts_fired)}. Check your Telegram!"
    else:
        status_message = "✅ All categories on track. No alerts needed."

    ai_summary = generate_summary(summary_df)

    return status_message, summary_df, ai_summary


# ──────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="SpendSense", page_icon="💰", layout="centered")

st.title("💰 SpendSense")
st.caption(
    "AI-powered spending monitor — auto-categorizes transactions and predicts "
    "budget overruns before they happen."
)

if st.button("🔍 Check My Spending", type="primary"):
    with st.spinner("Pulling live transactions and running the AI pipeline..."):
        try:
            status_message, summary_df, ai_summary = run_spendsense_check()

            if "Error" in status_message or summary_df.empty:
                st.warning(status_message)
            else:
                st.success(status_message)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                if ai_summary:
                    st.subheader("📝 AI Weekly Summary")
                    st.write(ai_summary)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info(
                "Double-check your Streamlit secrets are filled in correctly "
                "(Gemini key, Telegram token/chat ID, Google service account, sheet name)."
            )
else:
    st.info("Click the button above to pull your latest spending and check it against your budgets.")

st.divider()
st.caption("Built for the Decoding Data Science (DDS) AI Application Building Challenge.")
