"""
SpendSense — Daily Summary Script
Runs independently of the Streamlit app, triggered by GitHub Actions on a schedule.

Sends one consolidated evening summary to Telegram, regardless of whether
any category has breached its budget — a daily "here's where you stand" digest.

Required environment variables / secrets (set in GitHub repo Settings -> Secrets -> Actions):
    GEMINI_API_KEY
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
    GSHEET_NAME           (optional, defaults to "SpendSense Live Feed")
    GCP_SERVICE_ACCOUNT_JSON   (the full service account JSON, as a single-line string)
"""

import os
import re
import json
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ──────────────────────────────────────────────────────────────
# Config — same training data and budgets as app.py (keep these in sync!)
# ──────────────────────────────────────────────────────────────
SAMPLE_TRANSACTIONS = [
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
LABELS = (["food"] * 10 + ["shopping"] * 10 + ["travel"] * 10 + ["daily"] * 10)

BUDGETS = {
    "food": 15000,
    "shopping": 3000,
    "travel": 1000,
    "daily": 800,
}


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


def load_sheet_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_dict = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    sheet_name = os.environ.get("GSHEET_NAME", "SpendSense Live Feed")
    return client.open(sheet_name).sheet1


def parse_transaction(sms_text: str) -> dict:
    amount_match = re.search(r"(?:INR|Rs\.?)\s?([\d,]+\.?\d*)", sms_text, re.IGNORECASE)
    amount = float(amount_match.group(1).replace(",", "")) if amount_match else None

    merchant_match = re.findall(r"\b[A-Z][a-zA-Z&]{2,}(?:\s[A-Z][a-zA-Z&]{2,})*\b", sms_text)
    ignore_words = {
        "INR", "RS", "UPI", "CARD", "YOUR", "ACCOUNT", "FROM", "FOR",
        "YOU", "USING", "SPENT", "DEBITED", "PAID", "ENDING", "TO",
    }
    merchants = [m for m in merchant_match if m.upper() not in ignore_words]

    if merchants:
        merchant = merchants[0]
    else:
        fallback_match = re.search(
            r"(?:for|at|to)\s+([a-zA-Z\s]+?)(?:\s+via|\s+delivery|$)", sms_text, re.IGNORECASE
        )
        merchant = fallback_match.group(1).strip().title() if fallback_match else "UNKNOWN"

    return {"amount": amount, "merchant": merchant, "raw_text": sms_text}


def classify_transaction(sms_text, vectorizer, clf):
    X_new = vectorizer.transform([sms_text])
    return clf.predict(X_new)[0]


def forecast_breach(df, category, weekly_budget, week_start, week_end, today=None):
    if today is None:
        today = df["date"].max()

    week_data = df[
        (df["category"] == category) & (df["date"] >= week_start) & (df["date"] <= week_end)
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


def send_telegram_message(message: str) -> dict:
    bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload, timeout=10)
    return response.json()


def generate_ai_summary(summary_lines: str) -> str:
    """Best-effort AI summary — falls back silently if Gemini is unavailable."""
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        prompt = (
            "You are a friendly financial assistant. Based on this end-of-day "
            "spending snapshot, write a 2-3 sentence evening summary. Be warm "
            "and encouraging, not alarming, even if a category is over pace.\n\n"
            f"{summary_lines}"
        )
        response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
        return response.text
    except Exception:
        return ""


def main():
    vectorizer, clf = load_classifier()
    sheet = load_sheet_client()

    data = sheet.get_all_records()
    data = [{k.strip(): v for k, v in row.items()} for row in data]

    processed = []
    for row in data:
        sms_text = str(row.get("sms_text", "")).strip()
        timestamp = str(row.get("timestamp", "")).strip()
        if not sms_text or not timestamp:
            continue
        parsed = parse_transaction(sms_text)
        category = classify_transaction(sms_text, vectorizer, clf)
        processed.append({
            "date": timestamp,
            "merchant": parsed["merchant"],
            "amount": parsed["amount"],
            "category": category,
        })

    if not processed:
        send_telegram_message("🌙 SpendSense Evening Digest\n\nNo transactions recorded today.")
        return

    live_df = pd.DataFrame(processed)
    live_df["date"] = pd.to_datetime(live_df["date"])

    simulated_today = live_df["date"].max()
    week_start = live_df["date"].min().normalize()
    week_end = week_start + pd.Timedelta(days=6)

    lines = ["🌙 SpendSense Evening Digest", ""]
    summary_for_ai = []

    for cat, budget in BUDGETS.items():
        result = forecast_breach(live_df, cat, budget, week_start, week_end, today=simulated_today)
        status_icon = "⚠️" if result["will_breach"] else "✅"
        lines.append(
            f"{status_icon} {cat.capitalize()}: ₹{result['current_spend']} spent "
            f"(projected ₹{result['projected_total']} / ₹{result['budget']} budget)"
        )
        summary_for_ai.append(
            f"{cat}: spent {result['current_spend']}, projected {result['projected_total']}, "
            f"budget {result['budget']}, breach={result['will_breach']}"
        )

    ai_note = generate_ai_summary("\n".join(summary_for_ai))
    if ai_note:
        lines.append("")
        lines.append(f"📝 {ai_note}")

    send_telegram_message("\n".join(lines))


if __name__ == "__main__":
    main()
