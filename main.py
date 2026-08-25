import os
import requests

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = "-1003921675472"

print("--- TELEGRAM DIRECT TEST START ---")
if not BOT_TOKEN:
    print("❌ ERROR: BOT_TOKEN GitHub Secrets mein nahi mil raha hai!")
else:
    print(f"Bot Token Loaded: {BOT_TOKEN[:6]}... (Hidden)")

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
payload = {"chat_id": CHAT_ID, "text": "🔥 *DIRECT TEST SUCCESS:* Telegram setup 100% working hai!"}

res = requests.post(url, json=payload)
print(f"TELEGRAM API STATUS CODE: {res.status_code}")
print(f"TELEGRAM API RESPONSE: {res.text}")
