"""
Reminders — lets the app owner notify a farmer who has missed a scheduled
operation, by SMS or phone call.

Backend is Twilio, configured via environment variables so no code
changes are needed to go live:

    TWILIO_ACCOUNT_SID   - required
    TWILIO_AUTH_TOKEN    - required
    TWILIO_FROM_NUMBER   - required. Your Twilio phone number, e.g. +15551234567

Requires the `twilio` package:  pip install twilio

Until all three variables are set, send_sms() / make_call() return
(False, <helpful message>) instead of raising, so the Reminders page
stays usable during development.
"""

import os

TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM = os.environ.get("TWILIO_FROM_NUMBER", "").strip()

NOT_CONFIGURED_MSG = (
    "Reminders aren't connected yet — set TWILIO_ACCOUNT_SID, "
    "TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER in your .env file."
)


def is_configured():
    return bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM)


def _client():
    from twilio.rest import Client
    return Client(TWILIO_SID, TWILIO_TOKEN)


def send_sms(to_phone, message):
    """Returns (success: bool, detail: str)."""
    if not is_configured():
        return False, NOT_CONFIGURED_MSG
    if not to_phone:
        return False, "This farmer has no phone number on file."
    try:
        _client().messages.create(to=to_phone, from_=TWILIO_FROM, body=message)
        return True, f"SMS sent to {to_phone}."
    except Exception as e:
        return False, f"Couldn't send SMS: {e}"


def make_call(to_phone, message):
    """Places a call that reads `message` aloud via Twilio's text-to-speech.
    Returns (success: bool, detail: str)."""
    if not is_configured():
        return False, NOT_CONFIGURED_MSG
    if not to_phone:
        return False, "This farmer has no phone number on file."
    try:
        escaped = message.replace("&", "and").replace("<", "").replace(">", "")
        twiml = f"<Response><Say>{escaped}</Say></Response>"
        _client().calls.create(to=to_phone, from_=TWILIO_FROM, twiml=twiml)
        return True, f"Call placed to {to_phone}."
    except Exception as e:
        return False, f"Couldn't place call: {e}"
