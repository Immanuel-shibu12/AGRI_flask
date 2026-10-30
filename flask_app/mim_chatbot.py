"""
Mim — AgriSchedule's in-app assistant.

Backend is pluggable via environment variables so no code changes are
needed to go live:

    GROK_API_KEY        - required. Your API key (Groq, Grok/x.ai, or any
                           other OpenAI-compatible provider).
    GROK_API_BASE        - optional. Defaults to https://api.x.ai/v1.
                           Set to https://api.groq.com/openai/v1 for Groq.
    GROK_MODEL           - optional. Defaults to "grok-4".
    GROK_WHISPER_MODEL   - optional. Defaults to "whisper-large-v3"
                           (used for voice input transcription).

Until GROK_API_KEY is set, ask_mim() returns a friendly placeholder
message instead of failing, so the chat UI stays usable during
development.
"""

import os

import requests

API_KEY = os.environ.get("GROK_API_KEY", "").strip()
API_BASE = os.environ.get("GROK_API_BASE", "https://api.x.ai/v1").strip().rstrip("/")
MODEL = os.environ.get("GROK_MODEL", "grok-4").strip()

SYSTEM_PROMPT = (
    "You are Mim, the friendly in-app assistant for AgriSchedule, a crop "
    "calendar and farming-advisory app. Give practical, concise farming "
    "guidance in plain language. When the user's crop, location, or "
    "schedule context is provided to you, use it directly rather than "
    "asking the user to repeat it. Always reply in the same language the "
    "user just wrote or spoke in, even if earlier turns were in a "
    "different language. If a preferred app language is given in the "
    "context and the user's message doesn't make the language clear "
    "(e.g. it's very short), reply in that preferred language."
)


def is_configured():
    """True once a real API key has been set via environment variable."""
    return bool(API_KEY)


def ask_mim(user_message, history=None, context=None):
    """Send a message to the configured LLM and return the reply text.

    history: optional list of {"role": "user"/"assistant", "content": str}
             (recent turns only — keep this short to control token usage)
    context: optional string describing the current farmer/crop/schedule,
             injected as extra system context for a grounded answer
    """
    if not is_configured():
        q = user_message.lower()
        if "fertilizer" in q or "urea" in q or "nutrient" in q or "npk" in q:
            return "[Fertilizer Advisory]: For optimal yield, split nitrogen (Urea) into 2-3 top dressings — first during active tillering (day 20-25) and second at panicle initiation (day 50-55). Avoid applying nitrogen top dressing during heavy rains to prevent leaching."
        elif "pest" in q or "insect" in q or "bug" in q or "spray" in q or "borer" in q:
            return "[Pest Management Advisory]: Always scout at dusk and inspect the undersides of leaves. For stem borer or leaf folders, use neem-based biopesticides (Azadirachtin 0.03%) at first sighting. Only apply chemical sprays if economic threshold levels are crossed, and avoid spraying on windy or rainy days."
        elif "water" in q or "irrigation" in q or "rain" in q:
            return "[Irrigation Advisory]: Critical irrigation stages for rice include transplanting and flowering (keep 3-5 cm standing water). For wheat, the Crown Root Initiation (CRI at day 21) is the most critical stage. If heavy rain is forecast, postpone irrigation to conserve water and prevent waterlogging."
        elif "weed" in q:
            return "[Weeding Advisory]: Perform the first manual weeding or post-emergence weed control at 20-25 days after sowing/transplanting before weeds compete for nutrients. A second weeding around day 45 ensures clean fields before flowering."
        elif "harvest" in q:
            return "[Harvest Advisory]: Plan harvest when 80-85% of panicles or grains turn golden yellow and moisture content drops to 20-25%. Check the 3-day weather forecast beforehand to ensure dry weather for harvesting and sun-drying."
        elif "scheme" in q or "subsidy" in q or "pm-kisan" in q or "pmfby" in q:
            return "[Govt Schemes]: You can apply for PM-KISAN (Rs 6,000/yr direct income support) at pmkisan.gov.in, and PMFBY for crop insurance at subsidized premium rates. Visit our 'Govt. Schemes' section on the left sidebar for official direct links."
        elif "hello" in q or "hi" in q or "hey" in q:
            farmer_mention = f" for your {context}" if context else ""
            return f"Hello, I am Mim, your AgriSchedule agricultural assistant. How can I help you today{farmer_mention}? You can ask me about crop schedules, fertilizer doses, pest management, weather alerts, or government schemes."
        else:
            return (
                "Hello, I am Mim, your agricultural assistant. I can answer questions about crop management, fertilizers, pests, and weather advisories. "
                "(Tip: For live LLM reasoning, you can also set the GROK_API_KEY in your environment or .env file)."
            )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": f"Current farmer context:\n{context}"})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(
            f"{API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": MODEL, "messages": messages, "temperature": 0.4},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Mim couldn't reach the AI backend right now ({e}). Try again in a moment."
    except (KeyError, IndexError, ValueError):
        return "Mim got back an unexpected response from the AI backend. Please try again."


def transcribe_audio(audio_bytes, filename="voice_note.wav"):
    """Send recorded audio to the configured provider's Whisper-compatible
    transcription endpoint and return (text, error). Works with Groq
    (whisper-large-v3) and any other OpenAI-compatible provider that
    exposes /audio/transcriptions."""
    if not is_configured():
        return None, (
            "Voice input needs the same `GROK_API_KEY` used for chat — "
            "set it in your .env file."
        )

    whisper_model = os.environ.get("GROK_WHISPER_MODEL", "whisper-large-v3").strip()

    try:
        resp = requests.post(
            f"{API_BASE}/audio/transcriptions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": (filename, audio_bytes, "audio/wav")},
            data={"model": whisper_model},
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        return (text or None), (None if text else "Didn't catch any speech in that recording.")
    except requests.exceptions.RequestException as e:
        return None, f"Couldn't transcribe audio right now ({e})."
    except (KeyError, ValueError):
        return None, "Got an unexpected response while transcribing audio."
