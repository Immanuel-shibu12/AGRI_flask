"""
Simple multi-language dictionary for AgriSchedule.
Add more languages by adding another key to LANGUAGES and filling in
the same set of string keys used across the app.
"""

LANGUAGES = {
    "English": "en",
    "മലയാളം (Malayalam)": "ml",
    "हिंदी (Hindi)": "hi",
    "தமிழ் (Tamil)": "ta",
}

STRINGS = {
    "app_title": {
        "en": "AgriSchedule — Smart Crop Calendar",
        "ml": "അഗ്രിഷെഡ്യൂൾ — സ്മാർട്ട് കൃഷി കലണ്ടർ",
        "hi": "एग्रीशेड्यूल — स्मार्ट फसल कैलेंडर",
        "ta": "அக்ரிஷெட்யூல் — ஸ்மார்ட் பயிர் காலண்டர்",
    },
    "nav_register": {
        "en": "Farmer Registration",
        "ml": "കർഷക രജിസ്ട്രേഷൻ",
        "hi": "किसान पंजीकरण",
        "ta": "விவசாயி பதிவு",
    },
    "nav_schedule": {
        "en": "My Crop Schedule",
        "ml": "എന്റെ കൃഷി ഷെഡ്യൂൾ",
        "hi": "मेरा फसल कार्यक्रम",
        "ta": "எனது பயிர் அட்டவணை",
    },
    "nav_weather": {
        "en": "Weather Advisory",
        "ml": "കാലാവസ്ഥാ ഉപദേശം",
        "hi": "मौसम सलाह",
        "ta": "வானிலை ஆலோசனை",
    },
    "nav_pest": {
        "en": "Pest & Disease Guidance",
        "ml": "കീട-രോഗ മാർഗ്ഗനിർദ്ദേശം",
        "hi": "कीट एवं रोग मार्गदर्शन",
        "ta": "பூச்சி & நோய் வழிகாட்டுதல்",
    },
    "nav_harvest": {
        "en": "Harvest Countdown",
        "ml": "വിളവെടുപ്പ് കൗണ്ട്ഡൗൺ",
        "hi": "फसल कटाई उलटी गिनती",
        "ta": "அறுவடை கவுண்ட்டவுன்",
    },
    "nav_schemes": {
        "en": "Govt. Schemes & Subsidies",
        "ml": "സർക്കാർ പദ്ധതികളും സബ്സിഡികളും",
        "hi": "सरकारी योजनाएं और सब्सिडी",
        "ta": "அரசு திட்டங்கள் & மானியங்கள்",
    },
    "nav_admin": {
        "en": "Admin: Crop Schedule Builder",
        "ml": "അഡ്മിൻ: വിള ഷെഡ്യൂൾ",
        "hi": "एडमिन: फसल शेड्यूल",
        "ta": "நிர்வாகி: பயிர் அட்டவணை",
    },
    "name": {"en": "Name", "ml": "പേര്", "hi": "नाम", "ta": "பெயர்"},
    "location": {"en": "Location (Village/Town)", "ml": "സ്ഥലം (ഗ്രാമം/പട്ടണം)", "hi": "स्थान (गांव/शहर)", "ta": "இடம் (கிராமம்/நகரம்)"},
    "crop": {"en": "Crop", "ml": "വിള", "hi": "फसल", "ta": "பயிர்"},
    "variety": {"en": "Variety", "ml": "ഇനം", "hi": "किस्म", "ta": "வகை"},
    "field_area": {"en": "Field Area (acres)", "ml": "വയലിന്റെ വിസ്തീർണ്ണം (ഏക്കർ)", "hi": "खेत का क्षेत्रफल (एकड़)", "ta": "வயல் பரப்பு (ஏக்கர்)"},
    "sowing_date": {"en": "Sowing / Transplanting Date", "ml": "വിത്ത് വിതയ്ക്കൽ/ നടീൽ തീയതി", "hi": "बुवाई / रोपाई तिथि", "ta": "விதைப்பு / நடவு தேதி"},
    "register_btn": {"en": "Register", "ml": "രജിസ്റ്റർ ചെയ്യുക", "hi": "पंजीकरण करें", "ta": "பதிவு செய்யவும்"},
}


def t(key: str, lang_code: str) -> str:
    entry = STRINGS.get(key, {})
    return entry.get(lang_code, entry.get("en", key))
