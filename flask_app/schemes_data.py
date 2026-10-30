"""
Government scheme & subsidy reference content.
NOTE: Scheme rules/deadlines change every season — treat this as a starting
reference for the demo, and always link out to the official portal for the
current, authoritative details before a farmer relies on it.
"""

SCHEMES = [
    {
        "name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
        "summary": "Direct income support of ₹6,000/year (in 3 installments) to eligible landholding farmer families, paid via DBT to bank accounts.",
        "who": "Small and marginal landholding farmer families (subject to exclusion criteria for income-tax payers, institutional landholders, etc.).",
        "link": "https://pmkisan.gov.in",
    },
    {
        "name": "PMFBY (Pradhan Mantri Fasal Bima Yojana)",
        "summary": "Crop insurance covering losses from natural calamities, pests and diseases, from pre-sowing to post-harvest stage, at heavily subsidised premiums (about 2% for Kharif, 1.5% for Rabi crops).",
        "who": "All farmers (owner-cultivators, tenants, sharecroppers) growing notified crops in notified areas. Compulsory for loanee farmers with a Kisan Credit Card/crop loan; voluntary for others.",
        "link": "https://pmfby.gov.in",
    },
    {
        "name": "Kisan Credit Card (KCC)",
        "summary": "Low-interest short-term credit for crop production, post-harvest expenses, and allied activities.",
        "who": "Farmers, tenant farmers, and sharecroppers with valid land/crop records.",
        "link": "https://www.myscheme.gov.in",
    },
    {
        "name": "Soil Health Card Scheme",
        "summary": "Free soil testing every 2-3 years with crop-wise nutrient and fertilizer recommendations for the farmer's own field.",
        "who": "All farmers; apply through the local agriculture department/Krishi Bhavan.",
        "link": "https://soilhealth.dac.gov.in",
    },
    {
        "name": "Kerala-specific: Karshaka Pension / State horticulture & subsidy schemes",
        "summary": "Kerala runs additional state-level input subsidies (seeds, drip irrigation, farm mechanization) through the local Krishi Bhavan.",
        "who": "Varies by scheme — check with your local Krishi Bhavan / Agriculture Officer.",
        "link": "https://keralaagriculture.gov.in",
    },
]

DISCLAIMER = (
    "Scheme rules, subsidy amounts, and application deadlines change frequently. "
    "This screen is a starting reference for the demo — always confirm current details "
    "on the official portal or with your local Krishi Bhavan/Agriculture Office before applying."
)
