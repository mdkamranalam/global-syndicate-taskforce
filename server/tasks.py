SCENARIOS = {
    "TX-1024": {
        "difficulty": "Easy",
        "true_kyc": "CLEAN_VERIFIED",
        "hallucinated_kyc": "CLEAN_VERIFIED", # Analyst is actually right here!
        "required_mandate": "Standard_Audit"
    },
    "TX-5590": {
        "difficulty": "Medium",
        "true_kyc": "STOLEN_IDENTITY",
        "hallucinated_kyc": "CLEAN_VERIFIED", # Analyst hallucinates
        "required_mandate": "AML_Directive_4"
    },
    "TX-7742": {
        "difficulty": "Hard",
        "true_kyc": "SYNTHETIC_FRAUD",
        "hallucinated_kyc": "CLEAN_VERIFIED", # Analyst hallucinates
        "required_mandate": "IndiaStack_Section_9"
    }
}