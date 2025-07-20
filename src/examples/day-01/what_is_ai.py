"""
Day 1 Example: What is AI? - Simple Rule-Based Expert System
This script demonstrates a basic AI using rule-based logic to make decisions.
"""

def ai_expert_system(symptoms):
    """A simple expert system for diagnosing a cold or flu based on symptoms."""
    if 'fever' in symptoms and 'cough' in symptoms and 'body ache' in symptoms:
        return "Diagnosis: You may have the flu."
    elif 'cough' in symptoms and 'sore throat' in symptoms:
        return "Diagnosis: You may have a common cold."
    elif 'headache' in symptoms and 'nausea' in symptoms:
        return "Diagnosis: You may have a migraine."
    else:
        return "Diagnosis: Insufficient information. Please consult a doctor."

if __name__ == "__main__":
    print("--- Simple Rule-Based AI (Expert System) ---")
    print("Describe your symptoms (comma-separated):")
    user_input = input().lower()
    symptoms = [s.strip() for s in user_input.split(',')]
    result = ai_expert_system(symptoms)
    print(result) 