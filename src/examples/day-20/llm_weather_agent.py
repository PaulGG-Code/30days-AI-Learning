"""
Day 20 Example: LLM-Powered Weather Agent (Real-World Scenario)

This script demonstrates a simple AI agent that:
- Takes a user goal (e.g., 'What's the weather in Paris?')
- Uses an LLM to plan and summarize
- Fetches real weather data from a public API
- Returns a natural language summary

Requires: transformers, torch, requests
"""

from transformers import pipeline
import requests

# 1. User goal
user_goal = "What's the weather in Paris?"
print(f"User Goal: {user_goal}")

# 2. Agent uses LLM to extract city name (planning step)
llm = pipeline("text2text-generation", model="google/flan-t5-small")
city_prompt = f"Extract the city name from this request: '{user_goal}'"
city = llm(city_prompt, max_length=10)[0]['generated_text'].strip()
print(f"[Agent Reasoning] Extracted city: {city}")

# 3. Agent uses a weather API to fetch real data
# We'll use wttr.in for simplicity (no API key required)
def get_weather(city):
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        # Extract current condition
        current = data['current_condition'][0]
        temp_c = current['temp_C']
        weather_desc = current['weatherDesc'][0]['value']
        humidity = current['humidity']
        wind_kph = current['windspeedKmph']
        return {
            'temp_c': temp_c,
            'desc': weather_desc,
            'humidity': humidity,
            'wind_kph': wind_kph
        }
    except Exception as e:
        print(f"[Agent Error] Failed to fetch weather: {e}")
        return None

weather = get_weather(city)
if not weather:
    print("Sorry, I couldn't retrieve the weather information.")
    exit(1)

print(f"[Agent Action] Weather data: {weather}")

# 4. Agent uses LLM to summarize the weather in natural language
summary_prompt = (
    f"Summarize the current weather in {city} given this data: "
    f"Temperature: {weather['temp_c']}°C, "
    f"Condition: {weather['desc']}, "
    f"Humidity: {weather['humidity']}%, "
    f"Wind: {weather['wind_kph']} kph."
)
summary = llm(summary_prompt, max_length=60)[0]['generated_text'].strip()

print(f"\n[Agent Final Answer] {summary}") 