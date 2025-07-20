"""
Day 21 Example: ReAct Agent Workflow (Multi-Step Reasoning, OpenAI GPT Version, Updated for openai>=1.0.0)

This script demonstrates a simple ReAct-style agent loop using OpenAI's GPT models:
- The agent receives a complex question
- It reasons about the steps needed
- Selects and uses tools (search, calculator)
- Observes results and iterates until the goal is achieved

Requires: openai>=1.0.0, requests
Set your OpenAI API key as the environment variable OPENAI_API_KEY.
"""

import os
import openai
import requests
import re

# --- Tools ---
def search_tool(query):
    """Uses DuckDuckGo Instant Answer API to get a short answer."""
    url = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1&no_html=1"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get('AbstractText') or data.get('Answer') or data.get('Definition')
        if answer:
            return answer
        topics = data.get('RelatedTopics')
        if topics and isinstance(topics, list) and 'Text' in topics[0]:
            return topics[0]['Text']
        return "No answer found."
    except Exception as e:
        return f"[search_tool error: {e}]"

def calculator_tool(expression):
    """Evaluates a simple math expression (e.g., '5 * 1000000')."""
    try:
        if not re.match(r'^[\d\s\*\+\-/\.]+$', expression):
            return "Invalid expression."
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"[calculator_tool error: {e}]"

TOOLS = {
    'search': search_tool,
    'calculator': calculator_tool
}

# --- Few-shot ReAct Example ---
FEW_SHOT = """
User question: What is the capital of France and what is 2 plus 2?
Thought: The user is asking two things: the capital of France and a math question. I'll start by finding the capital of France.
Action: search("capital of France")
Observation: The capital of France is Paris.
Thought: Now I need to answer the math question: 2 plus 2.
Action: calculator("2 + 2")
Observation: 4
Thought: I have both answers. I can now answer the user's question.
Action: finish("The capital of France is Paris and 2 plus 2 is 4.")
"""

# --- Agent Loop ---
def gpt_react_agent(user_input, max_steps=5, model="gpt-3.5-turbo"):
    print(f"User: {user_input}\n")
    client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment
    messages = [
        {"role": "system", "content": (
            "You are a helpful agent that can use tools to answer complex questions. "
            "For each step, think out loud (Thought), then choose an action in the format Action: tool_name(input). "
            "After the action, you will receive an Observation. "
            "Repeat Thought -> Action -> Observation until you can answer the user's question. "
            "When you are done, respond with Action: finish(final_answer). "
            "Available tools: search, calculator.\n" + FEW_SHOT
        )}
    ]
    messages.append({"role": "user", "content": f"User question: {user_input}"})
    scratchpad = []
    for step in range(max_steps):
        # Add scratchpad to the conversation
        for entry in scratchpad:
            messages.append({"role": "assistant", "content": entry})
        # Add the next Thought prompt
        messages.append({"role": "assistant", "content": "Thought:"})
        # Get LLM output
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=256,
            stop=None
        )
        llm_out = response.choices[0].message.content
        new_lines = llm_out.strip().splitlines()
        if not new_lines:
            print(f"[Agent Error] LLM did not return output. Raw output: {repr(llm_out)}")
            break
        thought = new_lines[0].strip()
        action_line = None

        # Check if the thought line contains an embedded Action:
        if "Action:" in thought:
            thought_split = thought.split("Action:", 1)
            thought = thought_split[0].strip()
            action_line = "Action:" + thought_split[1].strip()
        else:
            # Look for Action: in subsequent lines
            action_line = next((l for l in new_lines[1:] if l.startswith("Action:")), None)

        print(f"Thought: {thought}")
        if not action_line:
            print("[Agent Error] No action found.")
            break
        print(action_line)
        scratchpad.append(f"Thought: {thought}")
        scratchpad.append(action_line)
        # Parse and execute action
        m = re.match(r'Action:\s*(\w+)\((.*)\)', action_line)
        if not m:
            print("[Agent Error] Could not parse action.")
            break
        tool, arg = m.group(1), m.group(2)
        tool = tool.strip().lower()
        arg = arg.strip().strip('"')
        if tool not in TOOLS:
            obs = f"[Unknown tool: {tool}]"
        else:
            obs = TOOLS[tool](arg)
        print(f"Observation: {obs}\n")
        scratchpad.append(f"Observation: {obs}")
    else:
        print("[Agent Error] Reached step limit without finishing.")

if __name__ == "__main__":
    user_input = "What is the population of Paris, and what is 5 times that number?"
    gpt_react_agent(user_input) 