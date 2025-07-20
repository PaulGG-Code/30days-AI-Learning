"""
Day 22 Example: Multi-Agent Collaboration (OpenAI GPT Version, Updated for openai>=1.0.0)

This script demonstrates a simple multi-agent system:
- A user asks for a Python function to filter even numbers from a list, and for unit tests for that function.
- Three agents collaborate: Planner (breaks down the task), Coder (writes the function), Tester (writes the tests).
- Agents communicate by passing messages/results.

Requires: openai>=1.0.0
Set your OpenAI API key as the environment variable OPENAI_API_KEY.
"""

import os
import openai

USER_TASK = "Write a Python function that takes a list of numbers and returns only the even ones. Then, write unit tests for it."

# --- Agent definitions ---
def agent_gpt(role, system_message, user_message, model="gpt-3.5-turbo"):
    client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

# 1. Planner agent breaks down the task
def planner_agent(task):
    system_message = (
        "You are a Planner agent. Your job is to break down the user's request into clear, actionable steps. "
        "Be concise and use numbered steps."
    )
    return agent_gpt("Planner", system_message, task)

# 2. Coder agent writes the function
def coder_agent(plan):
    system_message = (
        "You are a Coder agent. Your job is to write Python code to accomplish the first step of the plan. "
        "Return only the code, no explanation."
    )
    return agent_gpt("Coder", system_message, plan)

# 3. Tester agent writes unit tests
def tester_agent(function_code, plan):
    system_message = (
        "You are a Tester agent. Your job is to write Python unit tests for the function provided, "
        "based on the user's plan. Return only the test code, no explanation."
    )
    user_message = f"Function code:\n{function_code}\n\nPlan:\n{plan}"
    return agent_gpt("Tester", system_message, user_message)

# --- Multi-agent workflow ---
def multi_agent_workflow(user_task):
    print(f"User: {user_task}\n")
    # Step 1: Planner
    plan = planner_agent(user_task)
    print("[Planner] Task breakdown:")
    print(plan + "\n")
    # Step 2: Coder
    function_code = coder_agent(plan)
    print("[Coder] Function code:")
    print(function_code + "\n")
    # Step 3: Tester
    test_code = tester_agent(function_code, plan)
    print("[Tester] Unit tests:")
    print(test_code + "\n")

if __name__ == "__main__":
    multi_agent_workflow(USER_TASK) 