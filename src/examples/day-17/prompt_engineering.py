"""
Prompt Engineering Examples and Techniques

This module demonstrates various prompt engineering techniques
for working with Large Language Models (LLMs).

Note: This is a conceptual demonstration. In practice, you would
use actual LLM APIs like OpenAI's GPT, Anthropic's Claude, etc.
"""

class PromptEngineeringDemo:
    def __init__(self):
        self.examples = {}
    
    def basic_prompting(self):
        """Demonstrate basic prompting techniques"""
        examples = {
            "poor_prompt": "Write about AI",
            "good_prompt": "Write a 200-word explanation of artificial intelligence for a high school student, focusing on how AI is used in everyday applications like smartphones and social media."
        }
        return examples
    
    def few_shot_prompting(self):
        """Demonstrate few-shot learning with examples"""
        prompt = """
Task: Classify the sentiment of movie reviews as Positive, Negative, or Neutral.

Examples:
Review: "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
Sentiment: Positive

Review: "I found this film quite boring. The pacing was slow and the characters were uninteresting."
Sentiment: Negative

Review: "The movie was okay. Some parts were good, others not so much."
Sentiment: Neutral

Now classify this review:
Review: "An incredible masterpiece! Every scene was beautifully crafted and the soundtrack was amazing."
Sentiment: """
        return prompt
    
    def chain_of_thought_prompting(self):
        """Demonstrate chain-of-thought reasoning"""
        prompt = """
Problem: A store sells apples for $2 per pound and oranges for $3 per pound. 
If Sarah buys 4 pounds of apples and 2 pounds of oranges, how much does she spend in total?

Let me solve this step by step:

Step 1: Calculate the cost of apples
- Apples cost $2 per pound
- Sarah buys 4 pounds of apples
- Cost of apples = 4 × $2 = $8

Step 2: Calculate the cost of oranges
- Oranges cost $3 per pound
- Sarah buys 2 pounds of oranges
- Cost of oranges = 2 × $3 = $6

Step 3: Calculate the total cost
- Total cost = Cost of apples + Cost of oranges
- Total cost = $8 + $6 = $14

Therefore, Sarah spends $14 in total.

Now solve this problem using the same step-by-step approach:
Problem: A library charges $0.50 per day for late book returns. If Tom returns 3 books that are each 5 days late, how much does he owe?
"""
        return prompt
    
    def role_based_prompting(self):
        """Demonstrate role-based prompting"""
        examples = {
            "teacher_role": "You are an experienced high school biology teacher. Explain photosynthesis in a way that would help students remember it for their exam.",
            "expert_role": "You are a cybersecurity expert with 15 years of experience. Provide advice on protecting a small business from common cyber threats.",
            "creative_role": "You are a creative writing coach. Help me brainstorm unique plot ideas for a science fiction short story."
        }
        return examples
    
    def structured_prompting(self):
        """Demonstrate structured output prompting"""
        prompt = """
Analyze the following business idea and provide feedback in the specified format:

Business Idea: "A mobile app that uses AI to help people identify plants by taking photos of them."

Please structure your response as follows:

## Market Analysis
[Assess the market potential and target audience]

## Strengths
[List 3 main strengths of this idea]

## Weaknesses
[List 3 potential challenges or weaknesses]

## Recommendations
[Provide 2-3 specific recommendations for improvement]

## Overall Rating
[Rate from 1-10 with brief justification]
"""
        return prompt
    
    def constraint_prompting(self):
        """Demonstrate prompting with constraints"""
        examples = {
            "length_constraint": "Explain quantum computing in exactly 50 words.",
            "style_constraint": "Explain machine learning as if you're a pirate captain teaching your crew.",
            "format_constraint": "List the top 5 programming languages for beginners. Format your response as a numbered list with exactly one sentence explaining each language.",
            "audience_constraint": "Explain blockchain technology to a 10-year-old using only simple words and analogies they would understand."
        }
        return examples
    
    def iterative_prompting(self):
        """Demonstrate iterative prompt refinement"""
        iterations = {
            "iteration_1": "Write a story about a robot.",
            "iteration_2": "Write a 300-word science fiction story about a robot who discovers emotions.",
            "iteration_3": "Write a 300-word science fiction story about a household cleaning robot who discovers emotions when it accidentally damages a family heirloom. Focus on the robot's internal conflict and the family's reaction.",
            "iteration_4": "Write a 300-word science fiction story about a household cleaning robot named ARIA who discovers emotions when it accidentally damages a precious family heirloom - a grandmother's antique vase. Show ARIA's internal conflict between its programming and newfound feelings, and explore how the family's reaction teaches both the robot and readers about forgiveness and growth. Use vivid descriptions and dialogue."
        }
        return iterations
    
    def prompt_templates(self):
        """Provide reusable prompt templates"""
        templates = {
            "analysis_template": """
Analyze the following [TOPIC] and provide insights on:

1. Key characteristics
2. Advantages and disadvantages
3. Real-world applications
4. Future implications

[TOPIC]: {topic}

Please be specific and provide examples where relevant.
""",
            
            "comparison_template": """
Compare and contrast [ITEM_A] and [ITEM_B] across the following dimensions:

- Functionality
- Cost
- Ease of use
- Performance
- Best use cases

[ITEM_A]: {item_a}
[ITEM_B]: {item_b}

Conclude with a recommendation for different user types.
""",
            
            "problem_solving_template": """
I'm facing the following challenge: {problem}

Please help me by:
1. Breaking down the problem into smaller components
2. Suggesting 3 potential solutions
3. Evaluating the pros and cons of each solution
4. Recommending the best approach with reasoning

Consider constraints: {constraints}
"""
        }
        return templates

def demonstrate_prompt_engineering():
    """Run demonstrations of various prompt engineering techniques"""
    demo = PromptEngineeringDemo()
    
    print("=== PROMPT ENGINEERING TECHNIQUES DEMONSTRATION ===\n")
    
    print("1. BASIC PROMPTING")
    print("-" * 50)
    basic_examples = demo.basic_prompting()
    print("❌ Poor Prompt:")
    print(f'"{basic_examples["poor_prompt"]}"')
    print("\n✅ Good Prompt:")
    print(f'"{basic_examples["good_prompt"]}"')
    print("\nKey improvements: Specific length, target audience, clear focus\n")
    
    print("2. FEW-SHOT PROMPTING")
    print("-" * 50)
    few_shot = demo.few_shot_prompting()
    print(few_shot)
    print("Expected output: Positive\n")
    
    print("3. CHAIN-OF-THOUGHT PROMPTING")
    print("-" * 50)
    cot = demo.chain_of_thought_prompting()
    print(cot[:500] + "...")
    print("This technique helps the model reason step-by-step\n")
    
    print("4. ROLE-BASED PROMPTING")
    print("-" * 50)
    roles = demo.role_based_prompting()
    for role, prompt in roles.items():
        print(f"🎭 {role.replace('_', ' ').title()}:")
        print(f'"{prompt}"\n')
    
    print("5. STRUCTURED PROMPTING")
    print("-" * 50)
    structured = demo.structured_prompting()
    print(structured[:300] + "...")
    print("This ensures consistent, organized output format\n")
    
    print("6. CONSTRAINT PROMPTING")
    print("-" * 50)
    constraints = demo.constraint_prompting()
    for constraint_type, prompt in constraints.items():
        print(f"📏 {constraint_type.replace('_', ' ').title()}:")
        print(f'"{prompt}"\n')
    
    print("7. ITERATIVE PROMPTING")
    print("-" * 50)
    iterations = demo.iterative_prompting()
    for i, (iteration, prompt) in enumerate(iterations.items(), 1):
        print(f"Version {i}: {prompt}")
        print()
    print("Notice how each iteration adds more specificity and guidance\n")
    
    print("8. REUSABLE TEMPLATES")
    print("-" * 50)
    templates = demo.prompt_templates()
    for template_name, template in templates.items():
        print(f"📋 {template_name.replace('_', ' ').title()}:")
        print(template)
        print()

def prompt_engineering_best_practices():
    """Display best practices for prompt engineering"""
    practices = {
        "Be Specific": "Provide clear, detailed instructions rather than vague requests",
        "Use Examples": "Include examples of desired input/output format (few-shot learning)",
        "Set Context": "Establish the role, audience, and purpose clearly",
        "Break Down Complex Tasks": "Divide complex requests into smaller, manageable steps",
        "Specify Output Format": "Clearly define how you want the response structured",
        "Iterate and Refine": "Test prompts and improve them based on results",
        "Use Constraints": "Set boundaries for length, style, or content when needed",
        "Test Edge Cases": "Try your prompts with different inputs to ensure robustness"
    }
    
    print("=== PROMPT ENGINEERING BEST PRACTICES ===\n")
    for practice, description in practices.items():
        print(f"✅ {practice}: {description}")
    print()

if __name__ == "__main__":
    demonstrate_prompt_engineering()
    prompt_engineering_best_practices()
    
    print("=== INTERACTIVE PROMPT BUILDER ===")
    print("Try building your own prompt using these techniques!")
    print("Consider: What's your goal? Who's your audience? What format do you want?")

