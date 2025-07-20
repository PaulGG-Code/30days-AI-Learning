def simple_ai_program(input_data):
    """
    A very simple AI program that makes a decision based on input.
    In a real AI, this logic would be much more complex, involving
    machine learning models, rules engines, or deep neural networks.
    """
    if "hot" in input_data.lower() and "sunny" in input_data.lower():
        return "It's a good day for ice cream!"
    elif "cold" in input_data.lower() or "rainy" in input_data.lower():
        return "Perhaps a hot cup of tea is in order."
    else:
        return "I'm not sure what to recommend. Tell me more!"

if __name__ == "__main__":
    print("--- Simple AI Program ---")
    print("Enter observations about the weather (e.g., 'It's hot and sunny'):")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            print("Exiting program.")
            break
        
        recommendation = simple_ai_program(user_input)
        print(f"AI says: {recommendation}")


