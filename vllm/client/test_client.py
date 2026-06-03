#!/usr/bin/env python3
import sys

def main():
    # 1. Check for dependency
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: The 'openai' package is not installed.")
        print("Please install it using: pip install openai")
        sys.exit(1)

    print("Initializing OpenAI client targeting local vLLM server...")
    
    # 2. Configure the local client
    # vLLM exposes an OpenAI-compatible API on the served port (default: 8000)
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="token-not-needed"  # Authentication is open by default for local dev
    )

    prompt = "Say hi in 3 words"
    model_name = "google/gemma-4-e2b-it"

    print(f"\nSending streaming request to model: '{model_name}'")
    print(f"Prompt: \"{prompt}\"\n")
    print("-" * 60)

    # 3. Request completion with streaming
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a friend."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()

    except Exception as e:
        print("\nConnection Error!")
        print("Could not connect to the local vLLM API server.")
        print("Make sure your server is running by executing:")
        print("  ../server/start_server.sh")
        print(f"\nDetails: {e}")
        sys.exit(1)

    print("-" * 60)
    print("\nInference test completed successfully!")

if __name__ == "__main__":
    main()
