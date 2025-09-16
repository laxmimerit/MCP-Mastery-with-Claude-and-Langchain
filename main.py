import os
from dotenv import load_dotenv
load_dotenv()

def main():
    # print environment variables to verify they are loaded
    print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    print("CLAUDE_API_KEY:", os.getenv("CLAUDE_API_KEY"))
    print("LANGSMITH_API_KEY:", os.getenv("LANGSMITH_API_KEY"))
    
    print("Hello from mcp-mastery-claude-and-langchain!")


if __name__ == "__main__":
    main()
