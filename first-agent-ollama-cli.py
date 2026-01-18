from strands import Agent
from strands.models import OllamaModel
from strands_tools import http_request

SYSTEM_PROMPT = """You are a helpful assistant with access to the internet via HTTP requests.

When the user asks for real-time information, use free public APIs that require no authentication or API keys.

Guidelines:
- Search your knowledge for appropriate free, public APIs to answer the user's question
- Make HTTP requests to retrieve the data
- Parse the response and present it in a clear, human-readable format
- If a request fails, try an alternative API or explain the issue

Only use free, publicly accessible APIs that don't require credentials."""


def get_agent_response(prompt):
    model = OllamaModel(host="http://localhost:11434", model_id="ministral-3:3b")
    agent = Agent(model=model, tools=[http_request], system_prompt=SYSTEM_PROMPT)
    return agent(prompt)


def main():
    user_prompt = input('User: ')
    response = get_agent_response(user_prompt)
    print(f'Agent: {response.message}')


if __name__ == '__main__':
    main()
