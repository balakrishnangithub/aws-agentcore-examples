from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel
from strands_tools import http_request

app = BedrockAgentCoreApp()

SYSTEM_PROMPT = """You are a helpful assistant with access to the internet via HTTP requests.

When the user asks for real-time information, use free public APIs that require no authentication or API keys.

Guidelines:
- Search your knowledge for appropriate free, public APIs to answer the user's question
- Make HTTP requests to retrieve the data
- Parse the response and present it in a clear, human-readable format
- If a request fails, try an alternative API or explain the issue

Only use free, publicly accessible APIs that don't require credentials."""


def get_agent_response(prompt):
    model = BedrockModel(model_id="anthropic.claude-3-5-haiku-20241022-v1:0")
    agent = Agent(model=model, tools=[http_request], system_prompt=SYSTEM_PROMPT)
    return agent(prompt)


@app.entrypoint
def prompt_handler(payload):
    user_prompt = payload.get('prompt')
    response = get_agent_response(user_prompt)
    return response.message


if __name__ == '__main__':
    print('Starting Bedrock Agent Core Runtime')
    app.run(host='0.0.0.0', port=8080)
