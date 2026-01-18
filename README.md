# AWS AgentCore Examples

Learning Amazon Bedrock AgentCore with this tutorial: https://www.youtube.com/watch?v=j2wYT6jqXZY

## Run

```bash
python first-agent-bedrock-agentcore-runtime.py
```

## Test

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2 + 2?"}'
```

## Deploy

```commandline
agentcore configure --entrypoint multiagent-agentcore-runtime.py
agentcore launch
```