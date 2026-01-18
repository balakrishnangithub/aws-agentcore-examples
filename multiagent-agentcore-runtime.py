import time
from enum import Enum
from typing import Dict, Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel
from strands_tools import http_request

model = BedrockModel(model_id="anthropic.claude-3-5-haiku-20241022-v1:0")

app = BedrockAgentCoreApp()


class AgentRole(Enum):
    PLANNER = 'planner'
    RETRIEVER = 'retriever'
    ANALYST = 'analyst'
    VALIDATOR = 'validator'


class MultiAgentSystem:
    def __init__(self):
        self.agents = {
            AgentRole.PLANNER: Agent(
                model=model,
                system_prompt='Break complex problems into 2-3 actionable subtasks. '
                              'Provide clear, structured response.'
            ),
            AgentRole.RETRIEVER: Agent(
                model=model,
                system_prompt='Use free public APIs that require no authentication. '
                              'Make HTTP requests to retrieve the data. '
                              'Return only the facts from the API response. Be direct, no disclaimers.',
                tools=[http_request],
            ),
            AgentRole.ANALYST: Agent(
                model=model,
                system_prompt='Analyze the data provided in the context. '
                              'Trust the retrieved data and provide a clear, direct answer.'
            ),
            AgentRole.VALIDATOR: Agent(
                model=model,
                system_prompt='Provide the final answer based on the context. '
                              'Be concise and direct. No disclaimers needed.'
            )
        }
        self.execution_trace = []

    def execute_agent(self, role: AgentRole, prompt: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            agent = self.agents[role]
            response = agent(prompt)
            # Simple response extraction
            response_text = str(response)
            if hasattr(response, 'message'):
                response_text = response.message
            output = {'status': 'success', 'content': response_text}
            confidence = 0.9
        except Exception as e:
            output = {'status': 'error', 'error': str(e)}
            confidence = 0.2
        execution_time = time.time() - start_time
        return {
            'role': role.value,
            'output': output,
            'confidence': confidence,
            'execution_time': execution_time,
        }

    def process_query(self, user_query: str) -> Dict[str, Any]:
        pipeline_start = time.time()
        self.execution_trace = []
        context = ""

        # Execute agents in sequence, passing context forward
        agents_sequence = [
            (AgentRole.PLANNER, f'Break down this request: {user_query}'),
            (AgentRole.RETRIEVER, f'Fetch the data needed for: {user_query}'),
            (AgentRole.ANALYST, f'Analyze and answer: {user_query}'),
            (AgentRole.VALIDATOR, f'Verify and provide final answer for: {user_query}'),
        ]

        for role, base_prompt in agents_sequence:
            prompt = f'{base_prompt}\n\nContext from previous steps:\n{context}' if context else base_prompt
            result = self.execute_agent(role, prompt)
            self.execution_trace.append(result)
            # Pass output to next agent
            if result['output'].get('status') == 'success':
                context += f"\n[{role.value}]: {result['output'].get('content', '')}\n"

        # Calculate metrics
        total_time = time.time() - pipeline_start
        avg_confidence = sum(step['confidence'] for step in self.execution_trace) / len(self.execution_trace)

        # Build response
        return {
            'query': user_query,
            'status': 'success',
            'execution_trace': [
                {
                    'step': idx + 1,
                    'agent': step['role'],
                    'confidence': f'{step["confidence"]:.1f}',
                    'execution_time': f'{step["execution_time"]:.2f}',
                    'output': step['output']
                }
                for idx, step in enumerate(self.execution_trace)
            ],
            'summary': {
                'total_execution_time': f'{total_time:.2f}s',
                'average_confidence': f'{avg_confidence:.1%}',
                'agents_executed': len(self.execution_trace),
            },
            'results': {
                'confidence_level': 'high' if avg_confidence >= 0.8 else 'medium',
            }
        }


multi_agent_system = MultiAgentSystem()

@app.entrypoint
def invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_query = payload.get('prompt', 'Hello!')
    print(f'\n Processing query: {user_query}')
    print('=' * 60)
    try:
        result = multi_agent_system.process_query(user_query)
        print(f'Status: {result["status"]}')
        print(f'Time: {result["summary"]["total_execution_time"]}')
        print(f'Confidence: {result["summary"]["average_confidence"]}')
        return result
    except Exception as e:
        print(f'Error: {e}')
        return {
            'query': user_query,
            'status': 'error',
            'error_message': str(e),
        }


if __name__ == '__main__':
    print('Starting Multi Agent')
    app.run(host='0.0.0.0', port=8080)
