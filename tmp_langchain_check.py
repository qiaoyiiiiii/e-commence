import langchain
from langchain import agents
print('langchain', langchain.__version__)
print('has create_react_agent', hasattr(agents, 'create_react_agent'))
print('has initialize_agent', hasattr(agents, 'initialize_agent'))
print('has AgentExecutor', hasattr(agents, 'AgentExecutor'))
print('dir subset', [n for n in dir(agents) if 'Agent' in n or 'react' in n.lower()][:50])
