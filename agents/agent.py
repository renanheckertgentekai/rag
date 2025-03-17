from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

agent = Agent(  
    'openai:gpt-4o',
    deps_type=set,
    result_type=str,
    system_prompt="""
Answer the user question based on the list of chunks data that will receive.
If the list is empty, answer: 'Could not answer the question'
"""
)

@agent.tool
async def get_chunks(ctx: RunContext[set], search_query: str) -> list[str]:  
    """Add chunks to answer"""
    chunks = list(ctx.deps)
    print(f'\nReceived a total of {len(chunks)} chunks to answer.')
    for chunk in chunks:
        print(chunk)

    return chunks