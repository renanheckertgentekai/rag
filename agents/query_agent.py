from pydantic_ai import Agent, RunContext

query_agent = Agent(  
    'openai:gpt-4o',
    deps_type=int,
    result_type=list[str],
    system_prompt="""
You will receive a query to be embedded and used for similarity search. 
Currently, we are experiencing many problems with the poor quality of the retrieved chunks, 
so we need you to generate additional queries based on the one you receive. Its important
that the queries you will generates have a similar context of the original one, avoiding
subjects out of contexts.

Use the `max_queries` to get the size of the queries list you should create.

The first query in the list should be the original that you received
"""
)

@query_agent.tool
async def max_queries(ctx: RunContext[int]) -> str:  
    """Regenate queries"""
    return f'Create exactly {ctx.deps} additional queries'


if __name__ == '__main__':
    number_of_queries = 5
    query = 'What is minkube'
    result = query_agent.run_sync(f'Original query: {query}', deps=number_of_queries)
    print(result.data) 
