from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from langchain_community.vectorstores import FAISS

@dataclass
class MyDeps:  
    # score: float
    vector_db: FAISS
    max_chunks: int = 3


agent = Agent(  
    'openai:gpt-4o',
    deps_type=MyDeps,
    result_type=str,
    system_prompt="""
Answer the user question based on the list of chunks data that will receive.
If the list is empty, answer: 'Could not answer the question'
"""
)

@agent.tool
async def retrieve(ctx: RunContext[MyDeps], search_query: str) -> list[str]:  
    """Add text context"""
    print(f'Query used to search: {search_query}')

    chunks_list = ctx.deps.vector_db.similarity_search_with_score(
        search_query, ctx.deps.max_chunks
    )

    for item in chunks_list:
        chunk, score = item
        print(f'Chunk: {chunk.id}; Score: {score}')

    return [chunk[0].page_content for chunk in chunks_list]