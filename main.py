import sys
from vector_store import get_vector_store
from agents.agent import agent
from agents.query_agent import query_agent

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py '<your query>'")
        sys.exit(1)
    original_query = sys.argv[1]

    print(f'Query to answer: {original_query}\n')

    document = "minikube.pdf"
    max_chunks = 2
    total_queries = 4
    vector_store = get_vector_store(document)

    # Regenerate queries
    queries = query_agent.run_sync(original_query, deps=total_queries)

    # Get chunks from new queries
    chunks_list: list[str] = [] 
    for query in queries.data:
        seacrh_data = vector_store.similarity_search_with_score(query, max_chunks)
        print(f"Seacrching for query: '{query}'")
        for document, score in seacrh_data:
            if document.id:
                chunks_list.append(document.id)
            print(f'Chunk ID: {document.id}; Score: {score}')
    
    all_chunks = set(chunks_list)

    # Final seacrh
    result = agent.run_sync(f'Original query: {original_query}', deps=all_chunks)
    print('\nFinal Answer:')
    print(result.data)

if __name__ == "__main__":
    main()
