import sys
from vector_store import get_vector_store
from agent import agent, MyDeps
from query_agent import query_agent

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py '<your query>'")
        sys.exit(1)

    query = sys.argv[1]

    document = "minikube.pdf"

    vector_store = get_vector_store(document)

    deps = MyDeps(
        vector_db=vector_store,
        max_chunks=3
    )

    # Regenerate queries
    queries = query_agent.run_sync(query, deps=4)

    print(f'USER INPUT {query}')
    print(f'Query generated')
    for query in queries.data:
        print(query)
    
    for i, query in enumerate(queries.data):
        print(10 * '=' + 'START')
        result = agent.run_sync(f'Original query: {query}', deps=deps)
        if i == 0:
            print(f'Original Query: {query}')
        else:
            print(f'Generated Query {i}: {query}')
        print('------ > Result')
        print(result.data)
        print(10 * '=')
        print('\n')

if __name__ == "__main__":
    main()
