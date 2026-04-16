from duckduckgo_search import DDGS

def test_search():
    print("Testing DDGS text search...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text("who is the president of usa", max_results=3))
            print(f"Results: {results}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_search()
