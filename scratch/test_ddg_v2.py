from duckduckgo_search import DDGS
import time

def test_search():
    queries = ["who is the president of usa", "chatgpt", "genshin impact launch date"]
    with DDGS() as ddgs:
        for q in queries:
            print(f"Searching for: {q}")
            try:
                # Try with different parameters
                results = list(ddgs.text(q, max_results=3))
                print(f"Results (default): {len(results)}")
                if not results:
                    # Try specifying region
                    results = list(ddgs.text(q, region='wt-wt', max_results=3))
                    print(f"Results (wt-wt): {len(results)}")
                
                for r in results:
                    print(f" - {r.get('title')}")
            except Exception as e:
                print(f"Error searching for {q}: {e}")
            time.sleep(1)

if __name__ == "__main__":
    test_search()
