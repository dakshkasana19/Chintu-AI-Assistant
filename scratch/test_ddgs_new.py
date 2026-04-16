from ddgs import DDGS

def test_search():
    print("Testing DDGS (new) text search...")
    try:
        results = list(DDGS().text("who is the president of usa", max_results=3))
        print(f"Results: {len(results)}")
        for r in results:
            print(f" - {r.get('title')}: {r.get('body')[:50]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_search()
