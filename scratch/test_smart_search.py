import re
import requests
from ddgs import DDGS

def clean_text(text):
    if not text:
        return ""
    # Remove dates like "Jul 30, 2024 — ", "2 days ago — ", etc.
    text = re.sub(r'^(?:[A-Za-z]{3}\s+\d{1,2},\s+\d{4}|\d+\s+(?:days?|hours?|weeks?|months?|years?)\s+ago)\s*[-—–·•]\s*', '', text, flags=re.IGNORECASE)
    # Remove weird unicode artifacts
    text = text.replace('\xa0', ' ').replace('\u200b', ' ').replace('', '')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def search_wikipedia(query):
    try:
        # Search wikipedia api for best title match
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={requests.utils.quote(query)}&limit=1&namespace=0&format=json"
        res = requests.get(search_url, headers={'User-Agent': 'ChintuAI/2.0'}, timeout=3).json()
        if res and len(res) > 1 and res[1]:
            title = res[1][0]
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
            s_res = requests.get(summary_url, headers={'User-Agent': 'ChintuAI/2.0'}, timeout=3).json()
            extract = s_res.get('extract')
            if extract:
                # Take first 1-2 sentences
                sentences = re.split(r'(?<=[.!?])\s+', extract)
                return " ".join(sentences[:2])
    except Exception as e:
        pass
    return None

def search_ddg(query):
    try:
        results = list(DDGS().text(query, max_results=5))
        return results
    except Exception as e:
        print("DDG error:", e)
        return []

def summarize_results(query, ddg_results):
    if not ddg_results:
        return "I couldn't find any relevant results on the web right now."
    
    candidates = []
    query_words = set(re.findall(r'\w+', query.lower())) - {'what', 'is', 'the', 'who', 'how', 'when', 'where', 'are', 'in', 'of', 'and', 'a', 'to', 'for', 'was', 'tell', 'me', 'about'}
    
    for r in ddg_results:
        body = clean_text(r.get('body', ''))
        sentences = re.split(r'(?<=[.!?])\s+', body)
        for s in sentences:
            s = s.strip()
            if len(s) < 20 or len(s) > 220:
                continue
            # Score sentence by overlap with query tokens
            s_words = set(re.findall(r'\w+', s.lower()))
            overlap = len(query_words.intersection(s_words))
            score = overlap * 3
            if any(char.isdigit() for char in s):
                score += 2
            if any(s.lower().startswith(w) for w in ['it is', 'he is', 'she is', 'they are', 'this is']):
                score += 2
            # Filter out search junk
            if any(junk in s.lower() for junk in ['cookie', 'subscribe', 'sign in', 'click here', 'privacy policy']):
                score -= 10
            candidates.append((score, s))
            
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        # Return best 1-2 sentences
        return best
    return clean_text(ddg_results[0].get('body', ''))[:150]

def answer_query(query):
    # Try wikipedia first for who/what/where/when encyclopedic queries
    lower_q = query.lower()
    wiki_triggers = ['who is', 'what is', 'where is', 'when was', 'tell me about', 'define', 'meaning of', 'who was']
    if any(lower_q.startswith(t) for t in wiki_triggers):
        # Extract subject
        subject = lower_q
        for t in wiki_triggers:
            if subject.startswith(t):
                subject = subject[len(t):].strip(' ?.')
                break
        if subject:
            wiki_ans = search_wikipedia(subject)
            if wiki_ans:
                return wiki_ans
                
    # Fallback to DDG search & smart summarization
    ddg_res = search_ddg(query)
    return summarize_results(query, ddg_res)

test_queries = [
    "who is the prime minister of the united kingdom",
    "what is quantum computing",
    "when was the eiffel tower built",
    "who won the 2022 fifa world cup"
]

for q in test_queries:
    print(f"\nQ: {q}")
    ans = answer_query(q)
    print(f"A: {ans}")
