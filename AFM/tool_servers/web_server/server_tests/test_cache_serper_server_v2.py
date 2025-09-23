#!/usr/bin/env python
# coding=utf-8

"""
Test Serper Cache Server v2
Usage:
python test_cache_serper_server_v2.py <endpoint_url>
e.g. python test_cache_serper_server_v2.py http://127.0.0.1:9001/search

The request/response format follows verl/verl/tools/web_search_v2.py:
  - POST JSON: {"q": <query>, "num": <int>}
  - Response JSON contains field "organic" (list of results)
"""

import argparse
import json
import os
import sys
import time

import requests


def run_once(endpoint_url: str, query: str, num: int) -> bool:
    print("=" * 20)
    print(f"Testing query: '{query}' with num={num}")
    print("=" * 20)

    payload = {"q": query, "num": num}
    headers = {"Content-Type": "application/json"}

    # Optional: pass through X-API-KEY if user set it (as in web_search_v2.py)
    api_key = os.environ.get("WEB_SEARCH_SERPER_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key

    try:
        start = time.time()
        resp = requests.post(endpoint_url, json=payload, timeout=2500, headers=headers)
        elapsed = time.time() - start
        resp.raise_for_status()

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Response content: {resp.text}")
            return False

        organic = data.get("organic", [])
        count = len(organic)
        print(f"Success! HTTP {resp.status_code}. Time: {elapsed:.2f}s. Results: {count}")

        # Print top-k brief results
        top_k = min(count, 5)
        for i in range(top_k):
            item = organic[i]
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            print(f"{i+1}. {title}\n   {link}\n   {snippet[:200]}")

        # Basic expectation: count should be <= requested num
        if count > num:
            print(f"Warning: result count ({count}) exceeds requested num ({num}).")

        return True

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server!")
        return False
    except requests.exceptions.Timeout:
        print("Error: Request Timeout")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
        return False
    except Exception as e:
        print(f"Unknown Error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Serper cache v2 test script.')
    parser.add_argument('endpoint_url', type=str, help='Endpoint URL, e.g., http://127.0.0.1:9001/search')
    args = parser.parse_args()

    endpoint_url = args.endpoint_url
    print("Test start...")
    print(f"Endpoint: {endpoint_url}")

    # Two rounds: first small num, then larger num to exercise cache trimming/expansion logic
    ok1 = run_once(endpoint_url, query="Qwen is developed by?", num=5)
    ok2 = run_once(endpoint_url, query="Qwen is developed by?", num=10)

    # A different query to avoid cache hit
    ok3 = run_once(endpoint_url, query="What is OPPO AndesGPT?", num=8)

    all_ok = ok1 and ok2 and ok3
    print("\nAll test done.")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()


