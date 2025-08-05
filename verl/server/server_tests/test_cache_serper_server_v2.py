#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
import argparse
import requests

"""
Test Serper v2 server
Usage:
python test_cache_serper_server_v2.py <endpoint_url>
e.g. python test_cache_serper_server_v2.py http://127.0.0.1:9001/search
"""

# Serper API Key
API_KEY = os.environ.get("WEB_SEARCH_SERPER_API_KEY")

def test_serper_proxy(endpoint_url: str, query: str, num: int = 10):
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": API_KEY
    }
    payload = {
        "q": query,
        "num": num
    }

    print(f"--- Sending request for query: '{query}' ---")
    start_time = time.time()
    url = endpoint_url
    print(url, headers, payload)
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        elapsed_time = time.time() - start_time
        print(f"Status Code: {response.status_code} (Response time: {elapsed_time:.2f}s)")
        
        result = response.json()
        organic_results = result.get("organic", [])
        print(f"Found {len(organic_results)} organic results")
        
        if organic_results:
            print("\nTop results:")
            for i, item in enumerate(organic_results[:2]):
                print(f"{i+1}. {item.get('title', 'No title')}")
                print(f"   {item.get('link', 'No link')}")
        
        print("\nFull Response JSON:")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:1000] + "...")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Serper proxy test script.')
    parser.add_argument('endpoint_url', type=str, help='The endpoint URL to test against (e.g., http://127.0.0.1:9001/search)')
    args = parser.parse_args()

    print("=== TEST 1: First query (should miss cache) ===")
    test_serper_proxy(args.endpoint_url, "Apple Inc.", num=5)
    
    print("\n" + "="*50 + "\n")
    
    print("=== TEST 2: Same query again (should hit cache) ===")
    test_serper_proxy(args.endpoint_url, "Apple Inc.", num=5)
    
    print("\n" + "="*50 + "\n")
    
    print("=== TEST 3: Different query1 ===")
    test_serper_proxy(args.endpoint_url, "NVIDIA Corporation3", num=5)