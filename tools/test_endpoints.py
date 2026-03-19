#!/usr/bin/env python3
"""
Test different scrap.io endpoints for email extraction
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

scrap_io_api_key = os.getenv('SCRAP_IO_API_KEY')
if not scrap_io_api_key:
    print("❌ ERROR: SCRAP_IO_API_KEY not found in .env file")
    exit(1)

headers = {
    'Authorization': f'Bearer {scrap_io_api_key}',
    'Content-Type': 'application/json'
}

test_website = "www.americanintegratedsupply.com"

print("=" * 80)
print("TESTING DIFFERENT SCRAP.IO ENDPOINTS")
print("=" * 80)

endpoints_to_test = [
    ("https://scrap.io/api/v1/gmap/enrich", {'domain': 'americanintegratedsupply.com'}, "gmap/enrich with domain"),
    ("https://scrap.io/api/v1/gmap/enrich", {'url': test_website}, "gmap/enrich with url"),
    ("https://scrap.io/api/v1/gmap/enrich", {'url': 'https://' + test_website}, "gmap/enrich with https://url"),
    # Try direct website scraping
    ("https://scrap.io/api/v1/scrape", {'url': test_website}, "scrape endpoint with url"),
    ("https://scrap.io/api/v1/scrape", {'url': 'https://' + test_website}, "scrape with https://url"),
]

for endpoint, params, description in endpoints_to_test:
    print(f"\n🔵 Testing: {description}")
    print(f"   Endpoint: {endpoint}")
    print(f"   Params: {params}")
    print("-" * 80)
    
    try:
        response = requests.get(
            endpoint,
            params=params,
            headers=headers,
            timeout=15
        )
        print(f"   Status: {response.status_code}")
        
        result = response.json()
        
        # Show meta if exists
        if 'meta' in result:
            meta = result.get('meta', {})
            print(f"   META: status={meta.get('status')}, count={meta.get('count')}")
        
        # Show any data
        if 'data' in result:
            data = result.get('data', [])
            print(f"   DATA array length: {len(data)}")
            if data and len(data) > 0:
                print(f"   ✅ DATA FOUND!")
                print(f"   Keys: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
                print(json.dumps(data[0], indent=2)[:1000])
        else:
            print(f"   Response keys: {list(result.keys())}")
        
        # Show first 500 chars of response
        print(f"\n   Response preview:")
        print(json.dumps(result, indent=2)[:500])
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("-" * 80)

print("\n" + "=" * 80)
print("ENDPOINT TEST COMPLETE")
print("=" * 80)
