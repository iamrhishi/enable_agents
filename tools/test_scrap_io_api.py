#!/usr/bin/env python3
"""
Test script to diagnose scrap.io API responses for email extraction
Tests both domain and URL parameters to compare responses
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

# Test URLs/websites
test_websites = [
    "www.americanintegratedsupply.com",
    "www.acmenutandbolt.com",
    "www.allstatefastener.com",
    "https://www.americanintegratedsupply.com",
]

endpoint = "https://scrap.io/api/v1/gmap/enrich"

headers = {
    'Authorization': f'Bearer {scrap_io_api_key}',
    'Content-Type': 'application/json'
}

print("=" * 80)
print("SCRAP.IO API RESPONSE TESTING")
print("=" * 80)

print("=" * 80)
print("SCRAP.IO API RESPONSE TESTING")
print("=" * 80)

for website in test_websites:
    print(f"\n📍 Testing Website: {website}")
    print("-" * 80)
    
    # Extract domain for comparison
    domain = website.replace('https://', '').replace('http://', '').split('/')[0]
    if domain.startswith('www.'):
        domain_no_www = domain[4:]
    else:
        domain_no_www = domain
    
    # Test 1: URL parameter (current approach)
    print(f"\n🔵 Test 1: Using URL parameter = '{website}'")
    try:
        response = requests.get(
            endpoint,
            params={'url': website},
            headers=headers,
            timeout=15
        )
        print(f"   Status: {response.status_code}")
        result = response.json()
        
        # Show complete meta info
        meta = result.get('meta', {})
        print(f"   META: status={meta.get('status')}, count={meta.get('count')}")
        print(f"   META: next_cursor={meta.get('next_cursor')}, previous_cursor={meta.get('previous_cursor')}")
        print(f"   DATA array length: {len(result.get('data', []))}")
        
        # Show raw data if it exists
        if result.get('data'):
            for idx, data_item in enumerate(result['data']):
                print(f"\n   DATA[{idx}] Keys: {list(data_item.keys())}")
                print(f"   FULL DATA[{idx}]:")
                print(json.dumps(data_item, indent=2)[:1500])
        else:
            print(f"\n   FULL RESPONSE:")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 2: Domain parameter (old approach)
    print(f"\n🔴 Test 2: Using domain parameter = '{domain_no_www}'")
    try:
        response = requests.get(
            endpoint,
            params={'domain': domain_no_www},
            headers=headers,
            timeout=15
        )
        print(f"   Status: {response.status_code}")
        result = response.json()
        
        # Show complete meta info
        meta = result.get('meta', {})
        print(f"   META: status={meta.get('status')}, count={meta.get('count')}")
        print(f"   META: next_cursor={meta.get('next_cursor')}, previous_cursor={meta.get('previous_cursor')}")
        print(f"   DATA array length: {len(result.get('data', []))}")
        
        # Show raw data if it exists
        if result.get('data'):
            for idx, data_item in enumerate(result['data']):
                print(f"\n   DATA[{idx}] Keys: {list(data_item.keys())}")
                print(f"   FULL DATA[{idx}]:")
                print(json.dumps(data_item, indent=2)[:1500])
        else:
            print(f"\n   FULL RESPONSE:")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("-" * 80)

print("\n" + "=" * 80)
print("TESTING WITH POLLING/CURSORS")
print("=" * 80)

# Test with polling for async results
print(f"\n🟢 Test 3: Submit job and check cursor")
try:
    # First request to get cursor
    response = requests.get(
        endpoint,
        params={'url': 'https://www.americanintegratedsupply.com'},
        headers=headers,
        timeout=15
    )
    result = response.json()
    cursor = result.get('meta', {}).get('next_cursor')
    
    print(f"   Initial Status: {result.get('meta', {}).get('status')}")
    print(f"   Count: {result.get('meta', {}).get('count')}")
    print(f"   Next Cursor: {cursor}")
    
    # If we have a cursor, try to get results
    if cursor:
        print(f"\n   🔄 Polling with cursor: {cursor}")
        import time
        time.sleep(2)  # Wait before polling
        
        response2 = requests.get(
            endpoint,
            params={'url': 'https://www.americanintegratedsupply.com', 'cursor': cursor},
            headers=headers,
            timeout=15
        )
        result2 = response2.json()
        print(f"   Status after polling: {result2.get('meta', {}).get('status')}")
        print(f"   Count after polling: {result2.get('meta', {}).get('count')}")
        
        if result2.get('data'):
            print(f"   DATA FOUND! Keys: {list(result2['data'][0].keys())}")
            print(json.dumps(result2['data'][0], indent=2)[:1000])
        else:
            print(f"   Still no data in response")
except Exception as e:
    print(f"   ❌ Error: {str(e)}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
