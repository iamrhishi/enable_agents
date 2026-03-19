#!/usr/bin/env python3
"""
Test the improved email enrichment with retry mechanism
"""

import requests
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

scrap_io_api_key = os.getenv('SCRAP_IO_API_KEY')
if not scrap_io_api_key:
    print("❌ ERROR: SCRAP_IO_API_KEY not found in .env file")
    exit(1)

endpoint = "https://scrap.io/api/v1/gmap/enrich"

test_businesses = [
    {
        "name": "American Integrated Supply",
        "website": "www.americanintegratedsupply.com"
    },
    {
        "name": "Acme Nut and Bolt",
        "website": "www.acmenutandbolt.com"
    },
    {
        "name": "Allstate Fastener",
        "website": "www.allstatefastener.com"
    }
]

def call_scrap_io_with_retry(domain, max_retries=3, delay=2):
    """Call scrap.io with retries to handle async processing"""
    
    headers = {
        'Authorization': f'Bearer {scrap_io_api_key}',
        'Content-Type': 'application/json'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                endpoint,
                params={'domain': domain},
                headers=headers,
                timeout=15
            )
            
            if response.status_code in [200, 202]:
                result = response.json()
                data = result.get('data', [])
                meta = result.get('meta', {})
                status = meta.get('status', 'unknown')
                
                # If we got data, return it
                if data and len(data) > 0:
                    print(f"   ✅ Got data on attempt {attempt + 1}")
                    return result, True
                
                # If status is incomplete, retry with delay
                if status == 'incomplete' and attempt < max_retries - 1:
                    print(f"   ⏳ Status incomplete, waiting {delay}s before retry... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                
                # Status is completed with no data, or this is the last attempt
                print(f"   ℹ️  Status: {status}, no data retrieved (attempt {attempt + 1}/{max_retries})")
                return result, False
            else:
                print(f"   ❌ API Error {response.status_code}")
                return {}, False
                
        except requests.exceptions.Timeout:
            print(f"   ⏱️  Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            return {}, False
        except Exception as e:
            print(f"   ❌ Error on attempt {attempt + 1}: {str(e)}")
            return {}, False
    
    return {}, False

print("=" * 80)
print("TESTING IMPROVED EMAIL ENRICHMENT WITH RETRIES")
print("=" * 80)

for business in test_businesses:
    name = business['name']
    website = business['website']
    domain = website.replace('https://', '').replace('http://', '').split('/')[0]
    
    print(f"\n📍 {name}")
    print(f"   Website: {website}")
    print(f"   Domain: {domain}")
    print("-" * 80)
    
    result, has_data = call_scrap_io_with_retry(domain, max_retries=3, delay=2)
    
    if has_data:
        print(f"   ✅ Success! Data received.")
        data = result.get('data', [])
        if data:
            business_data = data[0]
            print(f"   Keys in response: {list(business_data.keys())}")
            if 'website_data' in business_data:
                website_data = business_data['website_data']
                print(f"   Website data keys: {list(website_data.keys())}")
                if 'emails' in website_data:
                    print(f"   Emails found: {website_data['emails']}")
    else:
        print(f"   ℹ️  No data returned from scrap.io")
    
    print("-" * 80)

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
