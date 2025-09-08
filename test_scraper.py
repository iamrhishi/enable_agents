#!/usr/bin/env python3
"""
Test script for the scraping API endpoints
"""

import requests
import json

# Base URL for your Flask app
BASE_URL = "http://localhost:5000"

def test_basic_scraping():
    """Test basic scraping functionality"""
    print("Testing basic scraping...")
    
    url = "https://httpbin.org/html"  # A simple test page
    
    payload = {
        "url": url,
        "type": "basic",
        "config": {}
    }
    
    try:
        response = requests.post(f"{BASE_URL}/scrape", json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_text_scraping():
    """Test text content scraping"""
    print("\nTesting text scraping...")
    
    url = "https://httpbin.org/html"
    
    payload = {
        "url": url,
        "type": "text",
        "config": {
            "clean_text": True,
            "max_length": 500
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/scrape", json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_product_scraping():
    """Test product scraping functionality"""
    print("\nTesting product scraping...")
    
    url = "https://httpbin.org/html"  # Simple test page
    
    payload = {
        "url": url,
        "interactive": False,  # Use non-interactive mode for testing
        "headless": True,
        "config": {}
    }
    
    try:
        response = requests.post(f"{BASE_URL}/scrape_product_info", json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_invalid_url():
    """Test handling of invalid URLs"""
    print("\nTesting invalid URL handling...")
    
    payload = {
        "url": "not-a-valid-url",
        "type": "basic",
        "config": {}
    }
    
    try:
        response = requests.post(f"{BASE_URL}/scrape", json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 400  # Should return 400 for invalid URL
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting scraper API tests...")
    print("=" * 50)
    
    tests = [
        test_basic_scraping,
        test_text_scraping,
        test_product_scraping,
        test_invalid_url
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        print("-" * 30)
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")

if __name__ == "__main__":
    main()
