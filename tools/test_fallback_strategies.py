#!/usr/bin/env python3
"""
Test the improved email enrichment with fallback strategies including website scraping
"""

import requests
import json
import re
from urllib.parse import urlparse

def extract_emails_from_text(text):
    """Extract email addresses from text using regex"""
    if not text:
        return []
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, str(text))
    return list(set(emails))  # Remove duplicates

def try_sync_website_scrape(website, business_name):
    """Try to scrape website for email addresses synchronously"""
    try:
        url = website if website.startswith('http') else f'https://{website}'
        print(f"   Attempting to scrape: {url}")
        
        response = requests.get(
            url,
            timeout=5,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        if response.status_code == 200:
            # Look for email patterns in HTML
            emails = extract_emails_from_text(response.text)
            if emails:
                # Filter out common auto-generated emails
                filtered = [e for e in emails if not any(x in e.lower() for x in ['noreply', 'no-reply', 'postmaster'])]
                if filtered:
                    return filtered[0]
    except requests.exceptions.Timeout:
        print(f"   Timeout while scraping")
    except Exception as e:
        print(f"   Error while scraping: {str(e)[:100]}")
    
    return None

def generate_common_email_patterns(domain, business_name):
    """Generate common email patterns for a domain"""
    # Remove www. if present for cleaner domain
    domain_clean = domain.replace('www.', '')
    
    # Common prefixes to try
    common_prefixes = [
        'info', 'contact', 'support', 'hello', 'business', 'sales', 
        'email', 'inquiry', 'admin', 'help', 'team'
    ]
    
    # Try with business name components
    name_parts = business_name.lower().split()[:2] if business_name else []
    
    patterns = []
    
    # Standard patterns
    for prefix in common_prefixes:
        patterns.append(f"{prefix}@{domain_clean}")
    
    # Business name patterns
    if name_parts:
        for part in name_parts:
            part_clean = ''.join(c for c in part if c.isalnum())
            if part_clean:
                patterns.append(f"{part_clean}@{domain_clean}")
    
    return patterns

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

print("=" * 80)
print("TESTING IMPROVED EMAIL ENRICHMENT WITH FALLBACK STRATEGIES")
print("=" * 80)

for business in test_businesses:
    name = business['name']
    website = business['website']
    domain = website.replace('https://', '').replace('http://', '').split('/')[0]
    
    print(f"\n📍 {name}")
    print(f"   Website: {website}")
    print(f"   Domain: {domain}")
    print("-" * 80)
    
    email = 'N/A'
    
    # Try direct website scraping
    print(f"   Trying direct website scrape...")
    scraped_email = try_sync_website_scrape(website, name)
    if scraped_email:
        email = scraped_email
        print(f"   ✅ Found via website scrape: {email}")
    else:
        print(f"   ℹ️  No email found via scraping")
    
    # If still no email, generate common patterns
    if email == 'N/A':
        print(f"   Generating common email patterns...")
        patterns = generate_common_email_patterns(domain, name)
        if patterns:
            email = patterns[0]
            print(f"   Suggested patterns:")
            for pattern in patterns[:5]:
                print(f"      - {pattern}")
    
    print(f"   Final result: {email}")
    print("-" * 80)

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
