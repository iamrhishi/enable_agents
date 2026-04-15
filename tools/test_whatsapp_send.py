#!/usr/bin/env python3
"""
Quick WhatsApp Sandbox Test Script
Tests if the Twilio WhatsApp Sandbox is properly configured
"""

import os
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables
load_dotenv()

ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER')

print("=" * 80)
print("TWILIO WHATSAPP SANDBOX TEST")
print("=" * 80)

print(f"\n✓ Config loaded:")
print(f"  - Account SID: {ACCOUNT_SID[:10]}...")
print(f"  - WhatsApp From Number: {TWILIO_WHATSAPP_NUMBER}")

# Test number - CHANGE THIS to your actual phone number
TEST_TO_NUMBER = input("\n📱 Enter YOUR phone number to test (format: +1234567890): ").strip()

if not TEST_TO_NUMBER.startswith('+'):
    TEST_TO_NUMBER = '+' + TEST_TO_NUMBER

print(f"\n🔄 Testing WhatsApp message send...")
print(f"  From: whatsapp:{TWILIO_WHATSAPP_NUMBER}")
print(f"  To: whatsapp:{TEST_TO_NUMBER}")

try:
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    
    message = client.messages.create(
        from_=f'whatsapp:{TWILIO_WHATSAPP_NUMBER}',
        to=f'whatsapp:{TEST_TO_NUMBER}',
        body='✅ Test message from Twilio WhatsApp Sandbox - If you received this, sandbox is working!'
    )
    
    print(f"\n✅ SUCCESS!")
    print(f"   Message SID: {message.sid}")
    print(f"   Status: {message.status}")
    print(f"\n🎉 Your WhatsApp Sandbox is configured correctly!")
    print(f"   You should receive the message on WhatsApp shortly.")
    
except Exception as e:
    error_msg = str(e)
    print(f"\n❌ FAILED")
    print(f"   Error: {error_msg}")
    
    print(f"\n🔧 TROUBLESHOOTING:")
    
    if "Unable to create record" in error_msg and "Channel" in error_msg:
        print("\n   ERROR 63007 - Twilio cannot find the WhatsApp channel")
        print("\n   ❌ LIKELY CAUSES:")
        print("      1. You haven't joined the Twilio WhatsApp Sandbox")
        print("      2. The sandbox invitation has expired (24 hours)")
        print("\n   ✅ TO FIX:")
        print("      1. Open WhatsApp on your phone")
        print("      2. Send a message to: +1 415-523-8886")
        print("      3. Type: join")
        print("      4. Wait for confirmation from Twilio")
        print("      5. Come back and run this test again")
        
    elif "Invalid" in error_msg and "phone" in error_msg:
        print("\n   Your phone number format is invalid")
        print("   Use format: +CountryCodeNumber (e.g., +14155551234)")
        
    else:
        print(f"\n   Check your Twilio credentials:")
        print(f"   - TWILIO_ACCOUNT_SID correct?")
        print(f"   - TWILIO_AUTH_TOKEN correct?")

print("\n" + "=" * 80)
