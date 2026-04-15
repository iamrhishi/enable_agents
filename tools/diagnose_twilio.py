#!/usr/bin/env python3
"""
Twilio WhatsApp Channel Diagnostic Script
Helps identify the correct WhatsApp number configured in your Twilio account
"""

import os
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables
load_dotenv()

ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
CONFIGURED_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER')

print("=" * 80)
print("TWILIO WHATSAPP CHANNEL DIAGNOSTIC")
print("=" * 80)

if not ACCOUNT_SID or not AUTH_TOKEN:
    print("❌ ERROR: TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN not found in .env")
    exit(1)

print(f"\n📝 Current Configuration:")
print(f"   TWILIO_ACCOUNT_SID: {ACCOUNT_SID}")
print(f"   TWILIO_WHATSAPP_NUMBER: {CONFIGURED_WHATSAPP_NUMBER}")

try:
    # Initialize Twilio client
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    
    print("\n🔍 Checking available WhatsApp Senders in your Twilio account...")
    print("-" * 80)
    
    # Get account details
    account = client.api.accounts(ACCOUNT_SID).fetch()
    print(f"\n✓ Connected to Twilio Account: {account.friendly_name}")
    
    # Try to get phone numbers
    print("\n📱 Checking Incoming Phone Numbers:")
    try:
        incoming_numbers = client.incoming_phone_numbers.stream(limit=20)
        whatsapp_numbers = []
        
        for number in incoming_numbers:
            print(f"   - {number.phone_number} ({number.friendly_name})")
            whatsapp_numbers.append(number.phone_number)
        
        if whatsapp_numbers:
            print(f"\n✓ Found {len(whatsapp_numbers)} phone number(s)")
        else:
            print("\n⚠️  No incoming phone numbers found")
            
    except Exception as e:
        print(f"   Error fetching numbers: {e}")
    
    # Check if we're using Twilio Sandbox
    print("\n🎮 Twilio Sandbox Information:")
    print("   If you're using the Twilio WhatsApp Sandbox (free testing):")
    print("   - Use: +14155238886 as your TWILIO_WHATSAPP_NUMBER")
    print("   - Make sure to join the sandbox first via WhatsApp")
    print("   - Sandbox number: WhatsApp to +1 415-523-8886 with message: 'join'")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if CONFIGURED_WHATSAPP_NUMBER == "+919850742880":
        print("\n⚠️  Current number (+919850742880) is not recognized as Twilio Sandbox")
        print("\nOptions:")
        print("1. If using FREE Twilio Sandbox:")
        print("   - Set TWILIO_WHATSAPP_NUMBER=+14155238886 in .env")
        print("   - Make sure you've joined the sandbox via WhatsApp")
        print("\n2. If using production WhatsApp Business Account:")
        print("   - Verify the number is connected to your Business Account")
        print("   - Check Twilio Console > Messaging > Services > WhatsApp")
        print("   - Ensure the number is verified and active")
    
    print("\n" + "=" * 80)
    
except Exception as e:
    print(f"\n❌ Error connecting to Twilio: {e}")
    print("\nTroubleshooting:")
    print("1. Verify TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are correct")
    print("2. Check that your Twilio account is active")
    print("3. Make sure your auth token hasn't expired")
