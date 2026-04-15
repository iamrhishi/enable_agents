#!/usr/bin/env python3
"""
Discover all WhatsApp channels configured in your Twilio account
This will show us which phone number(s) are actually set up for WhatsApp
"""

import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

if not ACCOUNT_SID or not AUTH_TOKEN:
    print("❌ Missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN in .env")
    exit(1)

print("=" * 80)
print("DISCOVERING TWILIO WHATSAPP CHANNELS")
print("=" * 80)

try:
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    
    # Method 1: Check incoming phone numbers
    print("\n🔍 Checking Incoming Phone Numbers:")
    print("-" * 80)
    incoming_numbers = list(client.incoming_phone_numbers.stream(limit=50))
    
    if incoming_numbers:
        for num in incoming_numbers:
            print(f"\n  Phone: {num.phone_number}")
            print(f"  Name: {num.friendly_name}")
            print(f"  SMS: {num.sms_enabled}")
            print(f"  MMS: {num.mms_enabled}")
    else:
        print("  No incoming phone numbers found")
    
    # Method 2: Check messaging services (WhatsApp usually uses services)
    print("\n\n🔍 Checking Messaging Services:")
    print("-" * 80)
    try:
        services = list(client.messaging.services.stream(limit=50))
        
        if services:
            for service in services:
                print(f"\n  Service: {service.friendly_name} (SID: {service.sid})")
                
                # Check phone numbers in this service
                try:
                    phone_numbers = list(client.messaging.services(service.sid).phone_numbers.stream(limit=20))
                    if phone_numbers:
                        for pn in phone_numbers:
                            print(f"    ├─ Phone: {pn.phone_number}")
                except:
                    pass
                
                # Check short codes
                try:
                    short_codes = list(client.messaging.services(service.sid).short_codes.stream(limit=20))
                    if short_codes:
                        for sc in short_codes:
                            print(f"    ├─ Short Code: {sc.short_code}")
                except:
                    pass
        else:
            print("  No messaging services found")
    except Exception as e:
        print(f"  Could not check services: {e}")
    
    # Method 3: Try to get account details
    print("\n\n🔍 Account Information:")
    print("-" * 80)
    try:
        account = client.api.accounts(ACCOUNT_SID).fetch()
        print(f"\n  Account SID: {account.sid}")
        print(f"  Friendly Name: {account.friendly_name}")
        print(f"  Status: {account.status}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    
    if incoming_numbers:
        print("\n✅ Found phone numbers. Check if any are configured for WhatsApp.")
        print("\nTo use one of these with WhatsApp:")
        print("1. Go to Twilio Console > Messaging")
        print("2. Check if WhatsApp is enabled for any of these numbers")
        print("3. Update TWILIO_WHATSAPP_NUMBER in .env with the correct number")
    else:
        print("\n⚠️  No phone numbers found in your account!")
        print("\nYou may need to:")
        print("1. Purchase a phone number from Twilio Console")
        print("2. OR use the Twilio WhatsApp Sandbox number: +14155238886")
        print("3. Join the sandbox via WhatsApp: send 'join' to +1 415-523-8886")
    
    print("\n" + "=" * 80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nMake sure your credentials are correct:")
    print(f"  TWILIO_ACCOUNT_SID: {ACCOUNT_SID[:10]}...")
    print(f"  TWILIO_AUTH_TOKEN: {AUTH_TOKEN[:10] if AUTH_TOKEN else 'NOT SET'}...")
