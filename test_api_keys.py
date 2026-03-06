"""
Test GOAPI Keys Validity
Check which API keys are still valid and working
"""

import os
import requests
from typing import List, Tuple
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# Load all API keys
api_keys: List[Tuple[str, str]] = []
for i in range(1, 10):
    key = os.getenv(f"GOAPI_KEY_{i}", "").strip()
    if key:
        api_keys.append((f"GOAPI_KEY_{i}", key))

if not api_keys:
    key = os.getenv("GOAPI_KEY", "").strip()
    if key:
        api_keys.append(("GOAPI_KEY", key))

print("="*70)
print("🔍 TESTING GOAPI KEYS")
print("="*70)
print(f"\nFound {len(api_keys)} API key(s) in .env\n")

# Test symbol
TEST_SYMBOL = "BBRI"
# Use yesterday's date
test_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

GOAPI_BASE = "https://api.goapi.io"
url = f"{GOAPI_BASE}/stock/idx/{TEST_SYMBOL}/broker_summary"

valid_keys: List[Tuple[str, str]] = []
invalid_keys: List[Tuple[str, str]] = []
rate_limited_keys: List[Tuple[str, str]] = []

for key_name, key_value in api_keys:
    masked_key: str = key_value[:8] + "..." + key_value[-8:] if len(key_value) > 16 else key_value
    
    print(f"Testing {key_name}: {masked_key}")
    
    try:
        response = requests.get(
            url,
            headers={"X-API-KEY": key_value},
            params={"date": test_date, "investor": "ALL"},  # type: ignore[arg-type]
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print(f"  ✅ VALID - Working perfectly!")
                valid_keys.append((key_name, masked_key))
            else:
                print(f"  ⚠️  Response OK but status: {data.get('status')}")
                
        elif response.status_code == 401:
            print(f"  ❌ INVALID - Unauthorized (expired or wrong key)")
            invalid_keys.append((key_name, masked_key))
            
        elif response.status_code == 429:
            print(f"  ⏱️  RATE LIMITED - Key valid but quota exceeded")
            rate_limited_keys.append((key_name, masked_key))
            
        elif response.status_code == 403:
            print(f"  🚫 FORBIDDEN - No access to this endpoint")
            invalid_keys.append((key_name, masked_key))
            
        else:
            print(f"  ⚠️  HTTP {response.status_code}: {response.text[:100]}")
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Network Error: {str(e)[:80]}")
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:80]}")
    
    print()

# Summary
print("="*70)
print("📊 SUMMARY")
print("="*70)
print(f"\n✅ Valid Keys: {len(valid_keys)}")
for key_name, masked_key in valid_keys:
    print(f"   - {key_name}: {masked_key}")

print(f"\n⏱️  Rate Limited Keys: {len(rate_limited_keys)}")
for key_name, masked_key in rate_limited_keys:
    print(f"   - {key_name}: {masked_key}")

print(f"\n❌ Invalid Keys: {len(invalid_keys)}")
for key_name, masked_key in invalid_keys:
    print(f"   - {key_name}: {masked_key}")

print("\n" + "="*70)

if valid_keys:
    print("✅ You have working API keys!")
    print("   Your script should work with these keys.")
elif rate_limited_keys:
    print("⚠️  All keys are rate limited!")
    print("   Wait for quota reset or get new keys.")
else:
    print("❌ No valid API keys found!")
    print("   Please update your .env with valid GOAPI keys.")

print("\n💡 Get API keys at: https://goapi.io/dashboard")
print("="*70)
