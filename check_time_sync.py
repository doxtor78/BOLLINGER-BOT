#!/usr/bin/env python3
"""
Time synchronization checker for BitMEX API
This script helps diagnose timestamp issues that cause "expires is in the past" errors
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone


def check_system_time():
    """Check system time against multiple time sources"""
    print("=== System Time Check ===")
    
    # Local system time
    local_time = datetime.now(timezone.utc)
    local_timestamp = local_time.timestamp()
    print(f"Local UTC time: {local_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Local timestamp: {local_timestamp}")
    
    # Check against world time API
    try:
        response = requests.get('http://worldtimeapi.org/api/timezone/UTC', timeout=10)
        if response.status_code == 200:
            data = response.json()
            utc_time = pd.to_datetime(data['utc_datetime']).timestamp()
            diff = local_timestamp - utc_time
            print(f"World Time API timestamp: {utc_time}")
            print(f"Difference from World Time: {diff:.2f} seconds")
            if abs(diff) > 5:
                print("⚠️  WARNING: Your system time differs by more than 5 seconds!")
            else:
                print("✅ System time looks good relative to World Time API")
    except Exception as e:
        print(f"❌ Could not check World Time API: {e}")
    
    # Check against BitMEX testnet
    try:
        response = requests.get('https://testnet.bitmex.com/api/v1/announcement', timeout=10)
        if response.status_code == 200:
            server_time_str = response.headers.get('Date')
            if server_time_str:
                server_timestamp = pd.to_datetime(server_time_str).timestamp()
                diff = local_timestamp - server_timestamp
                print(f"BitMEX testnet timestamp: {server_timestamp}")
                print(f"Difference from BitMEX: {diff:.2f} seconds")
                if abs(diff) > 5:
                    print("⚠️  WARNING: Your system time differs by more than 5 seconds from BitMEX!")
                    print("This will cause 'expires is in the past' errors!")
                else:
                    print("✅ System time looks good relative to BitMEX")
                return diff
    except Exception as e:
        print(f"❌ Could not check BitMEX time: {e}")
    
    print()
    return 0


def check_network_latency():
    """Check network latency to BitMEX"""
    print("=== Network Latency Check ===")
    
    try:
        start_time = time.time()
        response = requests.get('https://testnet.bitmex.com/api/v1/announcement', timeout=10)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Network latency to BitMEX testnet: {latency:.2f} ms")
        
        if latency > 1000:
            print("⚠️  WARNING: High latency detected! This may cause API timeout issues.")
        elif latency > 500:
            print("⚠️  Moderate latency detected. Monitor for timeout issues.")
        else:
            print("✅ Network latency looks good")
            
    except Exception as e:
        print(f"❌ Could not check network latency: {e}")
    
    print()


def suggest_fixes(time_offset):
    """Suggest fixes based on detected issues"""
    print("=== Recommendations ===")
    
    if abs(time_offset) > 5:
        print("🔧 Time synchronization issues detected. Try these fixes:")
        print("   1. Sync your system clock:")
        print("      - macOS: System Preferences → Date & Time → Set date and time automatically")
        print("      - Linux: sudo ntpdate -s time.nist.gov")
        print("      - Windows: Settings → Time & Language → Date & Time → Sync now")
        print("   2. Check your timezone settings")
        print("   3. Restart your system to ensure time sync takes effect")
    else:
        print("✅ No major time synchronization issues detected")
    
    print("\n🔧 Additional recommendations:")
    print("   - The updated bot now includes automatic time synchronization")
    print("   - API requests now have retry logic to handle temporary failures")
    print("   - Connection health monitoring will reinitialize on persistent errors")
    print("   - Consider running the bot on a server with reliable internet")


if __name__ == "__main__":
    print("BitMEX Time Synchronization Checker")
    print("=" * 40)
    
    time_offset = check_system_time()
    check_network_latency()
    suggest_fixes(time_offset)
    
    print("\nAfter making any system time changes, restart your bot to ensure")
    print("the time synchronization takes effect.") 