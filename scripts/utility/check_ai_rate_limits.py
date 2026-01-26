#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          check_ai_rate_limits.py
 Purpose:       Utility to check Gemini API connectivity and rate limits for 
                the model fallback chain.
 Usage:         PYTHONPATH=src python scripts/utility/check_ai_rate_limits.py
-------------------------------------------------------------------------------
"""

import os
import sys
import time
import json
import logging
import requests
import datetime

# Ensure 'src' is in the python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    import config
    from server.ai_analyzer import FALLBACK_MODELS
except ImportError as e:
    print(f"CRITICAL: Failed to import modules. Error: {e}")
    sys.exit(1)

def check_model_status(model_name, api_key):
    """
    Probes a specific model with a minimal request to check status.
    """
    base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [{
            "parts": [{"text": "Say 'OK'"}]
        }],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            return "✅ AVAILABLE", response.headers
        elif response.status_code == 429:
            return "⏳ RATE LIMITED (429)", response.headers
        elif response.status_code == 400:
            return "❌ BAD REQUEST (400)", response.headers
        elif response.status_code == 403:
            return "❌ FORBIDDEN (403)", response.headers
        elif response.status_code == 404:
            return "❌ NOT FOUND (404)", response.headers
        else:
            return f"❓ UNKNOWN STATUS ({response.status_code})", response.headers
            
    except Exception as e:
        return f"💥 ERROR: {str(e)}", {}

def main():
    print("="*60)
    print("        INVESTA AI RATE LIMIT & CONNECTIVITY CHECK")
    print("="*60)
    
    api_key = config.GEMINI_API_KEY
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in environment or .env file.")
        sys.exit(1)
        
    print(f"API Key found: {'*'*len(api_key[:-4]) + api_key[-4:]}")
    
    print(f"Checking {len(FALLBACK_MODELS)} models in current fallback chain...\n")
    
    results = []
    
    # Header
    print(f"{'Model Name':<25} | {'Status':<30}")
    print("-" * 60)
    
    for model in FALLBACK_MODELS:
        status, headers = check_model_status(model, api_key)
        print(f"{model:<25} | {status}")
        
        # Check for rate limit headers
        # Common headers: x-ratelimit-reset, retry-after
        ratelimit_info = []
        for h in headers:
            if 'ratelimit' in h.lower() or 'retry-after' in h.lower():
                ratelimit_info.append(f"  -> {h}: {headers[h]}")
        
        if ratelimit_info:
            for info in ratelimit_info:
                print(info)
        elif status.startswith("⏳"):
            # Estimate renewal time (Midnight PT)
            # PT is UTC-8 (PST) or UTC-7 (PDT). 
            # Midnight PT is 08:00 GMT.
            # Local time is UTC+7, so Midnight PT is 15:00 local time.
            now_local = datetime.datetime.now()
            reset_local = now_local.replace(hour=15, minute=0, second=0, microsecond=0)
            
            if now_local >= reset_local:
                reset_local += datetime.timedelta(days=1)
                
            wait_time = reset_local - now_local
            hours, remainder = divmod(wait_time.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            
            reset_time_str = reset_local.strftime("%H:%M")
            print(f"  -> Estimated daily reset in: {hours}h {minutes}m (approx. {reset_time_str} local time)")
            print("  -> Note: If this is a minute-based limit, it may reset in < 1 minute.")
        
        results.append((model, status))
        # Small delay between probes to avoid triggering rate limit ourselves
        time.sleep(1)
        
    print("\n" + "="*60)
    summary = "System Ready" if any("AVAILABLE" in s for m, s in results) else "System Restricted"
    print(f"OVERALL STATUS: {summary}")
    print("="*60)
    
    if summary == "System Ready":
        print("\nYou can proceed to run 'python src/ai_review_worker.py'.")
    else:
        print("\nReview the errors above. You may need to wait for rate limits to reset.")

if __name__ == "__main__":
    main()
