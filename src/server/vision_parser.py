import base64
import json
import logging
import requests
import os
import time
from typing import List, Dict, Any, Optional
import config

# Models that support Vision/Multi-modal
VISION_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

def parse_document_with_ai(file_path: str, mime_type: str) -> List[Dict[str, Any]]:
    """
    Parses a brokerage statement (PDF or Image) using Gemini Vision.
    Returns a list of transaction dictionaries compatible with Investa's schema.
    """
    if not config.GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not found. Cannot perform AI parsing.")
        return []

    try:
        # Read and encode the file
        with open(file_path, "rb") as f:
            file_data = base64.b64encode(f.read()).decode("utf-8")
        
        prompt = """
        Extract all investment transactions (BUY, SELL, DIVIDEND) from the attached brokerage statement.
        
        Important instructions:
        1. Identify the 'Transaction Date', 'Type', 'Symbol/Description', 'Quantity', 'Price', 'Total Amount', and 'Commission/Fees'.
        2. For 'Type', only use "Buy", "Sell", or "Dividend".
        3. For 'Symbol', extract the ticker (e.g., AAPL). If not clear, use the description.
        4. Ensure 'Quantity' and 'Price' are positive numbers.
        5. 'Total Amount' should be the net proceeds or cost.
        6. If the document has multiple pages, extract transactions from all of them.
        7. If 'Currency' is mentioned (e.g., USD, THB), include it. Default to USD if unsure.
        
        Return a JSON array of objects with these keys:
        [
          {
            "Date": "YYYY-MM-DD HH:MM:SS",
            "Type": "Buy/Sell/Dividend",
            "Symbol": "TICKER",
            "Quantity": 0.0,
            "Price/Share": 0.0,
            "Total Amount": 0.0,
            "Commission": 0.0,
            "Local Currency": "USD",
            "Account": "Account#",
            "Note": "AI Extracted from statement"
          }
        ]
        
        Return ONLY the JSON array. No markdown, no explanations.
        """

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": file_data
                        }
                    }
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }

        # Fallback chain for reliability
        for model in VISION_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={config.GEMINI_API_KEY}"
            
            try:
                logging.info(f"Vision Parser: Uploading {os.path.basename(file_path)} to {model}...")
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=90 # Vision takes longer
                )
                
                if response.status_code == 429:
                    logging.warning(f"Vision Parser: Rate limit hit for {model}. Trying next...")
                    continue
                
                response.raise_for_status()
                result_data = response.json()
                
                # Extract text from response
                if 'candidates' in result_data and result_data['candidates']:
                    content = result_data['candidates'][0]['content']['parts'][0]['text']
                    
                    # Robust JSON extraction
                    try:
                        start_idx = content.find('[')
                        end_idx = content.rfind(']')
                        if start_idx != -1 and end_idx != -1:
                            json_str = content[start_idx : end_idx + 1]
                            transactions = json.loads(json_str)
                            
                            # Type validation for each transaction
                            for tx in transactions:
                                # Ensure required fields exist or have defaults
                                tx.setdefault("Type", "Buy")
                                tx.setdefault("Quantity", 0.0)
                                tx.setdefault("Price/Share", 0.0)
                                tx.setdefault("Total Amount", 0.0)
                                tx.setdefault("Commission", 0.0)
                                tx.setdefault("Local Currency", "USD")
                                
                            logging.info(f"Vision Parser: Successfully extracted {len(transactions)} transactions using {model}.")
                            return transactions
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.warning(f"Vision Parser: Failed to parse JSON from {model}: {e}")
                        continue
                
            except Exception as e:
                logging.warning(f"Vision Parser: Attempt with {model} failed: {e}")
                continue

        logging.error("Vision Parser: All models failed to extract data.")
        return []

    except Exception as e:
        logging.error(f"Vision Parser: Unexpected error: {e}", exc_info=True)
        return []
