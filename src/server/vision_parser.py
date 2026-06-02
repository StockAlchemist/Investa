import base64
import json
import logging
import requests
import os
from typing import List, Dict, Any
import config

# Models that support Vision/Multi-modal
VISION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-pro-latest",
    "gemini-3-pro-preview"
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
        Extract ALL investment-related activities and cash movements from the attached brokerage statement.
        
        Important instructions:
        1. Scan EVERY page. Most transactions are in "Transaction History" or "Activity Detail".
        2. Categorize each activity into: "Buy", "Sell", "Dividend", "Interest", "Tax", "Deposit", "Withdrawal", "Fee", "Transfer".
        3. For 'Symbol':
           - Use the ticker (e.g., AAPL).
           - Use "$CASH" for non-security items.
           - For Dividend/Tax, use the ticker of the paying security.
        4. ALL numerical values (Quantity, Price/Share, Total Amount, Commission) must be POSITIVE absolute values.
           The system will handle the sign based on the Type.
        5. 'Total Amount' is the net amount (Quantity * Price) BEFORE commission/fees if possible, or the net impact.
        
        Also, extract the following summary information:
        - Statement Period (Start Date and End Date)
        - Ending Cash Balance (The final cash/money market balance at the end of the period)
        - Account Number
        
        Return a JSON object with this structure:
        {
          "statement_info": {
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD",
            "ending_cash_balance": 0.0,
            "account_number": "string"
          },
          "transactions": [
            {
              "Date": "YYYY-MM-DD HH:MM:SS",
              "Type": "Buy/Sell/Dividend/Interest/Tax/Deposit/Withdrawal/Fee/Transfer",
              "Symbol": "TICKER",
              "Quantity": 0.0,
              "Price/Share": 0.0,
              "Total Amount": 0.0,
              "Commission": 0.0,
              "Local Currency": "USD",
              "Account": "Account#",
              "Note": "Full description"
            }
          ]
        }
        
        Return ONLY the JSON. No markdown.
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
                        # Find the first { or [ to start parsing
                        start_obj = content.find('{')
                        start_arr = content.find('[')
                        
                        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
                            # It's an object
                            end_obj = content.rfind('}')
                            json_str = content[start_obj : end_obj + 1]
                            data = json.loads(json_str)
                            if isinstance(data, dict) and "transactions" in data:
                                transactions = data["transactions"]
                                # Store statement info in metadata if needed (optional for now)
                                if "statement_info" in data:
                                    logging.info(f"Vision Parser: Extracted statement info: {data['statement_info']}")
                            else:
                                transactions = [data] if isinstance(data, dict) else []
                        elif start_arr != -1:
                            # It's an array
                            end_arr = content.rfind(']')
                            json_str = content[start_arr : end_arr + 1]
                            transactions = json.loads(json_str)
                        else:
                            transactions = []

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
