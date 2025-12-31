
import json
import os

CONFIG_PATH = os.path.expanduser("~/Library/Application Support/StockAlchemist/Investa/gui_config.json")
NEW_DB_PATH = os.path.expanduser("~/Library/Application Support/StockAlchemist/Investa/investa_transactions.db")

def update_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found at {CONFIG_PATH}")
        return

    try:
        with open(CONFIG_PATH, 'r') as f:
            data = json.load(f)
        
        print(f"Old path: {data.get('transactions_file')}")
        data['transactions_file'] = NEW_DB_PATH
        print(f"New path: {NEW_DB_PATH}")

        with open(CONFIG_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        print("Config updated successfully.")
        
    except Exception as e:
        print(f"Error updating config: {e}")

if __name__ == "__main__":
    update_config()
