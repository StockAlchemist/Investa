import pandas as pd
import sqlite3
import os

db_path = "my_transactions.db"
if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM transactions", conn)
conn.close()

print("Columns:", df.columns)
if "Local Currency" in df.columns:
    currencies = df["Local Currency"].unique()
    print("Currencies:", currencies)
    if "IDLOX" in currencies:
        print("FOUND IDLOX in Local Currency!")
        # Show rows
        print(df[df["Local Currency"] == "IDLOX"])
else:
    print("Local Currency column not found")
