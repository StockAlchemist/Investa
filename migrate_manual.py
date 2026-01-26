import sys
import os
import logging

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

logging.basicConfig(level=logging.INFO)

from db_utils import initialize_database

if __name__ == "__main__":
    print("Running manual migration...")
    try:
        conn = initialize_database()
        print("Migration complete.")
        if conn:
            conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
