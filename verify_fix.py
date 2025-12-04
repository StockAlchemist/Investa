
import pandas as pd
from datetime import date
import logging

def verify():
    # Create a Series with DatetimeIndex
    dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
    s = pd.Series([1, 2, 3], index=dates)
    
    print(f"Original Index type: {type(s.index)}")
    
    # Simulate the fixed logic: Always convert to date
    try:
        s.index = pd.to_datetime(s.index, errors="coerce").date
        s = s[pd.notnull(s.index)]
        print(f"Converted Index type: {type(s.index)}")
        print(f"Converted Index dtype: {s.index.dtype}")
        
        split_date = date(2020, 1, 2)
        
        # This comparison should now work
        mask = s.index < split_date
        print("Comparison successful!")
        print(mask)
        
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    verify()
