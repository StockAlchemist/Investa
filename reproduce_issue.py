
import pandas as pd
from datetime import date
import logging

def reproduce():
    # Create a Series with DatetimeIndex
    dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
    s = pd.Series([1, 2, 3], index=dates)
    
    print(f"Index type: {type(s.index)}")
    print(f"Index dtype: {s.index.dtype}")
    
    split_date = date(2020, 1, 2)
    print(f"Split date type: {type(split_date)}")
    
    try:
        # This mimics the failing line: mask = forward_split_factor.index < split_date
        mask = s.index < split_date
        print("Comparison successful")
        print(mask)
    except Exception as e:
        print(f"Caught expected exception: {e}")

if __name__ == "__main__":
    reproduce()
