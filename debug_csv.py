#!/usr/bin/env python3

import pandas as pd
import numpy as np

def debug_csv():
    """Debug the CSV file by reading first 5 lines and showing detailed info"""
    
    csv_file = "tradeSteward-performanceLogs-1753563010.csv"
    
    print("=" * 60)
    print("CSV DEBUG ANALYSIS")
    print("=" * 60)
    
    try:
        # Read only first 5 rows
        df = pd.read_csv(csv_file, nrows=5)
        
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print("\nColumn names:")
        for i, col in enumerate(df.columns):
            print(f"{i+1:2d}. '{col}' (length: {len(col)})")
        
        print("\n" + "=" * 60)
        print("COLUMN DATA TYPES:")
        print("=" * 60)
        for col in df.columns:
            print(f"{col}: {df[col].dtype}")
        
        print("\n" + "=" * 60)
        print("FIRST 5 ROWS:")
        print("=" * 60)
        print(df.to_string())
        
        print("\n" + "=" * 60)
        print("TOTALNETPROFITLOSS COLUMN ANALYSIS:")
        print("=" * 60)
        
        if 'TotalNetProfitLoss' in df.columns:
            col_data = df['TotalNetProfitLoss']
            print(f"Column exists: YES")
            print(f"Data type: {col_data.dtype}")
            print(f"Values: {col_data.tolist()}")
            print(f"Non-null count: {col_data.notna().sum()}")
            print(f"Null count: {col_data.isna().sum()}")
            
            # Try to convert to numeric
            print("\nTrying numeric conversion:")
            numeric_converted = pd.to_numeric(col_data, errors='coerce')
            print(f"After conversion - data type: {numeric_converted.dtype}")
            print(f"After conversion - values: {numeric_converted.tolist()}")
            print(f"After conversion - non-null count: {numeric_converted.notna().sum()}")
            print(f"After conversion - null count: {numeric_converted.isna().sum()}")
            
            # Check for whitespace issues
            print("\nChecking for whitespace issues:")
            for i, val in enumerate(col_data):
                if pd.notna(val):
                    print(f"Row {i}: '{val}' (type: {type(val)}, repr: {repr(val)})")
        else:
            print("TotalNetProfitLoss column NOT FOUND!")
            print("Available columns containing 'profit' or 'loss' (case insensitive):")
            matching_cols = [col for col in df.columns if 'profit' in col.lower() or 'loss' in col.lower()]
            for col in matching_cols:
                print(f"  - {col}")
        
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS:")
        print("=" * 60)
        print(df.describe(include='all'))
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_csv()
