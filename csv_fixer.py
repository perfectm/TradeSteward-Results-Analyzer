#!/usr/bin/env python3

import csv
import pandas as pd
import sys

def fix_csv_parsing(input_file, output_file=None):
    """
    Fix CSV parsing issues by using Python's csv module for more robust parsing
    """
    if output_file is None:
        output_file = input_file.replace('.csv', '_fixed.csv')
    
    print(f"Reading {input_file}...")
    
    # Read the CSV using Python's csv module which handles complex quoting better
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        # Try different dialects to handle the CSV properly
        try:
            # First, try to detect the dialect
            sample = f.read(8192)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=',')
            reader = csv.reader(f, dialect=dialect)
        except:
            # Fall back to default comma-separated
            f.seek(0)
            reader = csv.reader(f, delimiter=',', quotechar='"')
        
        for row_num, row in enumerate(reader):
            if row_num == 0:
                header = row
                expected_cols = len(header)
                print(f"Header has {expected_cols} columns")
                print(f"Last few columns: {header[-5:]}")
            else:
                if len(row) != expected_cols:
                    print(f"Row {row_num} has {len(row)} columns, expected {expected_cols}")
                    # Pad or truncate to match header
                    if len(row) < expected_cols:
                        row.extend([''] * (expected_cols - len(row)))
                    elif len(row) > expected_cols:
                        row = row[:expected_cols]
            
            rows.append(row)
            
            if row_num < 5:  # Show first few rows for debugging
                print(f"Row {row_num}: {len(row)} columns")
                if row_num > 0:
                    print(f"  Last column value: '{row[-1]}'")
    
    print(f"Read {len(rows)} total rows")
    
    # Write the cleaned CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Wrote cleaned CSV to {output_file}")
    
    # Now test pandas parsing
    print("\nTesting pandas parsing...")
    df = pd.read_csv(output_file)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if 'TotalNetProfitLoss' in df.columns:
        tnpl = df['TotalNetProfitLoss']
        print(f"\nTotalNetProfitLoss column:")
        print(f"  Data type: {tnpl.dtype}")
        print(f"  Non-null count: {tnpl.count()}")
        print(f"  Null count: {tnpl.isnull().sum()}")
        print(f"  Sample values: {tnpl.dropna().head().tolist()}")
        print(f"  Unique non-null values: {tnpl.dropna().nunique()}")
    else:
        print("TotalNetProfitLoss column not found!")
    
    return output_file

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else '/Users/perfectm/Projects/TradeSteward-Results-Analyzer/tradeSteward-performanceLogs-cleaned.csv'
    fix_csv_parsing(input_file)
