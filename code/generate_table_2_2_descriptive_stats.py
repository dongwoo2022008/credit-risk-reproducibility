"""
Generate Table 2-2: Descriptive Statistics of Structured Variables (Raw Data)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import config

def generate_descriptive_stats():
    """Generate descriptive statistics table from raw data"""
    
    print("="*80)
    print("Generating Table 2-2: Descriptive Statistics")
    print("="*80)
    
    # Load raw data
    df = pd.read_excel(config.DATA_DIR / "raw" / "sentiment_scoring.25.12.30.xlsx")
    
    # Structured variables (13 variables)
    structured_vars = [
        '대출시기', '취소횟수', '실패횟수', '성공횟수', '총횟수', '성공률',
        '성별(남0)', '나이', '지역(수도권0)', '신용평점', '월DTI', '연소득(만원)', '월소득(만원)'
    ]
    
    # Calculate statistics
    stats_list = []
    
    for var in structured_vars:
        stats = {
            'Variable': var,
            'Mean': df[var].mean(),
            'Standard Deviation': df[var].std(),
            'Median': df[var].median(),
            'Q1': df[var].quantile(0.25),
            'Q3': df[var].quantile(0.75),
            'Min': df[var].min(),
            'Max': df[var].max()
        }
        stats_list.append(stats)
    
    # Create DataFrame
    stats_df = pd.DataFrame(stats_list)
    
    # Round to 2 decimal places
    numeric_cols = ['Mean', 'Standard Deviation', 'Median', 'Q1', 'Q3', 'Min', 'Max']
    stats_df[numeric_cols] = stats_df[numeric_cols].round(2)
    
    # Save to CSV
    output_path = config.RESULTS_DIR / "tables" / "table_2_2_descriptive_statistics.csv"
    stats_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nTable saved to: {output_path}")
    print("\nTable 2-2: Descriptive Statistics")
    print(stats_df.to_string(index=False))
    
    return stats_df

if __name__ == '__main__':
    generate_descriptive_stats()
