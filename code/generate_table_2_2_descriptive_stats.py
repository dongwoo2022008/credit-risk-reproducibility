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
    
    # Variable mapping: Korean -> English with descriptions
    variable_mapping = {
        '대출시기': {
            'english': 'Loan Period',
            'description': 'Year of loan application (2007-2015)'
        },
        '취소횟수': {
            'english': 'Cancel Count',
            'description': 'Number of loan application cancellations'
        },
        '실패횟수': {
            'english': 'Fail Count',
            'description': 'Number of failed loan applications'
        },
        '성공횟수': {
            'english': 'Success Count',
            'description': 'Number of successful loan applications'
        },
        '총횟수': {
            'english': 'Total Count',
            'description': 'Total number of loan applications'
        },
        '성공률': {
            'english': 'Success Rate',
            'description': 'Success rate of loan applications (0-1)'
        },
        '성별(남0)': {
            'english': 'Gender',
            'description': 'Gender (0=Male, 1=Female)'
        },
        '나이': {
            'english': 'Age',
            'description': 'Age of applicant (years)'
        },
        '지역(수도권0)': {
            'english': 'Region',
            'description': 'Region (0=Seoul Metropolitan, 1=Other)'
        },
        '신용평점': {
            'english': 'Credit Score',
            'description': 'Credit score (0-950)'
        },
        '월DTI': {
            'english': 'Monthly DTI',
            'description': 'Monthly debt-to-income ratio'
        },
        '연소득(만원)': {
            'english': 'Annual Income',
            'description': 'Annual income (10,000 KRW)'
        },
        '월소득(만원)': {
            'english': 'Monthly Income',
            'description': 'Monthly income (10,000 KRW)'
        }
    }
    
    # Calculate statistics
    stats_list = []
    
    for korean_name, info in variable_mapping.items():
        stats = {
            'Variable': info['english'],
            'Description': info['description'],
            'Mean': df[korean_name].mean(),
            'Std': df[korean_name].std(),
            'Median': df[korean_name].median(),
            'Q1': df[korean_name].quantile(0.25),
            'Q3': df[korean_name].quantile(0.75),
            'Min': df[korean_name].min(),
            'Max': df[korean_name].max()
        }
        stats_list.append(stats)
    
    # Create DataFrame
    stats_df = pd.DataFrame(stats_list)
    
    # Round to 2 decimal places
    numeric_cols = ['Mean', 'Std', 'Median', 'Q1', 'Q3', 'Min', 'Max']
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
