import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features.engineering import compute_features
from src.alphas.templates import generate_all_alphas
from src.evaluation.cross_validator import create_rolling_folds, run_cross_validation


def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Loaded {len(df)} rows, date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def main():
    print("=" * 60)
    print("BTCUSDT Alpha Mining Pipeline")
    print("=" * 60)
    
    data_path = 'data/BTCUSDT.csv'
    df = load_data(data_path)
    
    print("\n" + "=" * 60)
    print("Computing features...")
    print("=" * 60)
    features_df = compute_features(df)
    print(f"Computed {len(features_df.columns)} features")
    
    warmup_rows = features_df.dropna().shape[0]
    print(f"Rows after dropping NaN: {warmup_rows}")
    
    print("\n" + "=" * 60)
    print("Creating rolling folds (2-year train / 1-year test)...")
    print("=" * 60)
    
    features_with_year = features_df.copy()
    features_with_year['year'] = features_with_year['timestamp'].dt.year
    
    unique_years = sorted(features_with_year['year'].unique())
    print(f"Available years: {unique_years}")
    
    folds = [
        {'fold_id': 1, 'train_years': [2020], 'test_year': 2021},
        {'fold_id': 2, 'train_years': [2021], 'test_year': 2022},
        {'fold_id': 3, 'train_years': [2022], 'test_year': 2023},
        {'fold_id': 4, 'train_years': [2023], 'test_year': 2024},
    ]
    
    print(f"Created {len(folds)} folds:")
    for fold in folds:
        print(f"  Fold {fold['fold_id']}: Train {fold['train_years']} -> Test {fold['test_year']}")
    
    print("\n" + "=" * 60)
    print("Generating alphas and running cross-validation...")
    print("=" * 60)
    
    all_fold_results, aggregated_results = run_cross_validation(
        features_with_year,
        generate_all_alphas,
        folds
    )
    
    print(f"\nEvaluated {len(aggregated_results)} alphas across {len(folds)} folds")
    
    aggregated_results = aggregated_results.sort_values('ic_mean', ascending=False)
    
    print("\n" + "=" * 60)
    print("TOP 20 ALPHAS (Ranked by Average IC)")
    print("=" * 60)
    
    top_20 = aggregated_results.head(20).reset_index(drop=True)
    top_20.insert(0, 'rank', range(1, 21))
    
    display_cols = ['rank', 'alpha_name', 'ic_mean', 'ic_std', 'ic_ir', 
                    'sharpe_mean', 'max_dd_mean', 'total_return_mean']
    print(top_20[display_cols].to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    
    output_file = 'results/top_20_alphas.csv'
    top_20.to_csv(output_file, index=False)
    print(f"\nSaved top 20 alphas to {output_file}")
    
    full_results_file = 'results/all_alphas_full.csv'
    aggregated_results.to_csv(full_results_file, index=False)
    print(f"Saved all alpha results to {full_results_file}")
    
    fold_results_file = 'results/alpha_by_fold.csv'
    all_fold_results.to_csv(fold_results_file, index=False)
    print(f"Saved fold-level results to {fold_results_file}")
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total alphas evaluated: {len(aggregated_results)}")
    print(f"Positive IC mean alphas: {(aggregated_results['ic_mean'] > 0).sum()}")
    print(f"Best IC mean: {aggregated_results['ic_mean'].max():.4f}")
    print(f"Best IC IR: {aggregated_results['ic_ir'].max():.4f}")
    print(f"Best Sharpe: {aggregated_results['sharpe_mean'].max():.4f}")
    
    return top_20


if __name__ == "__main__":
    main()
