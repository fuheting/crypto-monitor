import pandas as pd
import numpy as np


def create_rolling_folds(df, train_years=2, test_year=1):
    df = df.copy()
    df['year'] = df['timestamp'].dt.year
    
    unique_years = sorted(df['year'].unique())
    
    folds = []
    for i in range(len(unique_years) - train_years - test_year + 1):
        train_start_idx = i
        train_end_idx = i + train_years
        test_idx = i + train_years
        
        train_years_list = unique_years[train_start_idx:train_end_idx]
        test_year_val = unique_years[test_idx]
        
        train_years_range = (unique_years[train_start_idx], unique_years[train_end_idx - 1])
        
        fold = {
            'fold_id': len(folds) + 1,
            'train_years': train_years_list,
            'test_year': test_year_val,
            'train_start': train_years_range[0],
            'train_end': train_years_range[1],
        }
        folds.append(fold)
    
    return folds


def split_data_by_folds(df, folds):
    fold_data = {}
    
    for fold in folds:
        test_year = fold['test_year']
        train_years = fold['train_years']
        
        train_mask = df['year'].isin(train_years)
        test_mask = df['year'] == test_year
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        train_df = train_df.drop(columns=['year'], errors='ignore')
        test_df = test_df.drop(columns=['year'], errors='ignore')
        
        fold_data[fold['fold_id']] = {
            'train': train_df,
            'test': test_df,
            'train_years': train_years,
            'test_year': test_year,
        }
    
    return fold_data


def run_cross_validation(features_df, alpha_generator_func, folds):
    fold_data = split_data_by_folds(features_df, folds)
    
    all_fold_results = []
    alpha_fold_metrics = {}
    
    for fold_id, data in fold_data.items():
        train_df = data['train']
        test_df = data['test']
        
        alphas_train = alpha_generator_func(train_df)
        
        if not alphas_train:
            continue
        
        alphas_test = alpha_generator_func(test_df)
        
        if not alphas_test:
            continue
        
        common_alphas = set(alphas_train.keys()) & set(alphas_test.keys())
        
        forward_returns = test_df['returns_1h'].shift(-1).dropna()
        
        from src.evaluation.evaluator import evaluate_alphas
        
        fold_results = evaluate_alphas(alphas_test, forward_returns)
        fold_results['fold_id'] = fold_id
        fold_results['test_year'] = data['test_year']
        
        all_fold_results.append(fold_results)
        
        for alpha_name in alphas_test.keys():
            if alpha_name not in alpha_fold_metrics:
                alpha_fold_metrics[alpha_name] = []
            
            alpha_metrics = fold_results[fold_results['alpha_name'] == alpha_name]
            if len(alpha_metrics) > 0:
                alpha_fold_metrics[alpha_name].append({
                    'fold_id': fold_id,
                    'ic': alpha_metrics['ic'].values[0],
                    'ic_spearman': alpha_metrics['ic_spearman'].values[0],
                    'sharpe': alpha_metrics['sharpe'].values[0],
                    'max_drawdown': alpha_metrics['max_drawdown'].values[0],
                    'total_return': alpha_metrics['total_return'].values[0],
                })
    
    all_results = pd.concat(all_fold_results, ignore_index=True)
    
    aggregated_results = []
    for alpha_name, fold_metrics_list in alpha_fold_metrics.items():
        fold_metrics_df = pd.DataFrame(fold_metrics_list)
        
        ic_values = fold_metrics_df['ic'].values
        ic_mean = np.nanmean(ic_values)
        ic_std = np.nanstd(ic_values)
        ic_ir = ic_mean / ic_std if ic_std != 0 and not np.isnan(ic_std) else np.nan
        
        sharpe_values = fold_metrics_df['sharpe'].values
        sharpe_mean = np.nanmean(sharpe_values)
        
        max_dd_values = fold_metrics_df['max_drawdown'].values
        max_dd_mean = np.nanmean(max_dd_values)
        
        total_ret_values = fold_metrics_df['total_return'].values
        total_ret_mean = np.nanmean(total_ret_values)
        
        aggregated_results.append({
            'alpha_name': alpha_name,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_fold1': ic_values[0] if len(ic_values) > 0 else np.nan,
            'ic_fold2': ic_values[1] if len(ic_values) > 1 else np.nan,
            'ic_fold3': ic_values[2] if len(ic_values) > 2 else np.nan,
            'sharpe_mean': sharpe_mean,
            'max_dd_mean': max_dd_mean,
            'total_return_mean': total_ret_mean,
        })
    
    aggregated_df = pd.DataFrame(aggregated_results)
    
    return all_results, aggregated_df
