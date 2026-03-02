import pandas as pd
import numpy as np


def compute_ic(alpha, returns, method='pearson'):
    alpha = alpha.dropna()
    common_idx = alpha.index.intersection(returns.index)
    if len(common_idx) < 10:
        return np.nan
    
    alpha_aligned = alpha.loc[common_idx]
    returns_aligned = returns.loc[common_idx]
    
    if method == 'pearson':
        return alpha_aligned.corr(returns_aligned)
    else:
        return alpha_aligned.rank().corr(returns_aligned.rank())


def compute_sharpe(returns, risk_free_rate=0.0):
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    
    excess_returns = returns - risk_free_rate / 24
    return np.sqrt(24) * excess_returns.mean() / excess_returns.std()


def compute_max_drawdown(cumulative_returns):
    cumulative_returns = cumulative_returns.dropna()
    if len(cumulative_returns) == 0:
        return np.nan
    
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def compute_returns(alpha, forward_returns, threshold=0.0):
    signal = alpha.copy()
    signal = signal.dropna()
    
    common_idx = signal.index.intersection(forward_returns.index)
    if len(common_idx) < 10:
        return pd.Series(dtype=float)
    
    signal_aligned = signal.loc[common_idx]
    returns_aligned = forward_returns.loc[common_idx]
    
    positions = (signal_aligned > threshold).astype(int) - (signal_aligned < -threshold).astype(int)
    position_returns = positions.shift(1) * returns_aligned
    
    return position_returns.dropna()


def compute_cumulative_returns(position_returns):
    return (1 + position_returns).cumprod() - 1


def evaluate_alpha(alpha, forward_returns):
    ic = compute_ic(alpha, forward_returns, 'pearson')
    ic_spearman = compute_ic(alpha, forward_returns, 'spearman')
    
    position_returns = compute_returns(alpha, forward_returns)
    
    if len(position_returns) == 0:
        return {
            'ic': np.nan,
            'ic_spearman': np.nan,
            'sharpe': np.nan,
            'max_drawdown': np.nan,
            'total_return': np.nan,
            'win_rate': np.nan,
            'n_positions': len(position_returns),
        }
    
    sharpe = compute_sharpe(position_returns)
    cumulative_returns = compute_cumulative_returns(position_returns)
    max_dd = compute_max_drawdown(cumulative_returns)
    total_return = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else np.nan
    
    win_rate = (position_returns > 0).mean()
    
    return {
        'ic': ic,
        'ic_spearman': ic_spearman,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'total_return': total_return,
        'win_rate': win_rate,
        'n_positions': len(position_returns),
    }


def evaluate_alphas(alphas, forward_returns):
    results = []
    for alpha_name, alpha_series in alphas.items():
        metrics = evaluate_alpha(alpha_series, forward_returns)
        metrics['alpha_name'] = alpha_name
        results.append(metrics)
    
    return pd.DataFrame(results)
