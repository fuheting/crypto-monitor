import pandas as pd
import numpy as np
from functools import wraps


ALPHA_TEMPLATES = []


def rank(x):
    return x.rank(pct=True)


def ts_rank(x, window):
    return x.rolling(window).apply(lambda y: pd.Series(y).rank(pct=True).iloc[-1], raw=False)


def ts_mean(x, window):
    return x.rolling(window).mean()


def ts_std(x, window):
    return x.rolling(window).std()


def ts_zscore(x, window):
    mean = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - mean) / std


def delay(x, period):
    return x.shift(period)


def delta(x, period):
    return x.diff(period)


def sign(x):
    return np.sign(x)


def register_alpha(name):
    def decorator(func):
        ALPHA_TEMPLATES.append({'name': name, 'func': func})
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def generate_all_alphas(df):
    alphas = {}
    for template in ALPHA_TEMPLATES:
        try:
            alpha = template['func'](df)
            if alpha is not None and not alpha.isnull().all():
                alphas[template['name']] = alpha
        except Exception as e:
            pass
    return alphas


@register_alpha('momentum_001')
def alpha_momentum_001(df):
    return rank(df['returns_24h']) - rank(df['returns_168h'])


@register_alpha('momentum_002')
def alpha_momentum_002(df):
    return rank(df['returns_12h']) - rank(df['returns_48h'])


@register_alpha('momentum_003')
def alpha_momentum_003(df):
    return rank(delta(df['close'], 4)) - rank(delta(df['close'], 24))


@register_alpha('momentum_004')
def alpha_momentum_004(df):
    return rank(df['returns_24h']) * rank(df['volume_ratio_24h'])


@register_alpha('momentum_005')
def alpha_momentum_005(df):
    return rank(df['roc_24h']) - rank(df['roc_168h'])


@register_alpha('momentum_006')
def alpha_momentum_006(df):
    return ts_rank(df['returns_24h'], 24) - ts_rank(df['returns_168h'], 168)


@register_alpha('momentum_007')
def alpha_momentum_007(df):
    return rank(df['sma_12_48_cross']) * rank(df['returns_24h'])


@register_alpha('momentum_008')
def alpha_momentum_008(df):
    return rank(df['price_to_sma_24h']) - rank(df['price_to_sma_168h'])


@register_alpha('momentum_009')
def alpha_momentum_009(df):
    return rank(df['macd']) * rank(df['volume_ratio_24h'])


@register_alpha('momentum_010')
def alpha_momentum_010(df):
    return ts_zscore(df['returns_24h'], 24)


@register_alpha('momentum_011')
def alpha_momentum_011(df):
    return rank(df['returns_24h'] - df['returns_12h'])


@register_alpha('momentum_012')
def alpha_momentum_012(df):
    return rank(df['ema_12_48_cross']) * rank(df['returns_48h'])


@register_alpha('momentum_013')
def alpha_momentum_013(df):
    return rank(df['close'] / df['sma_48h'] - 1)


@register_alpha('momentum_014')
def alpha_momentum_014(df):
    return ts_rank(df['volume_ratio_24h'], 24) * rank(df['returns_24h'])


@register_alpha('momentum_015')
def alpha_momentum_015(df):
    return rank(df['price_position_168h'])


@register_alpha('mr_001')
def alpha_mr_001(df):
    return -rank(df['returns_12h']) * rank(df['volume_ratio_24h'])


@register_alpha('mr_002')
def alpha_mr_002(df):
    return -rank(df['close'] / df['sma_24h'] - 1)


@register_alpha('mr_003')
def alpha_mr_003(df):
    return -rank(df['returns_24h']) * rank(df['realized_vol_24h'])


@register_alpha('mr_004')
def alpha_mr_004(df):
    return -ts_zscore(df['returns_24h'], 24)


@register_alpha('mr_005')
def alpha_mr_005(df):
    return -rank(df['returns_48h'] - df['returns_24h'])


@register_alpha('mr_006')
def alpha_mr_006(df):
    return -rank(df['close'] / df['sma_168h'] - 1)


@register_alpha('mr_007')
def alpha_mr_007(df):
    return -rank(df['vwap_deviation_24h'])


@register_alpha('mr_008')
def alpha_mr_008(df):
    return -rank(df['rsi_14'] - 50)


@register_alpha('mr_009')
def alpha_mr_009(df):
    return -rank(df['macd_hist']) * rank(df['volume_ratio_24h'])


@register_alpha('mr_010')
def alpha_mr_010(df):
    return -rank(df['bb_position_24h'] - 0.5)


@register_alpha('mr_011')
def alpha_mr_011(df):
    return -rank(delta(df['close'], 4)) * rank(delta(df['volume'], 4))


@register_alpha('mr_012')
def alpha_mr_012(df):
    return -rank(df['returns_4h']) * rank(df['realized_vol_24h'])


@register_alpha('mr_013')
def alpha_mr_013(df):
    return -ts_rank(df['close'], 24) * ts_rank(df['volume_ratio_24h'], 24)


@register_alpha('mr_014')
def alpha_mr_014(df):
    return -rank(df['price_position_24h'] - df['price_position_168h'])


@register_alpha('mr_015')
def alpha_mr_015(df):
    return -rank(df['upper_shadow']) * rank(df['lower_shadow'])


@register_alpha('vol_001')
def alpha_vol_001(df):
    return rank(df['realized_vol_24h']) * rank(df['volume_ratio_24h'])


@register_alpha('vol_002')
def alpha_vol_002(df):
    return rank(df['atr_24h'] / df['close']) - rank(df['returns_24h'])


@register_alpha('vol_003')
def alpha_vol_003(df):
    return -rank(df['realized_vol_24h'] - df['realized_vol_168h'])


@register_alpha('vol_004')
def alpha_vol_004(df):
    return rank(df['garman_klass_24h']) * rank(df['volume_ratio_24h'])


@register_alpha('vol_005')
def alpha_vol_005(df):
    return rank(df['bb_width_24h']) - rank(df['returns_24h'])


@register_alpha('vol_006')
def alpha_vol_006(df):
    return rank(df['volume_momentum_24h']) * rank(df['returns_24h'])


@register_alpha('vol_007')
def alpha_vol_007(df):
    return ts_rank(df['realized_vol_24h'], 24) * ts_rank(df['volume_ratio_24h'], 24)


@register_alpha('vol_008')
def alpha_vol_008(df):
    return rank(df['atr_168h'] / df['atr_24h'])


@register_alpha('vol_009')
def alpha_vol_009(df):
    return rank(df['garman_klass_24h'] - df['garman_klass_168h'])


@register_alpha('vol_010')
def alpha_vol_010(df):
    return -rank(df['volume_ratio_168h']) * rank(df['returns_168h'])


@register_alpha('vwap_001')
def alpha_vwap_001(df):
    return ts_rank(df['vwap_deviation_24h'], 24)


@register_alpha('vwap_002')
def alpha_vwap_002(df):
    return rank(df['vwap_deviation_24h'] - df['vwap_deviation_168h'])


@register_alpha('vwap_003')
def alpha_vwap_003(df):
    return (df['vwap_24h'] - df['close']) / df['atr_24h']


@register_alpha('vwap_004')
def alpha_vwap_004(df):
    return rank(df['volume']) * rank(df['vwap_deviation_24h'])


@register_alpha('time_001')
def alpha_time_001(df):
    return rank(df['day_of_week']) * rank(df['returns_24h'])


@register_alpha('time_002')
def alpha_time_002(df):
    return rank(df['us_session']) * rank(df['returns_24h'])


@register_alpha('time_003')
def alpha_time_003(df):
    return rank(df['asian_session']) * rank(df['volume_ratio_24h'])


@register_alpha('combo_001')
def alpha_combo_001(df):
    return (rank(df['returns_24h']) + rank(df['vwap_deviation_24h'])) / 2


@register_alpha('combo_002')
def alpha_combo_002(df):
    return rank(df['returns_24h']) * (1 - rank(df['rsi_14'] / 100))


@register_alpha('combo_003')
def alpha_combo_003(df):
    return (rank(ts_zscore(df['returns_24h'], 24)) + rank(ts_zscore(df['volume_ratio_24h'], 24))) / 2


@register_alpha('combo_004')
def alpha_combo_004(df):
    return rank(df['returns_24h']) * rank(1 - df['bb_position_24h'])


@register_alpha('combo_005')
def alpha_combo_005(df):
    return (rank(df['price_to_sma_24h']) - rank(df['price_to_sma_168h'])) * rank(df['volume_ratio_24h'])
