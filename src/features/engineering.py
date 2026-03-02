import pandas as pd
import numpy as np


def compute_features(df):
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    df['returns_1h'] = close.pct_change(1)
    df['returns_4h'] = close.pct_change(4)
    df['returns_12h'] = close.pct_change(12)
    df['returns_24h'] = close.pct_change(24)
    df['returns_48h'] = close.pct_change(48)
    df['returns_168h'] = close.pct_change(168)
    
    df['log_returns_1h'] = np.log(close / close.shift(1))
    df['log_returns_24h'] = np.log(close / close.shift(24))
    df['log_returns_168h'] = np.log(close / close.shift(168))
    
    for window in [12, 24, 48, 168]:
        df[f'roc_{window}h'] = (close - close.shift(window)) / close.shift(window)
    
    for window in [24, 48, 168]:
        df[f'realized_vol_{window}h'] = df['returns_1h'].rolling(window).std()
    
    df['atr_24h'] = compute_atr(high, low, close, 24)
    df['atr_168h'] = compute_atr(high, low, close, 168)
    
    df['garman_klass_24h'] = compute_garman_klass(high, low, close, open_, 24)
    df['garman_klass_168h'] = compute_garman_klass(high, low, close, open_, 168)
    
    for window in [24, 48, 168]:
        df[f'volume_ma_{window}h'] = volume.rolling(window).mean()
    
    df['volume_ratio_24h'] = volume / df['volume_ma_24h']
    df['volume_ratio_168h'] = volume / df['volume_ma_168h']
    
    df['volume_momentum_24h'] = volume.pct_change(24)
    df['volume_momentum_168h'] = volume.pct_change(168)
    
    df['vwap_24h'] = compute_vwap(high, low, close, volume, 24)
    df['vwap_deviation_24h'] = (close - df['vwap_24h']) / df['vwap_24h']
    
    df['vwap_168h'] = compute_vwap(high, low, close, volume, 168)
    df['vwap_deviation_168h'] = (close - df['vwap_168h']) / df['vwap_168h']
    
    df['rsi_14'] = compute_rsi(close, 14)
    df['rsi_24'] = compute_rsi(close, 24)
    
    df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(close)
    
    bb_24 = compute_bollinger_bands(close, 24)
    df['bb_position_24h'] = (close - bb_24['lower']) / (bb_24['upper'] - bb_24['lower'])
    df['bb_width_24h'] = (bb_24['upper'] - bb_24['lower']) / bb_24['middle']
    
    for window in [12, 24, 48, 168]:
        df[f'sma_{window}h'] = close.rolling(window).mean()
        df[f'ema_{window}h'] = close.ewm(span=window, adjust=False).mean()
    
    df['price_to_sma_24h'] = close / df['sma_24h'] - 1
    df['price_to_sma_168h'] = close / df['sma_168h'] - 1
    
    df['sma_12_48_cross'] = (df['sma_12h'] > df['sma_48h']).astype(int)
    df['sma_24_168_cross'] = (df['sma_24h'] > df['sma_168h']).astype(int)
    df['ema_12_48_cross'] = (df['ema_12h'] > df['ema_48h']).astype(int)
    
    df['high_low_range_24h'] = (high.rolling(24).max() - low.rolling(24).min())
    df['price_position_24h'] = (close - low.rolling(24).min()) / df['high_low_range_24h']
    
    df['high_low_range_168h'] = (high.rolling(168).max() - low.rolling(168).min())
    df['price_position_168h'] = (close - low.rolling(168).min()) / df['high_low_range_168h']
    
    df['close_minus_open'] = close - open_
    df['high_minus_close'] = high - close
    df['close_minus_low'] = close - low
    
    df['upper_shadow'] = (high - np.maximum(open_, close)) / (high - low)
    df['lower_shadow'] = (np.minimum(open_, close) - low) / (high - low)
    df['body_ratio'] = np.abs(close - open_) / (high - low)
    
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    df['us_session'] = ((df['hour_of_day'] >= 13) & (df['hour_of_day'] <= 21)).astype(int)
    df['asian_session'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] <= 8)).astype(int)
    
    return df


def compute_atr(high, low, close, window):
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def compute_garman_klass(high, low, close, open_prices, window):
    log_hl = np.log(high / low)
    log_co = np.log(close / open_prices)
    gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
    return np.sqrt(gk.rolling(window).mean())


def compute_vwap(high, low, close, volume, window):
    typical_price = (high + low + close) / 3
    return (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()


def compute_rsi(close, window):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_bollinger_bands(close, window, num_std=2):
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return pd.DataFrame({'upper': upper, 'middle': middle, 'lower': lower})
