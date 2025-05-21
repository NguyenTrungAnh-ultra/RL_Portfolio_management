import numpy as np
import pandas as pd

def precompute_technical_indicators(close, volume, df, lookback_period=252):
    """
    Tính toán trước các chỉ báo kỹ thuật tối ưu cho chiến lược theo xu hướng kết hợp volume.
    Tập trung vào các chỉ báo hiệu quả trong uptrend và sideway.
    
    Parameters:
    -----------
    close : pandas.DataFrame
        DataFrame với giá đóng cửa lịch sử (cột là tài sản, index là ngày)
    volume : pandas.DataFrame
        DataFrame với khối lượng giao dịch lịch sử (cột là tài sản, index là ngày)
    df : pandas.DataFrame
        DataFrame với giá cho giai đoạn training/testing (tập con của close)
    lookback_period : int
        Khoảng thời gian nhìn lại tối đa cần thiết cho tính toán (mặc định: 252 ngày)
    
    Returns:
    --------
    dict
        Dictionary lồng nhau với các chỉ báo được tính toán trước:
        {date: {asset: {indicator_name: value}}}
    """
    # Khởi tạo dictionary kết quả
    precomputed_indicators = {}
    
    # Tính toán sẵn các rolling window và ewm cho tất cả tài sản trước
    precalculated = {}
    for asset in df.columns:
        prices = close[asset]
        vols = volume[asset]
        
        precalculated[asset] = {
            # Moving Averages - dùng để xác định xu hướng
            'ma20': prices.rolling(20).mean(),
            'ma50': prices.rolling(50).mean(),
            'ma100': prices.rolling(100).mean(),
            
            # Giá trị price action
            'returns': prices.pct_change(),
            'returns_5d': prices.pct_change(5),
            'returns_10d': prices.pct_change(10),
            'returns_20d': prices.pct_change(20),
            
            # Volume indicators - thêm nhiều chỉ báo về volume
            'volume': vols,
            'volume_ma20': vols.rolling(20).mean(),
            'volume_ma50': vols.rolling(50).mean(),
            'volume_change': vols.pct_change(),
            'volume_change_5d': vols.pct_change(5),
            
            # Volume Price Trend (VPT) - kết hợp volume và giá
            'vpt': (prices.pct_change() * vols).cumsum(),
            
            # On-Balance Volume (OBV)
            'price_direction': np.sign(prices.diff()),
        }
        
        # Tính OBV một cách thủ công
        obv = pd.Series(0, index=prices.index)
        for i in range(1, len(prices)):
            if prices.iloc[i] > prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + vols.iloc[i]
            elif prices.iloc[i] < prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - vols.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        precalculated[asset]['obv'] = obv
        precalculated[asset]['obv_ma20'] = obv.rolling(20).mean()
        
        # Money Flow Index (MFI)
        # Tính typical price
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        typical_price = (high + low + prices) / 3
        
        # Money flow = typical price * volume
        money_flow = typical_price * vols
        
        # Positive and negative money flow
        delta = typical_price.diff()
        positive_flow = pd.Series(0, index=typical_price.index)
        negative_flow = pd.Series(0, index=typical_price.index)
        
        positive_flow[delta > 0] = money_flow[delta > 0]
        negative_flow[delta < 0] = money_flow[delta < 0]
        
        # Calculate MFI
        positive_mf_14 = positive_flow.rolling(window=14).sum()
        negative_mf_14 = negative_flow.rolling(window=14).sum()
        
        # Tránh chia cho 0
        money_ratio = np.where(negative_mf_14 != 0, positive_mf_14 / negative_mf_14, 100)
        mfi = 100 - (100 / (1 + money_ratio))
        
        precalculated[asset]['mfi'] = pd.Series(mfi, index=prices.index)
        
        # Accumulation/Distribution Line (ADL)
        money_flow_multiplier = ((prices - low) - (high - prices)) / (high - low)
        money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], 0)
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        money_flow_volume = money_flow_multiplier * vols
        adl = money_flow_volume.cumsum()
        
        precalculated[asset]['adl'] = adl
        precalculated[asset]['adl_ma20'] = adl.rolling(20).mean()
        
        # Chaikin Oscillator
        chaikin = adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean()
        precalculated[asset]['chaikin'] = chaikin
    
    # Xử lý từng ngày trong df
    for date in df.index:
        precomputed_indicators[date] = {}
        
        # Lấy chỉ số hiện tại trong close
        current_idx = close.index.get_loc(date)
        
        # Với mỗi tài sản
        for asset in df.columns:
            # Xử lý metrics từ dữ liệu đã tính trước
            metrics = {}
            
            # Lấy giá và volume hiện tại
            current_price = close[asset].iloc[current_idx]
            current_volume = volume[asset].iloc[current_idx]
            
            # 1. Phát hiện xu hướng bằng MA
            ma20 = precalculated[asset]['ma20'].iloc[current_idx]
            ma50 = precalculated[asset]['ma50'].iloc[current_idx]
            ma100 = precalculated[asset]['ma100'].iloc[current_idx]
            
            # Tính toán độ dốc của MA để xác định strength của xu hướng
            ma20_slope = precalculated[asset]['ma20'].iloc[current_idx] / precalculated[asset]['ma20'].iloc[current_idx-10] - 1 if current_idx >= 10 else 0
            ma50_slope = precalculated[asset]['ma50'].iloc[current_idx] / precalculated[asset]['ma50'].iloc[current_idx-10] - 1 if current_idx >= 10 else 0
            
            # Xác định loại xu hướng
            uptrend = ma20 > ma50 and ma50 > ma100
            strong_uptrend = uptrend and ma20_slope > 0.01 and ma50_slope > 0.005
            sideway = abs(ma20 / ma50 - 1) < 0.02 and abs(ma50 / ma100 - 1) < 0.02
            
            metrics['trend'] = {
                'is_uptrend': float(uptrend),
                'is_strong_uptrend': float(strong_uptrend),
                'is_sideway': float(sideway),
                'price_to_ma20': current_price / ma20 - 1 if not np.isnan(ma20) and ma20 != 0 else 0,
                'ma20_to_ma50': ma20 / ma50 - 1 if not np.isnan(ma20) and not np.isnan(ma50) and ma50 != 0 else 0,
                'ma20_slope': ma20_slope,
                'ma50_slope': ma50_slope
            }
            
            # 2. Volume Analysis
            vol_ma20 = precalculated[asset]['volume_ma20'].iloc[current_idx]
            vol_ma50 = precalculated[asset]['volume_ma50'].iloc[current_idx]
            vol_change = precalculated[asset]['volume_change'].iloc[current_idx]
            vol_change_5d = precalculated[asset]['volume_change_5d'].iloc[current_idx]
            
            # Volume Breakout: Volume tăng đột biến so với trung bình
            volume_breakout = current_volume / vol_ma20 > 1.5 if not np.isnan(vol_ma20) and vol_ma20 != 0 else False
            
            # Volume Trend: Volume tăng liên tục trong 5 ngày
            volume_trend_up = vol_change_5d > 0.1 if not np.isnan(vol_change_5d) else False
            
            # Volume Confirmation: Volume tăng khi giá tăng
            returns = precalculated[asset]['returns'].iloc[current_idx]
            volume_confirms_price = (returns > 0 and vol_change > 0) if not np.isnan(returns) and not np.isnan(vol_change) else False
            
            metrics['volume'] = {
                'rel_to_avg': current_volume / vol_ma20 if not np.isnan(vol_ma20) and vol_ma20 != 0 else 1,
                'breakout': float(volume_breakout),
                'trend_up': float(volume_trend_up),
                'confirms_price': float(volume_confirms_price)
            }
            
            # 3. Price-Volume Indicators
            
            # OBV Analysis
            obv = precalculated[asset]['obv'].iloc[current_idx]
            obv_ma20 = precalculated[asset]['obv_ma20'].iloc[current_idx]
            
            # OBV Confirmation: OBV tăng khi giá tăng
            obv_prev = precalculated[asset]['obv'].iloc[current_idx-1] if current_idx > 0 else obv
            price_prev = close[asset].iloc[current_idx-1] if current_idx > 0 else current_price
            
            obv_confirms_uptrend = (obv > obv_prev and current_price > price_prev) if not np.isnan(obv) and not np.isnan(obv_prev) else False
            obv_divergence = (obv < obv_prev and current_price > price_prev) if not np.isnan(obv) and not np.isnan(obv_prev) else False
            
            # Money Flow Index
            mfi = precalculated[asset]['mfi'].iloc[current_idx]
            mfi_oversold = mfi < 20 if not np.isnan(mfi) else False
            mfi_overbought = mfi > 80 if not np.isnan(mfi) else False
            
            # A/D Line
            adl = precalculated[asset]['adl'].iloc[current_idx]
            adl_ma20 = precalculated[asset]['adl_ma20'].iloc[current_idx]
            adl_trend_up = adl > adl_ma20 if not np.isnan(adl) and not np.isnan(adl_ma20) else False
            
            # Chaikin Oscillator
            chaikin = precalculated[asset]['chaikin'].iloc[current_idx]
            chaikin_prev = precalculated[asset]['chaikin'].iloc[current_idx-1] if current_idx > 0 else 0
            chaikin_crossover = (chaikin > 0 and chaikin_prev < 0) if not np.isnan(chaikin) and not np.isnan(chaikin_prev) else False
            
            metrics['price_volume'] = {
                'obv_confirms_uptrend': float(obv_confirms_uptrend),
                'obv_divergence': float(obv_divergence),
                'obv_above_ma': float(obv > obv_ma20) if not np.isnan(obv) and not np.isnan(obv_ma20) else 0,
                'mfi': mfi if not np.isnan(mfi) else 50,
                'mfi_oversold': float(mfi_oversold),
                'mfi_overbought': float(mfi_overbought),
                'adl_trend_up': float(adl_trend_up),
                'chaikin': chaikin if not np.isnan(chaikin) else 0,
                'chaikin_crossover': float(chaikin_crossover)
            }
            
            # 4. Tín hiệu giao dịch tổng hợp
            # Tín hiệu mạnh trong uptrend
            strong_uptrend_signal = (
                metrics['trend']['is_uptrend'] > 0 and
                metrics['trend']['price_to_ma20'] > 0 and
                metrics['volume']['confirms_price'] > 0 and
                metrics['price_volume']['obv_confirms_uptrend'] > 0
            )
            
            # Tín hiệu tích lũy trong sideway
            accumulation_signal = (
                metrics['trend']['is_sideway'] > 0 and
                metrics['volume']['rel_to_avg'] > 1.2 and
                metrics['price_volume']['obv_above_ma'] > 0 and
                metrics['price_volume']['mfi'] < 40
            )
            
            # Tín hiệu breakout từ sideway sang uptrend
            breakout_signal = (
                metrics['trend']['price_to_ma20'] > 0.02 and
                metrics['volume']['breakout'] > 0 and
                metrics['price_volume']['chaikin_crossover'] > 0
            )
            
            metrics['signals'] = {
                'strong_uptrend': float(strong_uptrend_signal),
                'accumulation': float(accumulation_signal),
                'breakout': float(breakout_signal),
                # Tính điểm tổng hợp từ 0-100
                'trend_score': min(100, max(0, 
                    50 + 
                    20 * metrics['trend']['is_uptrend'] +
                    10 * metrics['trend']['ma20_slope'] * 100 +
                    10 * metrics['volume']['confirms_price'] +
                    10 * metrics['price_volume']['obv_confirms_uptrend'] -
                    10 * metrics['price_volume']['obv_divergence'] -
                    10 * (1 if metrics['price_volume']['mfi_overbought'] > 0 else 0)
                ))
            }
            
            # 5. Thêm tín hiệu cho sideway
            # Xác định vùng tích lũy (sideway) bằng Bollinger Bands
            bb_ma20 = precalculated[asset]['ma20'].iloc[current_idx]
            bb_std20 = prices.rolling(window=20).std().iloc[current_idx]
            
            if not np.isnan(bb_ma20) and not np.isnan(bb_std20):
                bb_upper = bb_ma20 + (bb_std20 * 2)
                bb_lower = bb_ma20 - (bb_std20 * 2)
                
                # Tính %B (vị trí giá trong BB)
                bb_width = (bb_upper - bb_lower) / bb_ma20 if bb_ma20 != 0 else 0
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
                
                # Xác định BB thu hẹp (vùng tích lũy)
                bb_squeeze = bb_width < 0.05
                
                # Kiểm tra xu hướng đi ngang trong vùng giá hẹp
                price_range_20d = (prices.iloc[current_idx-20:current_idx+1].max() - prices.iloc[current_idx-20:current_idx+1].min()) / bb_ma20 if current_idx >= 20 else 0
                tight_range = price_range_20d < 0.05
                
                metrics['sideway'] = {
                    'bb_width': bb_width,
                    'bb_position': bb_position,
                    'bb_squeeze': float(bb_squeeze),
                    'tight_range': float(tight_range),
                    # Tín hiệu thoát khỏi vùng tích lũy với volume tăng
                    'breakout_up': float(current_price > bb_upper and metrics['volume']['rel_to_avg'] > 1.5),
                    # Độ mạnh của tích lũy (càng cao càng tốt cho breakout)
                    'accumulation_strength': float(metrics['price_volume']['obv_above_ma'] > 0 and bb_squeeze and metrics['volume']['trend_up'] > 0)
                }
            else:
                metrics['sideway'] = {
                    'bb_width': 0,
                    'bb_position': 0.5,
                    'bb_squeeze': 0,
                    'tight_range': 0,
                    'breakout_up': 0,
                    'accumulation_strength': 0
                }
            
            # 6. Tín hiệu kết hợp Volume và Trend cuối cùng
            # Strong Uptrend + Volume Confirmation
            uptrend_vol_confirmed = (
                metrics['trend']['is_uptrend'] > 0 and
                metrics['volume']['confirms_price'] > 0 and
                metrics['price_volume']['obv_confirms_uptrend'] > 0
            )
            
            # Sideway Accumulation
            sideway_accumulation = (
                metrics['trend']['is_sideway'] > 0 and
                metrics['sideway']['accumulation_strength'] > 0 and
                metrics['price_volume']['mfi'] < 50 and
                metrics['price_volume']['mfi'] > 20
            )
            
            # Tín hiệu breakout từ sideway với volume lớn
            sideway_to_uptrend = (
                metrics['sideway']['breakout_up'] > 0 and
                metrics['volume']['rel_to_avg'] > 1.3
            )
            
            # Điểm tổng hợp cuối cùng cho chiến lược follow trend + volume
            trend_follow_score = 0
            
            if uptrend_vol_confirmed:
                trend_follow_score += 40
            
            if sideway_accumulation:
                trend_follow_score += 30
            
            if sideway_to_uptrend:
                trend_follow_score += 50
            
            # Các yếu tố bổ sung
            if metrics['price_volume']['chaikin_crossover'] > 0:
                trend_follow_score += 10
            
            if metrics['volume']['breakout'] > 0 and metrics['trend']['price_to_ma20'] > 0:
                trend_follow_score += 15
            
            if metrics['price_volume']['mfi_oversold'] > 0 and metrics['trend']['is_uptrend'] > 0:
                trend_follow_score += 20
            
            # Giới hạn điểm từ 0-100
            trend_follow_score = min(100, max(0, trend_follow_score))
            
            metrics['final_signals'] = {
                'uptrend_vol_confirmed': float(uptrend_vol_confirmed),
                'sideway_accumulation': float(sideway_accumulation),
                'sideway_to_uptrend': float(sideway_to_uptrend),
                'trend_follow_score': trend_follow_score,
                'recommendation': 'strong_buy' if trend_follow_score > 80 else 
                                 'buy' if trend_follow_score > 60 else
                                 'hold' if trend_follow_score > 40 else
                                 'neutral' if trend_follow_score > 20 else 'avoid'
            }
            
            # Lưu trữ tất cả các metrics cho tài sản này vào ngày này
            precomputed_indicators[date][asset] = metrics
    
    return precomputed_indicators

def precompute_risk_metrics(original_data, df, risk_free_rate=0.02, var_confidence_level=0.95, lookback_period=252):
    """
    Tính toán trước các chỉ số rủi ro cho mỗi tài sản và ngày.
    
    Parameters:
    -----------
    original_data : pandas.DataFrame
        DataFrame với giá lịch sử (cột là tài sản, index là ngày)
    df : pandas.DataFrame
        DataFrame với giá cho giai đoạn training/testing (tập con của original_data)
    risk_free_rate : float
        Lãi suất phi rủi ro hàng năm (mặc định: 0.02 tương đương 2%)
    var_confidence_level : float
        Mức độ tin cậy cho tính toán VaR (mặc định: 0.95)
    lookback_period : int
        Khoảng thời gian nhìn lại tối đa cần thiết cho tính toán (mặc định: 252 ngày)
    
    Returns:
    --------
    dict
        Dictionary lồng nhau với các chỉ số rủi ro được tính toán trước:
        {date: {asset: {risk_metric_name: value}}}
    """
    # Precompute market returns once
    market_prices = original_data.mean(axis=1)
    market_returns = market_prices.pct_change().values
    
    # Precompute indices to avoid repeated lookups
    original_index_dict = {date: idx for idx, date in enumerate(original_data.index)}
    
    # Precompute returns for all assets to avoid repeated calculations
    asset_returns = {}
    for asset in original_data.columns:
        asset_returns[asset] = original_data[asset].pct_change().values
    
    # Optimize result dictionary creation
    precomputed_risk_metrics = {}
    
    for date in df.index:
        # Get current index in original data
        current_idx = original_index_dict[date]
        start_idx = max(0, current_idx - lookback_period)
        
        # Slice market returns for this period
        market_returns_period = market_returns[start_idx:current_idx + 1]
        market_returns_period = market_returns_period[~np.isnan(market_returns_period)]
        
        # Prepare result for this date
        date_metrics = {}
        
        for asset in df.columns:
            # Get asset returns for this period
            returns = asset_returns[asset][start_idx:current_idx + 1]
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                continue
            
            # Compute metrics using NumPy for speed
            metrics = {}
            
            # Annualized metrics
            metrics['mean_return'] = np.mean(returns) * 252
            metrics['volatility'] = np.std(returns) * np.sqrt(252)
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, (1 - var_confidence_level) * 100)
            metrics['var_95'] = var_95
            
            # Conditional Value at Risk (CVaR/Expected Shortfall)
            below_var = returns[returns <= var_95]
            metrics['cvar_95'] = np.mean(below_var) if len(below_var) > 0 else var_95
            
            # Maximum Drawdown (faster computation)
            prices = original_data[asset].values[start_idx:current_idx + 1]
            rolling_max = np.maximum.accumulate(prices)
            drawdowns = (prices - rolling_max) / rolling_max
            metrics['max_drawdown'] = np.min(drawdowns)
            
            # Sharpe Ratio
            excess_returns = metrics['mean_return'] - risk_free_rate
            metrics['sharpe_ratio'] = excess_returns / metrics['volatility'] if metrics['volatility'] != 0 else 0
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else metrics['volatility']
            metrics['sortino_ratio'] = excess_returns / downside_std if downside_std != 0 else 0
            
            # Beta calculation
            if len(returns) > 1 and len(market_returns_period) > 1:
                try:
                    # Use NumPy for covariance calculation
                    min_length = min(len(returns), len(market_returns_period))
                    returns_slice = returns[-min_length:]
                    market_returns_slice = market_returns_period[-min_length:]
                    
                    covariance = np.cov(returns_slice, market_returns_slice)[0, 1]
                    market_variance = np.var(market_returns_slice)
                    metrics['beta'] = covariance / market_variance if market_variance != 0 else 1
                except Exception:
                    metrics['beta'] = 1
            else:
                metrics['beta'] = 1
            
            # Skewness and Kurtosis
            try:
                metrics['skewness'] = np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3)
                metrics['kurtosis'] = np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4) - 3
            except Exception:
                metrics['skewness'] = 0
                metrics['kurtosis'] = 0
            
            # Information Ratio
            if len(returns) > 1 and len(market_returns_period) > 1:
                try:
                    min_length = min(len(returns), len(market_returns_period))
                    returns_slice = returns[-min_length:]
                    market_returns_slice = market_returns_period[-min_length:]
                    
                    tracking_error = np.std(returns_slice - market_returns_slice) * np.sqrt(252)
                    metrics['tracking_error'] = tracking_error
                    
                    metrics['info_ratio'] = (
                        (np.mean(returns_slice) * 252 - np.mean(market_returns_slice) * 252) / 
                        tracking_error if tracking_error != 0 else 0
                    )
                except Exception:
                    metrics['tracking_error'] = 0
                    metrics['info_ratio'] = 0
            else:
                metrics['tracking_error'] = 0
                metrics['info_ratio'] = 0
            
            date_metrics[asset] = metrics
        
        precomputed_risk_metrics[date] = date_metrics
    
    return precomputed_risk_metrics

def precompute_market_conditions(market_index, df, market_memory=252, regime_lookback=126, 
                                vol_regime_threshold=0.2, trend_threshold=0.05,
                                volatility_threshold=0.2, stress_threshold=-0.15):
    """
    Tính toán trước các chỉ số môi trường thị trường cho mỗi ngày.
    
    Parameters:
    -----------
    market_index : pandas.DataFrame
        DataFrame với dữ liệu chỉ số thị trường (cần có cột 'close', 'high', 'low', 'volume')
    df : pandas.DataFrame
        DataFrame với giá cho giai đoạn training/testing
    
    Returns:
    --------
    dict
        Dictionary với các chỉ số môi trường thị trường được tính toán trước:
        {date: {metric_name: value}}
    """
    # Khởi tạo dictionary kết quả
    precomputed_market_conditions = {}
    
    # Xử lý từng ngày trong df
    for date in df.index:
        # Tìm chỉ số trong market_index
        if date not in market_index.index:
            # Bỏ qua nếu ngày không có trong dữ liệu thị trường
            continue
            
        current_idx = market_index.index.get_loc(date)
        start_idx = max(0, current_idx - market_memory)
        
        # Lấy dữ liệu thị trường
        market_prices = market_index['close'].iloc[start_idx:current_idx + 1]
        market_high = market_index['high'].iloc[start_idx:current_idx + 1]
        market_low = market_index['low'].iloc[start_idx:current_idx + 1]
        market_volume = market_index['volume'].iloc[start_idx:current_idx + 1]
        market_returns = market_prices.pct_change().dropna()
        
        metrics = {}
        
        # Tính MFI (Money Flow Index)
        typical_price = (market_high + market_low + market_prices) / 3
        money_flow = typical_price * market_volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        
        # Xử lý trường hợp negative_mf = 0
        mfi_divisor = negative_mf.replace(0, np.nan)
        mfi = 100 - (100 / (1 + (positive_mf / mfi_divisor)))
        metrics['mfi'] = mfi.iloc[-1] if not np.isnan(mfi.iloc[-1]) else 50
        
        # Phân tích khối lượng
        vol_ma = market_volume.rolling(window=20).mean()
        price_ma = market_prices.rolling(window=20).mean()
        
        current_vol = market_volume.iloc[-1]
        avg_vol = vol_ma.iloc[-1]
        vol_ratio = current_vol / avg_vol if not np.isnan(avg_vol) and avg_vol != 0 else 1
        
        price_change = market_prices.pct_change().iloc[-1] if len(market_prices) > 1 else 0
        price_trend = (market_prices.iloc[-1] / price_ma.iloc[-1] - 1) if not np.isnan(price_ma.iloc[-1]) and price_ma.iloc[-1] != 0 else 0
        
        metrics['volume_strength'] = vol_ratio * (1 + price_change)
        metrics['buying_pressure'] = 0
        
        # Phát hiện áp lực mua
        if price_trend < -0.05:
            if vol_ratio > 1.5 and price_change > -0.01:
                metrics['buying_pressure'] += 0.3
            elif vol_ratio < 0.8:
                metrics['buying_pressure'] += 0.2
        
        # Phân tích xu hướng
        sma_50 = market_prices.rolling(50).mean().iloc[-1] if len(market_prices) >= 50 else np.nan
        sma_200 = market_prices.rolling(200).mean().iloc[-1] if len(market_prices) >= 200 else np.nan
        current_price = market_prices.iloc[-1]
        
        metrics['trend_signal'] = 1 if not np.isnan(sma_200) and current_price > sma_200 else -1
        metrics['trend_strength'] = (current_price / sma_200 - 1) if not np.isnan(sma_200) and sma_200 != 0 else 0
        
        # Phân tích biến động
        current_vol_value = market_returns.std() * np.sqrt(252) if len(market_returns) > 1 else 0
        metrics['volatility'] = current_vol_value
        metrics['vol_regime'] = 1 if current_vol_value > vol_regime_threshold else 0
        
        # Stress thị trường
        rolling_max = market_prices.rolling(min(market_memory, len(market_prices))).max()
        drawdown = (market_prices - rolling_max) / rolling_max
        metrics['market_stress'] = drawdown.iloc[-1] if not np.isnan(drawdown.iloc[-1]) else 0
        
        # RSI
        if len(market_returns) >= 14:
            delta = market_returns
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gains = gains.rolling(14).mean()
            avg_losses = losses.rolling(14).mean()
            
            # Xử lý trường hợp avg_losses = 0
            rs_divisor = avg_losses.replace(0, np.nan)
            rs = avg_gains / rs_divisor
            rsi_value = (100 - (100 / (1 + rs))).iloc[-1]
            metrics['rsi'] = rsi_value if not np.isnan(rsi_value) else 50
        else:
            metrics['rsi'] = 50  # Giá trị mặc định nếu không đủ dữ liệu
        
        # Phân tích chế độ thị trường
        regime_lookback_actual = min(regime_lookback, len(market_returns))
        if regime_lookback_actual > 0:
            regime_returns = market_returns.iloc[-regime_lookback_actual:]
            regime_trend = regime_returns.mean() * 252
            regime_vol = regime_returns.std() * np.sqrt(252)
            
            # Xác định chế độ
            if regime_trend > trend_threshold and regime_vol < vol_regime_threshold:
                if metrics['volume_strength'] > 0 and metrics['mfi'] > 40:
                    metrics['regime'] = 'strong_bullish'
                else:
                    metrics['regime'] = 'bullish'
            elif regime_trend < -trend_threshold and regime_vol > vol_regime_threshold:
                if metrics['buying_pressure'] > 0.2:
                    metrics['regime'] = 'weakening_bearish'
                else:
                    metrics['regime'] = 'bearish'
            else:
                metrics['regime'] = 'neutral'
        else:
            metrics['regime'] = 'neutral'  # Giá trị mặc định nếu không đủ dữ liệu
        
        # Phát hiện khủng hoảng
        crisis_indicators = 0
        if current_vol_value > volatility_threshold:
            crisis_indicators += 1
        if metrics['market_stress'] < stress_threshold:
            if metrics['buying_pressure'] > 0.2:
                crisis_indicators += 0.5
            else:
                crisis_indicators += 1
        if 'regime_vol' in locals() and regime_vol > vol_regime_threshold * 1.5:
            crisis_indicators += 1
        
        metrics['crisis_score'] = crisis_indicators / 3
        
        # Lưu trữ metrics cho ngày này
        precomputed_market_conditions[date] = metrics
    
    return precomputed_market_conditions

def precompute_weinstein_stage_analysis(close_week, high_week, low_week, volume_week, df, lookback_period=252):
    """
    Tính toán phân tích giai đoạn theo phương pháp Stan Weinstein cho mỗi tài sản.
    
    Parameters:
    -----------
    close_week, high_week, low_week, volume_week : pandas.DataFrame
        DataFrame với giá lịch sử theo tuần (cột là tài sản, index là ngày)
    df : pandas.DataFrame
        DataFrame với giá theo ngày cho giai đoạn training/testing
    lookback_period : int
        Khoảng thời gian nhìn lại tối đa cần thiết cho tính toán (mặc định: 252 ngày)
    
    Returns:
    --------
    dict
        Dictionary lồng nhau với các chỉ báo stage analysis được ánh xạ về ngày:
        {date: {asset: {indicator_name: value}}}
    """
    # Khởi tạo dictionary kết quả
    precomputed_indicators = {}
    
    # Tính toán sẵn các chỉ báo cho tất cả tài sản trước (theo tuần)
    precalculated_weekly = {}
    
    # Tạo một dataframe trống để lưu giai đoạn theo Weinstein cho từng tài sản theo tuần
    assets = df.columns
    stages_weekly = pd.DataFrame(index=close_week.index, columns=assets)
    
    # Tạo ánh xạ từ ngày sang tuần
    # Giả định rằng index của close_week là ngày cuối cùng của mỗi tuần (thường là thứ 6)
    # Tạo ánh xạ từ ngày sang tuần (cải tiến)
    weekly_to_daily_map = {}
    # Chuyển đổi index thành list để dễ dàng làm việc
    weekly_dates = close_week.index.tolist()
    weekly_dates.sort()  # Đảm bảo đã sắp xếp

    # Với mỗi ngày trong df
    for daily_date in df.index:
        # Tìm ngày kết thúc tuần gần nhất (ngày tuần nhỏ nhất mà >= daily_date)
        current_week = None
        for week_date in weekly_dates:
            if daily_date <= week_date:
                current_week = week_date
                break
        
        # Nếu không tìm thấy (ngày nằm sau tất cả các tuần), sử dụng tuần cuối cùng
        if current_week is None and weekly_dates:
            current_week = weekly_dates[-1]
        
        # Lưu ánh xạ
        if current_week is not None:
            weekly_to_daily_map[daily_date] = current_week
    
    for asset in assets:
        # 1. Tính toán 30-week Moving Average (Weinstein sử dụng cái này rất nhiều)
        ma30 = close_week[asset].rolling(30).mean()
        
        # 2. Tính toán slope (độ dốc) của 30-week MA
        ma30_slope = ma30.diff(4) / ma30.shift(4)  # Độ dốc trong 4 tuần
        
        # 3. Tính Volume Moving Average
        vol_ma10 = volume_week[asset].rolling(10).mean()
        
        # 4. Tính Relative Strength so với một benchmark
        # Thông thường Weinstein so sánh với S&P 500 nhưng ở đây chúng ta sẽ tạo một RS trung bình
        # bằng cách so sánh với trung bình của tất cả các tài sản
        market_avg = close_week.mean(axis=1)
        rs = (close_week[asset] / close_week[asset].shift(13)) / (market_avg / market_avg.shift(13))
        rs_ma = rs.rolling(10).mean()  # Làm mượt RS
        
        # 5. Tính trước các phạm vi giá cao/thấp trong 10 tuần
        high_10week = high_week[asset].rolling(10).max()
        low_10week = low_week[asset].rolling(10).min()
        
        # 6. Lưu tất cả chỉ báo hàng tuần đã tính
        precalculated_weekly[asset] = {
            'ma30': ma30,
            'ma30_slope': ma30_slope,
            'vol_ma10': vol_ma10,
            'rs': rs,
            'rs_ma': rs_ma,
            'high_10week': high_10week,
            'low_10week': low_10week
        }
        
        # 7. Xác định các giai đoạn Weinstein cho từng tuần
        for i in range(30, len(close_week)):
            week_date = close_week.index[i]
            
            # Lấy giá trị hiện tại và trước đó
            price = close_week[asset].iloc[i]
            price_prev = close_week[asset].iloc[i-1]
            ma30_val = ma30.iloc[i]
            ma30_prev = ma30.iloc[i-1] if i > 0 else float('nan')
            ma30_slope_val = ma30_slope.iloc[i]
            volume = volume_week[asset].iloc[i]
            vol_avg = vol_ma10.iloc[i]
            rs_val = rs_ma.iloc[i]
            rs_prev = rs_ma.iloc[i-1] if i > 0 else float('nan')
            
            # Các biến để lưu trữ đặc điểm của từng giai đoạn
            stage = 0  # 0 = không xác định, 1-4 = giai đoạn theo Weinstein
            
            # Giai đoạn 1: Basing/Tích lũy
            # - Giá dao động ngang
            # - MA30 đi ngang hoặc có xu hướng nhẹ xuống
            # - Giá thường dưới MA30
            if (abs(ma30_slope_val) < 0.01 or ma30_slope_val < 0) and (abs(price/ma30_val - 1) < 0.03):
                stage = 1
                
                # Kiểm tra dấu hiệu sắp kết thúc giai đoạn 1 (chuẩn bị chuyển sang giai đoạn 2)
                # - Tăng khối lượng
                # - Giá vượt lên trên MA30
                if volume > vol_avg*1.3 and price > ma30_val and price > price_prev:
                    stage = 1.5  # Giai đoạn 1 với dấu hiệu sắp chuyển sang giai đoạn 2
            
            # Giai đoạn 2: Advancing/Markup
            # - Giá trên MA30
            # - MA30 đang đi lên
            # - Relative Strength tăng
            elif price > ma30_val and ma30_slope_val > 0 and rs_val > rs_prev:
                stage = 2
                
                # Kiểm tra dấu hiệu sắp kết thúc giai đoạn 2 (chuẩn bị chuyển sang giai đoạn 3)
                # - RS bắt đầu suy yếu
                # - Khối lượng giảm
                if rs_val < rs_prev and volume < vol_avg*0.8:
                    stage = 2.5  # Giai đoạn 2 với dấu hiệu sắp chuyển sang giai đoạn 3
            
            # Giai đoạn 3: Top/Distribution
            # - Giá dao động ngang ở mức cao
            # - MA30 bắt đầu đi ngang
            # - Relative Strength suy yếu
            elif (abs(ma30_slope_val) < 0.01 or ma30_slope_val < 0) and price > ma30_val * 0.98 and rs_val < rs_prev:
                stage = 3
                
                # Kiểm tra dấu hiệu sắp kết thúc giai đoạn 3 (chuẩn bị chuyển sang giai đoạn 4)
                # - Giá phá xuống dưới MA30
                # - Tăng khối lượng khi giá giảm
                if price < ma30_val and volume > vol_avg and price < price_prev:
                    stage = 3.5  # Giai đoạn 3 với dấu hiệu sắp chuyển sang giai đoạn 4
            
            # Giai đoạn 4: Declining/Markdown
            # - Giá dưới MA30
            # - MA30 đang đi xuống
            # - Relative Strength yếu
            elif price < ma30_val and ma30_slope_val < 0 and rs_val < 1:
                stage = 4
                
                # Kiểm tra dấu hiệu sắp kết thúc giai đoạn 4 (chuẩn bị chuyển sang giai đoạn 1)
                # - Giá bắt đầu dao động ngang
                # - Volume bắt đầu cạn kiệt
                if abs(price/price_prev - 1) < 0.01 and volume < vol_avg*0.7:
                    stage = 4.5  # Giai đoạn 4 với dấu hiệu sắp chuyển sang giai đoạn 1
            
            # Lưu giai đoạn vào dataframe
            stages_weekly.loc[week_date, asset] = stage
    
    # Bây giờ, ánh xạ các chỉ báo từ tuần sang ngày và tạo dictionary kết quả
    last_valid_stages = {}
    last_valid_stage_indicators = {}
    for date in df.index:
        precomputed_indicators[date] = {}
        
        # Tìm ngày cuối tuần tương ứng
        if date not in weekly_to_daily_map:
            continue
            
        week_date = weekly_to_daily_map[date]
        if week_date not in close_week.index:
            continue
            
        week_idx = close_week.index.get_loc(week_date)
        
        # Với mỗi tài sản
        for asset in assets:
            # Khởi tạo dictionary cho tài sản
            metrics = {}
            
            # Lấy giá hiện tại và giá tuần
            current_price = df[asset].loc[date]
            weekly_price = close_week[asset].loc[week_date]
            
            # Lấy chỉ báo Weinstein từ dữ liệu đã tính theo tuần
            if week_idx >= 30:  # Đảm bảo có đủ dữ liệu
                ma30_val = precalculated_weekly[asset]['ma30'].iloc[week_idx]
                ma30_slope_val = precalculated_weekly[asset]['ma30_slope'].iloc[week_idx]
                rs_val = precalculated_weekly[asset]['rs_ma'].iloc[week_idx]
                volume = volume_week[asset].loc[week_date]
                vol_avg = precalculated_weekly[asset]['vol_ma10'].iloc[week_idx]
                high_10w = precalculated_weekly[asset]['high_10week'].iloc[week_idx]
                low_10w = precalculated_weekly[asset]['low_10week'].iloc[week_idx]
                
                # Lấy stage từ dataframe đã tính toán
                stage = stages_weekly.loc[week_date, asset]
                
                # Tính các chỉ báo Weinstein
                stage_indicators = {
                    'current_stage': stage,
                    'stage_1_base': float(1 <= stage < 2),  # Giai đoạn 1: Tích lũy
                    'stage_2_advance': float(2 <= stage < 3),  # Giai đoạn 2: Tăng giá
                    'stage_3_top': float(3 <= stage < 4),  # Giai đoạn 3: Đỉnh
                    'stage_4_decline': float(4 <= stage <= 4.5),  # Giai đoạn 4: Giảm giá
                    'stage_1_to_2': float(stage == 1.5),  # Chuẩn bị chuyển từ 1 sang 2
                    'stage_2_to_3': float(stage == 2.5),  # Chuẩn bị chuyển từ 2 sang 3
                    'stage_3_to_4': float(stage == 3.5),  # Chuẩn bị chuyển từ 3 sang 4
                    'stage_4_to_1': float(stage == 4.5),  # Chuẩn bị chuyển từ 4 sang 1
                }
                
                # Nếu current_stage là 0, dùng giá trị từ ngày trước
                if stage_indicators['current_stage'] == 0 and asset in last_valid_stages:
                    stage_indicators['current_stage'] = last_valid_stages[asset]
                    stage_indicators['stage_1_base'] = last_valid_stage_indicators[asset]['stage_1_base']
                    stage_indicators['stage_2_advance'] = last_valid_stage_indicators[asset]['stage_2_advance']
                    stage_indicators['stage_3_top'] = last_valid_stage_indicators[asset]['stage_3_top']
                    stage_indicators['stage_4_decline'] = last_valid_stage_indicators[asset]['stage_4_decline']
                    # Không thay đổi các chỉ báo stage_X_to_Y và các chỉ báo khác
                # Lưu giá trị hợp lệ hiện tại nếu stage khác 0
                elif stage_indicators['current_stage'] != 0:
                    last_valid_stages[asset] = stage_indicators['current_stage']
                    if asset not in last_valid_stage_indicators:
                        last_valid_stage_indicators[asset] = {}
                    last_valid_stage_indicators[asset]['stage_1_base'] = stage_indicators['stage_1_base']
                    last_valid_stage_indicators[asset]['stage_2_advance'] = stage_indicators['stage_2_advance']
                    last_valid_stage_indicators[asset]['stage_3_top'] = stage_indicators['stage_3_top']
                    last_valid_stage_indicators[asset]['stage_4_decline'] = stage_indicators['stage_4_decline']
                
                metrics['weinstein_stage'] = stage_indicators
                
                # Tính các metrics từ MA30 (30-week moving average)
                metrics['ma30'] = {
                    'value': ma30_val,
                    'slope': ma30_slope_val,
                    'price_to_ma30': current_price / ma30_val - 1 if not np.isnan(ma30_val) else 0,
                    'above_ma30': float(current_price > ma30_val) if not np.isnan(ma30_val) else 0,
                    'ma30_rising': float(ma30_slope_val > 0) if not np.isnan(ma30_slope_val) else 0,
                    'ma30_flat': float(abs(ma30_slope_val) < 0.01) if not np.isnan(ma30_slope_val) else 0,
                }
                
                # Tính metrics về Relative Strength
                metrics['relative_strength'] = {
                    'value': rs_val,
                    'rising': float(rs_val > 1) if not np.isnan(rs_val) else 0,
                    'strong': float(rs_val > 1.05) if not np.isnan(rs_val) else 0,
                    'weak': float(rs_val < 0.95) if not np.isnan(rs_val) else 0,
                }
                
                # Tính metrics về Volume
                metrics['volume'] = {
                    'vs_average': volume / vol_avg if not np.isnan(vol_avg) and vol_avg != 0 else 1,
                    'high_volume': float(volume > vol_avg * 1.3) if not np.isnan(vol_avg) and vol_avg != 0 else 0,
                    'low_volume': float(volume < vol_avg * 0.7) if not np.isnan(vol_avg) and vol_avg != 0 else 0,
                    'declining_volume': float(volume < vol_avg) if not np.isnan(vol_avg) and vol_avg != 0 else 0,
                }
                
                # Tính metrics về Support/Resistance và Price Patterns (theo phương pháp Weinstein)
                metrics['price_patterns'] = {
                    'near_resistance': float(current_price > high_10w * 0.95) if not np.isnan(high_10w) else 0,
                    'near_support': float(current_price < low_10w * 1.05) if not np.isnan(low_10w) else 0,
                    'breakout_potential': float(current_price > high_10w * 0.97 and volume > vol_avg * 1.2) if not np.isnan(high_10w) and not np.isnan(vol_avg) and vol_avg != 0 else 0,
                    'breakdown_potential': float(current_price < low_10w * 1.03 and volume > vol_avg * 1.2) if not np.isnan(low_10w) and not np.isnan(vol_avg) and vol_avg != 0 else 0,
                }
                
                # Tính các tín hiệu đặc biệt theo phương pháp Weinstein
                metrics['weinstein_signals'] = {
                    'buying_signal': float(stage == 1.5 or (stage == 2 and current_price > ma30_val and rs_val > 1.05)),
                    'selling_signal': float(stage == 3.5 or (stage == 4 and current_price < ma30_val and rs_val < 0.95)),
                    'hold_signal': float(stage == 2 and current_price > ma30_val and rs_val > 1),
                    # Tín hiệu Buy-Watch-Sell theo Weinstein
                    'buy': float(stage == 1.5 or stage == 2),
                    'watch': float(stage == 2.5 or stage == 3 or stage == 4.5),
                    'sell': float(stage == 3.5 or stage == 4),
                }
            else:
                # Nếu không đủ dữ liệu, trả về các giá trị mặc định
                metrics['weinstein_stage'] = {
                    'current_stage': 0,  # Không đủ dữ liệu để xác định
                    'stage_1_base': 0, 
                    'stage_2_advance': 0,
                    'stage_3_top': 0,
                    'stage_4_decline': 0,
                    'stage_1_to_2': 0,
                    'stage_2_to_3': 0,
                    'stage_3_to_4': 0,
                    'stage_4_to_1': 0,
                }
                metrics['ma30'] = {'value': 0, 'slope': 0, 'price_to_ma30': 0, 'above_ma30': 0, 'ma30_rising': 0, 'ma30_flat': 0}
                metrics['relative_strength'] = {'value': 1, 'rising': 0, 'strong': 0, 'weak': 0}
                metrics['volume'] = {'vs_average': 1, 'high_volume': 0, 'low_volume': 0, 'declining_volume': 0}
                metrics['price_patterns'] = {'near_resistance': 0, 'near_support': 0, 'breakout_potential': 0, 'breakdown_potential': 0}
                metrics['weinstein_signals'] = {'buying_signal': 0, 'selling_signal': 0, 'hold_signal': 0, 'buy': 0, 'watch': 0, 'sell': 0}
            
            # Lưu trữ tất cả các metrics cho tài sản này vào ngày này
            precomputed_indicators[date][asset] = metrics
    
    return precomputed_indicators












