import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TechnicalEnv(gym.Env):
    def __init__(self, df, precomputed_indicators, n_assets=5, initial_assets=None, initial_balance=10_000_000, 
                 reweight=20, min_pctweight=0.05, limit=0.1, rebalance_window=40, 
                 num_assets_change=2, transaction_cost=0.01):
        super(TechnicalEnv, self).__init__()
        
        # Store parameters
        self.df = df
        self.n_assets = len(initial_assets) if initial_assets is not None else n_assets
        self.initial_assets = initial_assets
        self.initial_balance = initial_balance
        self.reweight = reweight
        self.min_pctweight = min_pctweight
        self.limit = limit
        self.rebalance_window = rebalance_window
        self.num_assets_change = num_assets_change
        self.transaction_cost = transaction_cost
        self.days_since_rebalance = 0
        self.pending_reweight = False
        self.pending_reweight_days = 0
        self.precomputed_indicators = precomputed_indicators
        self.all_assets = df.columns.tolist()
        
        assert all(asset in df.columns for asset in self.all_assets)
        
        # Set up observation space for n_features_per_asset (12) + weights
        n_features_per_asset = 13
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_assets * n_features_per_asset + self.n_assets,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
    def get_technical_indicators(self, asset, date):
        """
        Lấy các chỉ báo kỹ thuật đã được tính trước cho một tài sản tại một ngày cụ thể.
        Tận dụng các chỉ báo xu hướng + volume mới.
        """
        if date in self.precomputed_indicators and asset in self.precomputed_indicators[date]:
            return self.precomputed_indicators[date][asset]
        else:
            # Trả về giá trị mặc định nếu dữ liệu không có sẵn
            return {
                'trend': {
                    'is_uptrend': 0,
                    'is_strong_uptrend': 0,
                    'is_sideway': 0,
                    'price_to_ma20': 0,
                    'ma20_to_ma50': 0,
                    'ma20_slope': 0,
                    'ma50_slope': 0
                },
                'volume': {
                    'rel_to_avg': 1,
                    'breakout': 0,
                    'trend_up': 0,
                    'confirms_price': 0
                },
                'price_volume': {
                    'obv_confirms_uptrend': 0,
                    'obv_divergence': 0,
                    'obv_above_ma': 0,
                    'mfi': 50,
                    'mfi_oversold': 0,
                    'mfi_overbought': 0,
                    'adl_trend_up': 0,
                    'chaikin': 0,
                    'chaikin_crossover': 0
                },
                'signals': {
                    'strong_uptrend': 0,
                    'accumulation': 0,
                    'breakout': 0,
                    'trend_score': 50
                },
                'sideway': {
                    'bb_width': 0,
                    'bb_position': 0.5,
                    'bb_squeeze': 0,
                    'tight_range': 0,
                    'breakout_up': 0,
                    'accumulation_strength': 0
                },
                'final_signals': {
                    'uptrend_vol_confirmed': 0,
                    'sideway_accumulation': 0,
                    'sideway_to_uptrend': 0,
                    'trend_follow_score': 50,
                    'recommendation': 'neutral'
                }
            }

    def _calculate_scores(self, metrics):
        """
        Tính toán điểm đánh giá kỹ thuật cho một tài sản dựa trên các chỉ báo.
        Tập trung vào hệ thống điểm mạnh hơn cho uptrend và accumulation.
        """
        # Khởi tạo các điểm thành phần
        trend_score = 0
        vol_strength_score = 0
        accumulation_score = 0
        momentum_score = 0
        reversal_risk_score = 0  # Điểm cao = rủi ro đảo chiều cao
        
        # 1. Phân tích xu hướng - tận dụng các chỉ báo xu hướng mới
        if metrics['trend']['is_strong_uptrend'] > 0:
            trend_score += 40  # Xu hướng tăng mạnh được đánh giá cao nhất
        elif metrics['trend']['is_uptrend'] > 0:
            trend_score += 25  # Xu hướng tăng bình thường
        
        # Độ dốc của MA cho thấy tốc độ tăng
        trend_score += metrics['trend']['ma20_slope'] * 300  # Ma20 slope thường là giá trị nhỏ (0.01 = 1%)
        trend_score += metrics['trend']['ma50_slope'] * 200
        
        # Vị trí giá so với MA20 (không quá cao và không quá thấp)
        price_ma20_diff = metrics['trend']['price_to_ma20']
        if 0 < price_ma20_diff < 0.03:  # Giá trên MA20 nhưng không quá xa
            trend_score += 10
        elif price_ma20_diff >= 0.03:  # Giá đã tăng khá nhiều so với MA20
            trend_score -= price_ma20_diff * 100  # Trừ điểm nếu giá đã tăng quá nhiều
        
        # 2. Phân tích sức mạnh volume
        if metrics['volume']['confirms_price'] > 0:
            vol_strength_score += 15  # Volume tăng khi giá tăng
            
        # Volume breakout là tín hiệu tốt
        if metrics['volume']['breakout'] > 0:
            vol_strength_score += 15
        
        # Volume cao hơn trung bình
        vol_rel = metrics['volume']['rel_to_avg']
        if vol_rel > 1.2:
            vol_strength_score += min((vol_rel - 1) * 50, 15)  # Tối đa 15 điểm
        
        # 3. Phân tích tích lũy (accumulation) - rất quan trọng để mua ở đáy
        if metrics['final_signals']['sideway_accumulation'] > 0:
            accumulation_score += 30  # Sideway với tích lũy là tín hiệu tốt để mua
        
        if metrics['sideway']['accumulation_strength'] > 0:
            accumulation_score += 20
        
        # MFI oversold trong uptrend là cơ hội mua tốt
        if metrics['price_volume']['mfi_oversold'] > 0 and metrics['trend']['is_uptrend'] > 0:
            accumulation_score += 25
        
        # Breakout từ vùng sideway
        if metrics['final_signals']['sideway_to_uptrend'] > 0:
            accumulation_score += 30
            trend_score += 15  # Cộng thêm cho trend
        
        # 4. Phân tích momentum
        if metrics['price_volume']['chaikin_crossover'] > 0:
            momentum_score += 15
        
        if metrics['price_volume']['obv_confirms_uptrend'] > 0:
            momentum_score += 15
        elif metrics['price_volume']['obv_divergence'] > 0:
            momentum_score -= 20  # Phân kỳ âm là dấu hiệu xấu
        
        # ADL xu hướng tăng
        if metrics['price_volume']['adl_trend_up'] > 0:
            momentum_score += 10
        
        # 5. Phân tích nguy cơ đảo chiều
        if metrics['price_volume']['mfi_overbought'] > 0:
            reversal_risk_score += 30  # MFI quá mua là dấu hiệu sắp đảo chiều
        
        if metrics['trend']['is_strong_uptrend'] > 0 and metrics['price_volume']['obv_divergence'] > 0:
            reversal_risk_score += 30  # Phân kỳ âm trong xu hướng tăng mạnh
        
        # Khi giá đã tăng quá xa MA
        if metrics['trend']['price_to_ma20'] > 0.05:
            reversal_risk_score += min(metrics['trend']['price_to_ma20'] * 200, 25)
        
        # Xác định loại thị trường dựa trên các điểm thành phần
        combined_score = (trend_score * 0.35) + (momentum_score * 0.25) + \
                        (vol_strength_score * 0.15) + (accumulation_score * 0.25) - \
                        (reversal_risk_score * 0.5)  # Trừ điểm nếu có nguy cơ đảo chiều
        
        # Xác định giai đoạn thị trường
        if trend_score > 20 and momentum_score > 10 and reversal_risk_score < 20:
            market_phase = "strong_uptrend"  # Xu hướng tăng mạnh
        elif accumulation_score > 30 and metrics['trend']['is_sideway'] > 0:
            market_phase = "accumulation"  # Giai đoạn tích lũy - tốt để mua
        elif trend_score > 10 and reversal_risk_score < 15:
            market_phase = "moderate_uptrend"  # Xu hướng tăng vừa phải
        elif reversal_risk_score > 25:
            market_phase = "distribution"  # Giai đoạn phân phối - nên bán
        else:
            market_phase = "neutral"  # Chưa xác định rõ
        
        # Chuẩn hóa điểm về thang 0-100
        normalized_score = max(0, min(100, 50 + combined_score))
        
        # Kết hợp thêm trend_follow_score từ final_signals
        final_technical_score = (normalized_score * 0.7) + (metrics['final_signals']['trend_follow_score'] * 0.3)
        
        return {
            'technical_score': final_technical_score,
            'market_phase': market_phase,
            'analysis': {
                'trend': trend_score, 
                'momentum': momentum_score,
                'volume_strength': vol_strength_score, 
                'accumulation': accumulation_score,
                'reversal_risk': reversal_risk_score
            },
            'buy_signal': accumulation_score > 25 or (trend_score > 15 and momentum_score > 10),
            'sell_signal': reversal_risk_score > 30 or 
                        (metrics['price_volume']['mfi_overbought'] > 0 and metrics['price_volume']['obv_divergence'] > 0),
            'metrics': metrics
        }
    
    def _evaluate_all_assets(self):
        """Evaluate all assets using technical indicators."""
        all_metrics = {}
        current_date = self.df.index[self.current_step]
        
        for asset in self.all_assets:
            metrics = self.get_technical_indicators(asset, current_date)
            all_metrics[asset] = self._calculate_scores(metrics)
        
        return all_metrics
    
    def _evaluate_selected_assets(self):
        """Evaluate only the selected assets in portfolio."""
        evaluation = {}
        current_date = self.df.index[self.current_step]
        
        for asset in self.selected_assets:
            metrics = self.get_technical_indicators(asset, current_date)
            evaluation[asset] = self._calculate_scores(metrics)
        
        return evaluation

    def _rebalance_portfolio(self):
        """
        Thay đổi tài sản trong danh mục dựa trên hiệu suất kỹ thuật.
        Áp dụng chiến lược mua ở giai đoạn tích lũy, giữ trong uptrend, bán ở đỉnh.
        """
        # Đánh giá tất cả các tài sản
        all_metrics = self._evaluate_all_assets()
        
        # Phân loại tài sản theo giai đoạn thị trường
        market_phases = {
            'strong_uptrend': [],
            'moderate_uptrend': [],
            'accumulation': [],
            'distribution': [],
            'neutral': []
        }
        
        for asset, data in all_metrics.items():
            phase = data['market_phase']
            score = data['technical_score']
            market_phases[phase].append((asset, score))
        
        # Sắp xếp theo điểm trong từng giai đoạn
        for phase in market_phases:
            market_phases[phase].sort(key=lambda x: x[1], reverse=True)
        
        # Tìm tài sản tệ nhất trong danh mục hiện tại
        portfolio_performance = {
            asset: {
                'score': all_metrics[asset]['technical_score'],
                'phase': all_metrics[asset]['market_phase'],
                'sell_signal': all_metrics[asset]['sell_signal']
            } for asset in self.selected_assets
        }
        
        # Ưu tiên bán những tài sản có tín hiệu bán hoặc trong giai đoạn phân phối
        worst_performers = []
        
        # 1. Trước tiên, bán tài sản có tín hiệu bán rõ ràng
        sell_candidates = [(asset, data) for asset, data in portfolio_performance.items() if data['sell_signal']]
        sell_candidates.sort(key=lambda x: x[1]['score'])
        worst_performers.extend([asset for asset, _ in sell_candidates[:self.num_assets_change]])
        
        # 2. Nếu chưa đủ, bán tài sản trong giai đoạn phân phối
        if len(worst_performers) < self.num_assets_change:
            distribution_candidates = [(asset, data) for asset, data in portfolio_performance.items() 
                                    if data['phase'] == 'distribution' and asset not in worst_performers]
            distribution_candidates.sort(key=lambda x: x[1]['score'])
            worst_performers.extend([asset for asset, _ in distribution_candidates[:self.num_assets_change - len(worst_performers)]])
        
        # 3. Nếu vẫn chưa đủ, bán tài sản có điểm thấp nhất
        if len(worst_performers) < self.num_assets_change:
            remaining_candidates = [(asset, data) for asset, data in portfolio_performance.items() 
                                if asset not in worst_performers]
            remaining_candidates.sort(key=lambda x: x[1]['score'])
            worst_performers.extend([asset for asset, _ in remaining_candidates[:self.num_assets_change - len(worst_performers)]])
        
        # Lựa chọn tài sản tốt nhất để mua
        available_assets = set(self.all_assets) - set(self.selected_assets)
        best_performers = []
        
        # 1. Ưu tiên tài sản trong giai đoạn tích lũy (để mua ở đáy)
        accumulation_assets = [(asset, score) for asset, score in market_phases['accumulation'] 
                            if asset in available_assets]
        accumulation_assets.sort(key=lambda x: x[1], reverse=True)
        best_performers.extend([asset for asset, _ in accumulation_assets[:self.num_assets_change]])
        
        # 2. Nếu chưa đủ, xem xét các xu hướng tăng mạnh
        if len(best_performers) < self.num_assets_change:
            uptrend_assets = [(asset, score) for asset, score in market_phases['strong_uptrend'] 
                            if asset in available_assets and asset not in best_performers]
            uptrend_assets.sort(key=lambda x: x[1], reverse=True)
            best_performers.extend([asset for asset, _ in uptrend_assets[:self.num_assets_change - len(best_performers)]])
        
        # 3. Nếu vẫn chưa đủ, xem xét các xu hướng tăng vừa phải
        if len(best_performers) < self.num_assets_change:
            moderate_assets = [(asset, score) for asset, score in market_phases['moderate_uptrend'] 
                            if asset in available_assets and asset not in best_performers]
            moderate_assets.sort(key=lambda x: x[1], reverse=True)
            best_performers.extend([asset for asset, _ in moderate_assets[:self.num_assets_change - len(best_performers)]])
        
        # 4. Nếu vẫn thiếu, lấy tài sản có điểm cao nhất từ các giai đoạn khác
        if len(best_performers) < self.num_assets_change:
            remaining = [(asset, all_metrics[asset]['technical_score']) for asset in available_assets 
                        if asset not in best_performers]
            remaining.sort(key=lambda x: x[1], reverse=True)
            best_performers.extend([asset for asset, _ in remaining[:self.num_assets_change - len(best_performers)]])
        
        # Đảm bảo số lượng tài sản thay thế không vượt quá số lượng cần thay đổi
        worst_performers = worst_performers[:self.num_assets_change]
        best_performers = best_performers[:self.num_assets_change]
        
        # Thực hiện giao dịch thay thế tài sản
        transactions = []
        current_prices = self.df.loc[self.df.index[self.current_step], self.selected_assets].values
        total_transaction_costs = 0
        
        for old_asset, new_asset in zip(worst_performers, best_performers):
            # Tính toán giá trị và chi phí
            old_idx = list(self.selected_assets).index(old_asset)
            old_value = current_prices[old_idx] * self.shares[old_idx]
            
            # Ghi nhận lý do bán
            sell_reason = "distribution"  # Mặc định
            if old_asset in [a for a, _ in sell_candidates]:
                sell_reason = "sell_signal"
            elif old_asset in [a for a, _ in distribution_candidates]:
                sell_reason = "distribution"
            else:
                sell_reason = "low_score"
            
            # Giao dịch bán
            sell_cost = old_value * self.transaction_cost
            total_transaction_costs += sell_cost
            transactions.append({
                'asset': old_asset, 
                'type': 'sell', 
                'value': old_value,
                'cost': sell_cost, 
                'reason': sell_reason,
                'technical_score': all_metrics[old_asset]['technical_score'],
                'market_phase': all_metrics[old_asset]['market_phase']
            })
            
            # Giao dịch mua
            remaining_value = old_value - sell_cost
            buy_cost = remaining_value * self.transaction_cost
            total_transaction_costs += buy_cost
            remaining_value -= buy_cost
            
            # Cập nhật danh mục và số lượng cổ phiếu
            new_price = self.df.loc[self.df.index[self.current_step], new_asset]
            new_shares = remaining_value / new_price
            
            # Ghi nhận lý do mua
            buy_reason = "unknown"
            if new_asset in [a for a, _ in accumulation_assets]:
                buy_reason = "accumulation"  # Mua ở đáy
            elif new_asset in [a for a, _ in uptrend_assets]:
                buy_reason = "strong_uptrend"  # Mua xu hướng tăng mạnh
            elif new_asset in [a for a, _ in moderate_assets]:
                buy_reason = "moderate_uptrend"
            else:
                buy_reason = "high_score"
            
            self.selected_assets[old_idx] = new_asset
            self.shares[old_idx] = new_shares
            
            transactions.append({
                'asset': new_asset, 
                'type': 'buy', 
                'value': remaining_value,
                'cost': buy_cost, 
                'reason': buy_reason,
                'technical_score': all_metrics[new_asset]['technical_score'],
                'market_phase': all_metrics[new_asset]['market_phase']
            })
        
        # Cập nhật giá trị danh mục và trọng số
        self.portfolio_value = self._calculate_portfolio_value()
        current_prices = self.df.loc[self.df.index[self.current_step], self.selected_assets].values
        self.weights = (self.shares * current_prices) / self.portfolio_value
        
        return transactions
    
    def _adjust_weights_based_on_performance(self, action):
        """Adjust weights based on technical indicators and agent action."""
        technical_evaluation = self._evaluate_selected_assets()
        combined_scores = np.zeros(len(self.selected_assets))
        
        for i, asset in enumerate(self.selected_assets):
            tech_score = technical_evaluation[asset]['technical_score']
            combined_scores[i] = tech_score
        
        # Normalize scores to [-1, 1]
        min_score, max_score = np.min(combined_scores), np.max(combined_scores)
        if max_score != min_score:
            normalized_scores = 2 * (combined_scores - min_score) / (max_score - min_score) - 1
        else:
            normalized_scores = np.zeros_like(combined_scores)
        
        # Combine analysis impact (40%) with agent action (60%)
        performance_impact = normalized_scores * 0.4
        action_impact = action * 0.6
        total_adjustment = performance_impact + action_impact
        
        # Apply adjustments to current weights
        current_weights = self.weights.copy()
        proposed_weights = current_weights + total_adjustment
        
        # Ensure weights are at least limit
        proposed_weights = np.maximum(proposed_weights, self.limit)
        
        # Normalize to sum to 1
        proposed_weights = proposed_weights / np.sum(proposed_weights)
        
        # Apply min_pctweight logic
        weight_changes = proposed_weights - current_weights
        decreasing_assets = weight_changes < 0
        valid_changes = np.abs(weight_changes) >= self.min_pctweight
        need_adjustment = decreasing_assets & ~valid_changes
        
        if np.any(need_adjustment):
            # Keep original weights for assets not meeting min_pctweight
            proposed_weights[need_adjustment] = current_weights[need_adjustment]
            
            # Re-normalize remaining weights
            remaining_weight = 1 - np.sum(proposed_weights[need_adjustment])
            mask = ~need_adjustment
            if np.any(mask):
                proposed_weights[mask] = proposed_weights[mask] * remaining_weight / np.sum(proposed_weights[mask])
        
        return proposed_weights, normalized_scores
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.selected_assets = self.initial_assets if self.initial_assets is not None else \
            np.random.choice(self.all_assets, self.n_assets, replace=False)
        
        self.current_step = 0
        self.days_since_reweight = 0
        self.days_since_rebalance = 0
        self.pending_reweight = False
        self.pending_reweight_days = 0
        self.portfolio_value = self.initial_balance
        
        self.weights = np.array([1/self.n_assets] * self.n_assets)
        self.shares = np.zeros(self.n_assets)
        
        initial_prices = self.df[self.selected_assets].iloc[self.current_step].values
        self.shares = (self.portfolio_value * self.weights) / initial_prices
        
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def _get_observation(self):
        """Tạo observation vector với các chỉ báo kỹ thuật quan trọng nhất từ hệ thống mới."""
        features = []
        current_date = self.df.index[self.current_step]
        
        for asset in self.selected_assets:
            metrics = self.get_technical_indicators(asset, current_date)
            
            # Chỉ lấy các chỉ báo quan trọng nhất
            asset_features = [
                # Xu hướng - Các chỉ báo xu hướng chính
                metrics['trend']['is_uptrend'],
                metrics['trend']['is_strong_uptrend'],
                metrics['trend']['price_to_ma20'],
                metrics['trend']['ma20_slope'],
                
                # Volume - Chỉ báo volume quan trọng nhất
                metrics['volume']['rel_to_avg'],
                metrics['volume']['confirms_price'],
                
                # Price-Volume kết hợp - Các chỉ báo hàng đầu
                metrics['price_volume']['obv_confirms_uptrend'],
                metrics['price_volume']['mfi'] / 100,  # Chuẩn hóa về [0,1]
                
                # Tín hiệu giai đoạn thị trường
                metrics['final_signals']['uptrend_vol_confirmed'],
                metrics['final_signals']['sideway_accumulation'],
                metrics['final_signals']['sideway_to_uptrend'],
                
                # Điểm tổng hợp - Rất quan trọng cho việc ra quyết định
                metrics['final_signals']['trend_follow_score'] / 100,  # Chuẩn hóa về [0,1]
                
                # Giá hiện tại - Cần thiết để tính hiệu suất
                self.df[asset].iloc[self.current_step]
            ]
            
            features.extend(asset_features)
        
        # Thêm thông tin trọng số hiện tại
        features.extend(self.weights)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_portfolio_value(self):
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        return np.sum(self.shares * current_prices)
    
    def calculate_reward(self, old_value, new_value, info):
        """Calculate reward as portfolio return."""
        reward = (new_value - old_value) / old_value
        return reward * 100
    
    def _process_weight_change(self, new_weights, performance_scores):
        """Process weight changes with transactions."""
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        current_value = self._calculate_portfolio_value()
        
        old_values = current_prices * self.shares
        new_values = current_value * new_weights
        value_changes = new_values - old_values
        
        transactions = []
        total_transaction_costs = 0
        
        for i, asset in enumerate(self.selected_assets):
            if value_changes[i] != 0:
                transaction_value = abs(value_changes[i])
                transaction_cost = transaction_value * self.transaction_cost
                total_transaction_costs += transaction_cost
                
                actual_value = transaction_value - transaction_cost
                
                transactions.append({
                    'asset': asset,
                    'type': 'buy' if value_changes[i] > 0 else 'sell',
                    'value': transaction_value,
                    'cost': transaction_cost,
                    'actual_value': actual_value,
                    'old_weight': self.weights[i],
                    'new_weight': new_weights[i],
                    'performance_score': performance_scores[i],
                    'reason': 'reweight'
                })
        
        # Update portfolio after costs
        current_value -= total_transaction_costs
        self.portfolio_value = current_value
        
        # Recalculate shares based on new weights
        for i, asset in enumerate(self.selected_assets):
            self.shares[i] = (current_value * new_weights[i]) / current_prices[i]
        
        self.weights = new_weights
        
        return transactions, total_transaction_costs
    
    def step(self, action):
        """Execute one step in the environment."""
        # Save old state for reward calculation
        old_step = self.current_step
        old_value = self._calculate_portfolio_value()
        old_weights = self.weights.copy()
        
        # Update state
        self.current_step += 1
        self.days_since_reweight += 1
        self.days_since_rebalance += 1
        
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Initialize info dictionary
        info = {
            'rebalanced': False,
            'reweighted': False,
            'transactions': [],
            'metrics': {},
            'old_weights': old_weights,
            'current_value': old_value,
            'transaction_costs': 0,
            'trading_summary': {
                'total_trades': 0, 'buy_trades': 0, 
                'sell_trades': 0, 'total_volume': 0
            }
        }
        
        # Handle rebalance portfolio
        if self.days_since_rebalance >= self.rebalance_window:
            self.days_since_rebalance = 0
            transactions = self._rebalance_portfolio()
            info['transactions'].extend(transactions)
            info['rebalanced'] = True
            
            self._update_trading_summary(info, transactions)
            info['transaction_costs'] += sum(t['cost'] for t in transactions)
            
            # Delay reweight if needed
            if self.days_since_reweight >= self.reweight:
                self.pending_reweight = True
                self.pending_reweight_days = 2
        
        # Handle reweighting (combine the two reweight cases)
        should_reweight = (
            (self.pending_reweight and self.pending_reweight_days <= 0) or
            (self.days_since_reweight >= self.reweight and not self.pending_reweight)
        )
        
        if should_reweight:
            self.days_since_reweight = 0
            self.pending_reweight = False
            
            new_weights, performance_scores = self._adjust_weights_based_on_performance(action)
            transactions, costs = self._process_weight_change(new_weights, performance_scores)
            
            info['transactions'].extend(transactions)
            info['reweighted'] = True
            info['transaction_costs'] += costs
            
            self._update_trading_summary(info, transactions)
        
        # Decrement pending reweight days if needed
        if self.pending_reweight:
            self.pending_reweight_days -= 1
        
        # Calculate new portfolio value and reward
        new_value = self._calculate_portfolio_value()
        reward = self.calculate_reward(old_value, new_value, info)
        
        # Add performance info
        info['portfolio_return'] = (new_value - old_value) / old_value
        info['portfolio_value'] = new_value
        info['cumulative_return'] = (new_value - self.initial_balance) / self.initial_balance
        
        # Add detailed asset info
        asset_info = {}
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        for i, asset in enumerate(self.selected_assets):
            asset_value = current_prices[i] * self.shares[i]
            asset_return = (current_prices[i] - self.df[asset].iloc[old_step]) / self.df[asset].iloc[old_step]
            
            asset_info[asset] = {
                'price': current_prices[i],
                'shares': self.shares[i],
                'weight': self.weights[i],
                'value': asset_value,
                'daily_return': asset_return,
                'contribution': (asset_value / new_value) * asset_return
            }
        info['assets'] = asset_info
        
        observation = self._get_observation()
        
        return observation, reward, done, truncated, info

    def _update_trading_summary(self, info, transactions):
        """Helper method to update trading summary info."""
        for t in transactions:
            info['trading_summary']['total_trades'] += 1
            info['trading_summary']['total_volume'] += t['value']
            info['trading_summary']['buy_trades' if t['type'] == 'buy' else 'sell_trades'] += 1

class RiskPortfolioEnv(gym.Env):
    def __init__(self, df, original_data, risk_metrics, market_index, n_assets=5, initial_balance=10_000_000,
                 reweight=20, min_pctweight=0.05, limit=0.1,
                 lookback_period=20, rebalance_window=40,
                 num_assets_change=2, transition_cost=0.01):
        super(RiskPortfolioEnv, self).__init__()

        self.df = df
        self.original_data = original_data
        self.n_assets = n_assets
        self.initial_balance = initial_balance
        self.reweight = reweight
        self.min_pctweight = min_pctweight
        self.limit = limit
        self.lookback_period = lookback_period
        self.rebalance_window = rebalance_window
        self.num_assets_change = num_assets_change
        self.transition_cost = transition_cost
        self.days_since_rebalance = 0
        self.pending_reweight = False
        self.pending_reweight_days = 0
        
        self.risk_metrics = risk_metrics

        # Risk management parameters
        self.max_drawdown_limit = 0.05  # 10% maximum drawdown limit
        self.var_confidence_level = 0.95  # 95% VaR confidence level
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.max_volatility = 0.05  # 15% maximum portfolio volatility
        self.max_correlation = 0.4

        # Add cash management parameters
        self.cash = initial_balance  # Initialize cash holdings
        self.min_cash_ratio = 0.05   # Minimum cash to maintain (5%)
        self.max_cash_ratio = 0.95   # Maximum cash allowed (95%)
        self.optimal_cash_uptrend = 0.10  # Target cash ratio trong uptrend
        self.optimal_cash_neutral = 0.15   # Target cash ratio trong thị trường đi ngang

        # Market condition parameters
        self.market_index = market_index
        self.volatility_threshold = 0.15  # Annual volatility threshold
        self.stress_threshold = -0.15     # Market stress threshold

        # Market regime parameters
        self.trend_threshold = 0.05      # 5% trend threshold
        self.strong_trend_threshold = 0.10  # Ngưỡng cho strong uptrend
        self.vol_regime_threshold = 0.2   # Volatility regime threshold

        self.all_assets = df.columns.tolist()

        assert all(asset in original_data.columns for asset in self.all_assets)

        # Modified observation space to include cash and market features
        n_risk_features = 12  # Original risk features
        n_market_features = 8  # New market condition features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_assets * n_risk_features + n_assets + n_market_features + 1,),  # +1 for cash ratio
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0, high=1, shape=(n_assets,), dtype=np.float32
        )

        self.start_date = df.index[0]
        self.original_start_idx = original_data.index.get_loc(self.start_date)

    def get_risk_metrics(self, asset, date):
        """
        Get precomputed risk metrics for an asset at a specific date.
        If indicators aren't precomputed, return default values.
        """
        if date in self.risk_metrics and asset in self.risk_metrics[date]:
            return self.risk_metrics[date][asset]
        else:
            # Return default values if data not available
            # This structure matches the new indicators from precompute_technical_indicators
            return {
                'mean_return': 0,
                'volatility': 0,
                'var_95': 0,
                'cvar_95': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'beta': 0,
                'skewness': 0,
                'kurtosis': 0,
                'info_ratio': 0,
                'tracking_error': 0
            }

    def get_market_metrics(self, date):
        """
        Get precomputed technical indicators for an asset at a specific date.
        If indicators aren't precomputed, return default values.
        """
        if date in self.market_index:
            return self.market_index[date]
        else:
            # Return default values if data not available
            # This structure matches the new indicators from precompute_technical_indicators
            return {
                'trend_signal': 0,
                'trend_strength': 0,
                'mfi': 0,
                'buying_pressure': 0,
                'volume_strength': 0,
                'market_stress': 0,
                'crisis_score': 0,
                'volatility': 0,
                'rsi': 0,
                'regime': 0
            }

    def _determine_optimal_cash_ratio(self, market_metrics):
        """Xác định tỷ lệ tiền mặt tối ưu dựa trên điều kiện thị trường."""
        base_cash_ratio = self.min_cash_ratio

        # 1. Phân tích market regime với logic mới
        regime_adjustments = {
            'strong_bullish': 0.0,    # Không cần giữ nhiều tiền
            'bullish': 0.05,          # Giữ ít tiền
            'neutral': 0.10,          # Giữ tiền vừa phải
            'weakening_bearish': 0.20, # Tăng tiền khi thị trường yếu
            'bearish': 0.30           # Giữ nhiều tiền trong bearish
        }
        base_cash_ratio += regime_adjustments.get(market_metrics['regime'], 0.10)

        # 2. Điều chỉnh theo trend với logic mới
        if market_metrics['trend_signal'] > 0:  # Uptrend
            if market_metrics['trend_strength'] > self.strong_trend_threshold:
                # Strong uptrend - giữ minimum cash
                base_cash_ratio = min(base_cash_ratio, self.optimal_cash_uptrend)
            elif market_metrics['trend_strength'] > self.trend_threshold:
                # Normal uptrend - giữ ít cash
                base_cash_ratio = min(base_cash_ratio, self.optimal_cash_uptrend + 0.05)

        # 3. Điều chỉnh theo MFI với ngưỡng mới
        mfi = market_metrics['mfi']
        if mfi > 80:  # Thị trường quá mua
            if market_metrics['trend_signal'] > 0:  # Trong uptrend
                base_cash_ratio += 0.05  # Tăng nhẹ cash
            else:
                base_cash_ratio += 0.10  # Tăng cash nhiều hơn
        elif mfi < 40:  # Thị trường quá bán
            if market_metrics['buying_pressure'] > 0.2:
                base_cash_ratio = max(self.min_cash_ratio, base_cash_ratio - 0.15)

        # 4. Điều chỉnh theo Volume Strength
        vol_strength = market_metrics['volume_strength']
        if vol_strength > 1.5:  # Volume tăng mạnh
            if market_metrics['trend_signal'] > 0:
                base_cash_ratio = max(self.min_cash_ratio, base_cash_ratio - 0.05)

        # 5. Điều chỉnh theo Market Stress với logic mới
        stress_level = abs(market_metrics['market_stress'])
        if stress_level < 0.05:  # Thị trường ổn định
            if market_metrics['trend_signal'] > 0:
                base_cash_ratio = min(base_cash_ratio, self.optimal_cash_uptrend)
        else:
            stress_adjustment = min(0.3, stress_level)
            if market_metrics['buying_pressure'] > 0.2:
                stress_adjustment *= 0.5
            base_cash_ratio += stress_adjustment

        # 6. Điều chỉnh theo RSI với logic mới
        if market_metrics['rsi'] > 70:
            if market_metrics['trend_signal'] > 0:  # Trong uptrend
                base_cash_ratio += 0.05  # Tăng nhẹ
            else:
                base_cash_ratio += 0.10  # Tăng nhiều hơn
        elif market_metrics['rsi'] < 30:
            if market_metrics['buying_pressure'] > 0.2:
                base_cash_ratio = max(self.min_cash_ratio, base_cash_ratio - 0.15)

        # 7. Điều chỉnh theo Crisis Score với logic mới
        if market_metrics['crisis_score'] > 0.5:
            crisis_adjustment = market_metrics['crisis_score'] * 0.4
            if market_metrics['buying_pressure'] > 0.2:
                crisis_adjustment *= 0.6
            base_cash_ratio += crisis_adjustment
        else:
            # Không có crisis - ưu tiên đầu tư
            if market_metrics['trend_signal'] > 0:
                base_cash_ratio = min(base_cash_ratio, self.optimal_cash_uptrend)

        # 8. Volatility check với logic mới
        if market_metrics['volatility'] < self.vol_regime_threshold:
            if market_metrics['trend_signal'] > 0:
                # Volatility thấp + uptrend = giảm cash
                base_cash_ratio = min(base_cash_ratio, self.optimal_cash_uptrend)
        else:
            vol_adjustment = (market_metrics['volatility'] - self.vol_regime_threshold) * 2
            base_cash_ratio += vol_adjustment

        # Đảm bảo tỷ lệ nằm trong giới hạn
        return np.clip(base_cash_ratio, self.min_cash_ratio, self.max_cash_ratio)

    def _evaluate_all_assets(self):
        """Evaluate all assets based on risk metrics."""
        all_metrics = {}
        current_date = self.df.index[self.current_step]
        for asset in self.all_assets:
            metrics = self.get_risk_metrics(asset, current_date)

            # Calculate risk score based on multiple factors
            risk_score = (
                0.25 * metrics['sharpe_ratio'] +  # Higher is better
                0.20 * (-metrics['volatility']) +  # Lower is better
                0.15 * (-metrics['max_drawdown']) +  # Lower is better
                0.15 * metrics['sortino_ratio'] +  # Higher is better
                0.15 * (-abs(metrics['beta'] - 1)) +  # Closer to 1 is better
                0.10 * metrics['info_ratio']  # Higher is better
            )

            all_metrics[asset] = {
                'risk_score': risk_score,
                'metrics': metrics
            }

        return all_metrics

    def _calculate_portfolio_risk_metrics(self):
        """Calculate risk metrics for the entire portfolio."""
        current_date = self.df.index[self.current_step]
        current_idx = self.original_data.index.get_loc(current_date)
        start_idx = max(0, current_idx - self.lookback_period)

        portfolio_returns = []
        asset_returns = []

        # Calculate returns for each asset
        for asset in self.selected_assets:
            prices = self.original_data[asset].iloc[start_idx:current_idx + 1]
            returns = prices.pct_change().dropna()
            asset_returns.append(returns)

        # Ensure all return series are aligned and have same length
        min_length = min(len(returns) for returns in asset_returns)
        if min_length > 0:
            asset_returns_aligned = [returns.iloc[-min_length:] for returns in asset_returns]

            # Calculate portfolio returns using aligned data
            portfolio_returns = np.sum([returns * weight for returns, weight in
                                    zip(asset_returns_aligned, self.weights)], axis=0)

            # Calculate basic portfolio metrics
            port_mean_return = portfolio_returns.mean() * 252  # Annualized
            port_volatility = portfolio_returns.std() * np.sqrt(252)

            # Portfolio VaR and CVaR
            port_var_95 = np.percentile(portfolio_returns, (1 - self.var_confidence_level) * 100)
            port_cvar_95 = portfolio_returns[portfolio_returns <= port_var_95].mean()

            # Portfolio Beta calculation with error handling
            try:
                market_returns = self.df.mean(axis=1).iloc[start_idx:current_idx + 1].pct_change().dropna()

                # Align market returns with portfolio returns
                if len(market_returns) >= min_length:
                    market_returns_aligned = market_returns.iloc[-min_length:]

                    # Calculate beta using aligned data
                    covariance = np.cov(portfolio_returns, market_returns_aligned)[0][1]
                    market_variance = market_returns_aligned.var()
                    port_beta = covariance / market_variance if market_variance != 0 else 1
                else:
                    port_beta = 1
            except Exception:
                port_beta = 1  # Fallback to neutral beta

            # Portfolio Sharpe Ratio
            port_excess_returns = port_mean_return - self.risk_free_rate
            port_sharpe = port_excess_returns / port_volatility if port_volatility != 0 else 0

        else:
            # Default values if not enough data
            port_mean_return = 0
            port_volatility = 0
            port_var_95 = 0
            port_cvar_95 = 0
            port_sharpe = 0
            port_beta = 1

        return {
            'return': port_mean_return,
            'volatility': port_volatility,
            'var_95': port_var_95,
            'cvar_95': port_cvar_95,
            'sharpe': port_sharpe,
            'beta': port_beta
        }

    def _rebalance_portfolio(self):
        """Rebalance portfolio based on risk metrics and dynamic correlation thresholds."""
        # Evaluate all assets based on risk metrics
        all_metrics = self._evaluate_all_assets()

        # Analyze portfolio trend
        lookback_window = self.lookback_period  # Số ngày để xác định trend
        start_idx = max(0, self.current_step - lookback_window)
        portfolio_values = []

        for i in range(start_idx, self.current_step + 1):
            prices = self.df[self.selected_assets].iloc[i].values
            value = np.sum(self.shares * prices)
            portfolio_values.append(value)

        # Tính toán portfolio trend
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        trend_strength = np.mean(portfolio_returns)

        # Điều chỉnh correlation threshold dựa trên trend
        base_correlation = self.max_correlation  # e.g., 0.4
        if trend_strength > 0:
            # Nếu trend tăng, cho phép correlation cao hơn
            adjusted_correlation = min(0.8, base_correlation + trend_strength * 2)
        else:
            # Nếu trend giảm, yêu cầu correlation thấp hơn
            adjusted_correlation = max(0.2, base_correlation + trend_strength * 2)

        # Calculate correlation matrix for current portfolio
        returns_matrix = []
        for asset in self.selected_assets:
            prices = self.original_data[asset].iloc[max(0, self.current_step - 126):self.current_step + 1]
            returns = prices.pct_change().dropna()
            returns_matrix.append(returns)
        current_correlation = np.corrcoef(returns_matrix)

        # Calculate average correlation and trend alignment for each asset
        current_correlations = {}
        trend_alignments = {}
        for i, asset in enumerate(self.selected_assets):
            avg_corr = np.mean([c for j, c in enumerate(current_correlation[i]) if j != i])
            current_correlations[asset] = avg_corr

            # Tính độ phù hợp của asset với trend
            asset_returns = returns_matrix[i]
            asset_trend = np.mean(asset_returns[-lookback_window:])
            trend_alignments[asset] = 1 if (asset_trend > 0) == (trend_strength > 0) else -1

        # Adjust risk scores based on correlation and trend alignment
        portfolio_performance = {}
        for asset in self.selected_assets:
            base_score = all_metrics[asset]['risk_score']

            if trend_strength > 0:
                # Trong uptrend, ưu tiên assets có correlation cao nếu chúng cũng đang tăng
                if trend_alignments[asset] > 0:
                    corr_bonus = max(0, current_correlations[asset] - base_correlation)
                    adjusted_score = base_score * (1 + corr_bonus)
                else:
                    corr_penalty = max(0, current_correlations[asset] - adjusted_correlation)
                    adjusted_score = base_score * (1 - corr_penalty)
            else:
                # Trong downtrend, phạt nặng assets có correlation cao
                corr_penalty = max(0, current_correlations[asset] - adjusted_correlation)
                adjusted_score = base_score * (1 - corr_penalty * 1.5)

            portfolio_performance[asset] = adjusted_score

        # Sort current portfolio assets by adjusted performance
        worst_performers = sorted(portfolio_performance.items(),
                                key=lambda x: x[1])[:self.num_assets_change]

        # For available assets, calculate potential correlation with remaining portfolio
        available_assets = set(self.all_assets) - set(self.selected_assets)
        remaining_assets = set(self.selected_assets) - {asset for asset, _ in worst_performers}

        # Calculate correlations and trend alignment for potential new assets
        available_performance = {}
        for asset in available_assets:
            # Get returns for candidate asset
            candidate_prices = self.original_data[asset].iloc[max(0, self.current_step - 126):self.current_step + 1]
            candidate_returns = candidate_prices.pct_change().dropna()

            # Calculate correlation with remaining portfolio assets
            correlations = []
            for remain_asset in remaining_assets:
                remain_prices = self.original_data[remain_asset].iloc[max(0, self.current_step - 126):self.current_step + 1]
                remain_returns = remain_prices.pct_change().dropna()

                min_length = min(len(candidate_returns), len(remain_returns))
                if min_length > 0:
                    corr = np.corrcoef(candidate_returns[-min_length:], remain_returns[-min_length:])[0,1]
                    correlations.append(corr)

            avg_corr = np.mean(correlations) if correlations else 0

            # Calculate trend alignment for candidate asset
            asset_trend = np.mean(candidate_returns[-lookback_window:])
            trend_align = 1 if (asset_trend > 0) == (trend_strength > 0) else -1

            # Adjust score based on trend and correlation
            base_score = all_metrics[asset]['risk_score']
            if trend_strength > 0:
                if trend_align > 0:
                    # Trong uptrend, ưu tiên assets có correlation cao nếu cùng trend
                    corr_bonus = max(0, avg_corr - base_correlation)
                    adjusted_score = base_score * (1 + corr_bonus)
                else:
                    corr_penalty = max(0, avg_corr - adjusted_correlation)
                    adjusted_score = base_score * (1 - corr_penalty)
            else:
                # Trong downtrend, tìm assets có correlation thấp
                corr_penalty = max(0, avg_corr - adjusted_correlation)
                adjusted_score = base_score * (1 - corr_penalty * 1.5)

            available_performance[asset] = adjusted_score

        # Select best performing assets
        best_performers = sorted(available_performance.items(),
                            key=lambda x: x[1],
                            reverse=True)[:self.num_assets_change]

        # Execute portfolio rebalancing
        transactions = []
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        total_transaction_costs = 0

        # Process each asset swap
        for (old_asset, _), (new_asset, _) in zip(worst_performers, best_performers):
            old_idx = list(self.selected_assets).index(old_asset)
            old_value = current_prices[old_idx] * self.shares[old_idx]

            # Calculate and apply selling costs
            sell_cost = old_value * self.transition_cost
            total_transaction_costs += sell_cost

            # Record sell transaction with enhanced info
            transactions.append({
                'asset': old_asset,
                'type': 'sell',
                'value': old_value,
                'cost': sell_cost,
                'reason': 'rebalance',
                'risk_metrics': all_metrics[old_asset]['metrics'],
                'correlation': current_correlations.get(old_asset, 0),
                'trend_alignment': trend_alignments.get(old_asset, 0),
                'portfolio_trend': trend_strength
            })

            remaining_value = old_value - sell_cost
            buy_cost = remaining_value * self.transition_cost
            total_transaction_costs += buy_cost
            remaining_value -= buy_cost

            # Update portfolio with new asset
            new_price = self.df[new_asset].iloc[self.current_step]
            new_shares = remaining_value / new_price

            self.selected_assets[old_idx] = new_asset
            self.shares[old_idx] = new_shares

            # Record buy transaction with enhanced info
            transactions.append({
                'asset': new_asset,
                'type': 'buy',
                'value': remaining_value,
                'cost': buy_cost,
                'reason': 'rebalance',
                'risk_metrics': all_metrics[new_asset]['metrics'],
                'expected_correlation': available_performance.get(new_asset, 0),
                'portfolio_trend': trend_strength,
                'correlation_threshold': adjusted_correlation
            })

        # Update portfolio value and weights
        self.portfolio_value = self._calculate_portfolio_value()
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        self.weights = (self.shares * current_prices) / self.portfolio_value

        # Add portfolio risk info
        portfolio_risk = self._calculate_portfolio_risk_metrics()
        for transaction in transactions:
            transaction['portfolio_risk_after'] = portfolio_risk

        return transactions

    def _adjust_weights_based_on_risk(self, action):
        """Adjust portfolio weights based on risk metrics and action while maintaining original constraints."""
        # Get current weights for comparison
        current_weights = self.weights.copy()
        current_date = self.df.index[self.current_step]
        # Get individual asset risk metrics
        risk_metrics = {asset: self.get_risk_metrics(asset, current_date)
                    for asset in self.selected_assets}

        # Calculate risk-adjusted weights
        risk_scores = np.array([
            risk_metrics[asset]['sharpe_ratio'] / (risk_metrics[asset]['volatility'] + 1e-6)
            for asset in self.selected_assets
        ])

        # Normalize risk scores
        risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-6)

        # Combine action and risk scores
        combined_scores = 0.3 * action + 0.2 * risk_scores

        # Apply risk limits
        for i, asset in enumerate(self.selected_assets):
            metrics = risk_metrics[asset]

            # Reduce weight if asset has high volatility
            if metrics['volatility'] > self.max_volatility:
                combined_scores[i] *= (self.max_volatility / metrics['volatility'])

            # Reduce weight if asset has large drawdown
            if abs(metrics['max_drawdown']) > self.max_drawdown_limit:
                combined_scores[i] *= (self.max_drawdown_limit / abs(metrics['max_drawdown']))

        # Initial normalization to get proposed weights, considering cash weight
        total_investment_ratio = 1 - self.cash_weight
        proposed_weights = combined_scores / combined_scores.sum() * total_investment_ratio

        # Ensure minimum weight constraint
        proposed_weights = np.maximum(proposed_weights, self.limit * total_investment_ratio)
        proposed_weights = proposed_weights / proposed_weights.sum() * total_investment_ratio

        # Calculate weight changes from current weights
        weight_changes = proposed_weights - current_weights

        # Identify assets with weight decreases
        decreasing_assets = weight_changes < 0

        # Check minimum percentage change constraint
        valid_changes = np.abs(weight_changes) >= self.min_pctweight

        # Only apply min_pctweight for decreasing weights
        need_adjustment = decreasing_assets & ~valid_changes

        if np.any(need_adjustment):
            # Keep original weights for assets not meeting min_pctweight
            proposed_weights[need_adjustment] = current_weights[need_adjustment]

            # Normalize remaining weights while preserving cash weight
            remaining_weight = total_investment_ratio - np.sum(proposed_weights[need_adjustment])
            mask = ~need_adjustment
            if np.any(mask):
                proposed_weights[mask] = proposed_weights[mask] * remaining_weight / np.sum(proposed_weights[mask])

        return proposed_weights, risk_scores

    def _adjust_portfolio_for_cash(self, proposed_weights, optimal_cash_ratio):
        """Adjust portfolio weights to account for cash position."""
        # Calculate current portfolio value including cash
        total_value = self._calculate_portfolio_value()

        # Calculate the current and target investment ratios
        current_investment_ratio = 1 - self.cash_weight
        target_investment_ratio = 1 - optimal_cash_ratio

        # Scale asset weights to match target investment ratio
        adjusted_weights = proposed_weights * (target_investment_ratio / current_investment_ratio)

        # Create transactions to adjust to new weights
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        current_asset_values = current_prices * self.shares
        new_asset_values = total_value * adjusted_weights
        value_changes = new_asset_values - current_asset_values

        transactions = []
        total_transaction_costs = 0

        # Process each asset
        for i, asset in enumerate(self.selected_assets):
            if abs(value_changes[i]) > 1:  # Add small threshold to avoid tiny trades
                transaction_value = abs(value_changes[i])
                transaction_cost = transaction_value * self.transition_cost
                total_transaction_costs += transaction_cost

                if value_changes[i] < 0:  # Selling
                    actual_value = transaction_value - transaction_cost
                    self.cash += actual_value
                else:  # Buying
                    actual_value = transaction_value + transaction_cost
                    self.cash -= actual_value

                # Update shares
                price = current_prices[i]
                shares_change = value_changes[i] / price
                self.shares[i] += shares_change

                transactions.append({
                    'asset': asset,
                    'type': 'buy' if value_changes[i] > 0 else 'sell',
                    'value': transaction_value,
                    'cost': transaction_cost,
                    'shares_change': shares_change,
                    'reason': 'cash_adjustment'
                })

        # Update weights after all transactions
        self._update_weights()

        return adjusted_weights, transactions, total_transaction_costs

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state with proper initialization of all variables."""
        super().reset(seed=seed)
        self.selected_assets = np.random.choice(self.all_assets, self.n_assets, replace=False)

        self.current_step = 0
        self.days_since_reweight = 0
        self.days_since_rebalance = 0
        self.pending_reweight = False
        self.pending_reweight_days = 0
        # Initialize portfolio value and cash position
        self.portfolio_value = self.initial_balance
        self.cash = self.initial_balance * self.min_cash_ratio
        self.cash_weight = self.min_cash_ratio

        # Initialize asset weights
        total_investment_ratio = 1 - self.cash_weight
        self.weights = np.array([total_investment_ratio/self.n_assets] * self.n_assets)

        # Calculate initial shares
        initial_prices = self.df[self.selected_assets].iloc[self.current_step].values
        investment_value = self.initial_balance * total_investment_ratio
        self.shares = (investment_value * self.weights / total_investment_ratio) / initial_prices

        # Verify initial setup
        actual_portfolio_value = self._calculate_portfolio_value()
        assert np.abs(actual_portfolio_value - self.initial_balance) < 1e-6, (
            f"Initial portfolio value incorrect: {actual_portfolio_value} != {self.initial_balance}"
        )

        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        """Create observation vector with risk metrics."""
        features = []
        current_date = self.df.index[self.current_step]
        self.portfolio_value = self._calculate_portfolio_value()
        for asset in self.selected_assets:
            metrics = self.get_risk_metrics(asset, current_date)

            asset_features = [
                metrics['mean_return'],
                metrics['volatility'],
                metrics['var_95'],
                metrics['cvar_95'],
                metrics['max_drawdown'],
                metrics['sharpe_ratio'],
                metrics['sortino_ratio'],
                metrics['beta'],
                metrics['skewness'],
                metrics['kurtosis'],
                metrics['info_ratio'],
                metrics['tracking_error']
            ]

            # Fill NaN với 0 cho từng asset
            asset_features = [0.0 if np.isnan(x) else x for x in asset_features]
            features.extend(asset_features)

        # Add current weights
        features.extend(self.weights)

        # Add market condition features
        market_metrics = self.get_market_metrics(current_date)
        market_features = [
            market_metrics['trend_signal'],
            market_metrics['trend_strength'],
            market_metrics['volatility'],
            market_metrics['vol_regime'],
            market_metrics['market_stress'],
            market_metrics['rsi'] / 100,  # Normalize RSI
            float(market_metrics['regime'] == 'bullish'),
            market_metrics['crisis_score']
        ]

        # Fill NaN với 0 cho market features
        market_features = [0.0 if np.isnan(x) else x for x in market_features]
        features.extend(market_features)

        # Add cash ratio và xử lý NaN
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value != 0 else 0.0
        features.append(0.0 if np.isnan(cash_ratio) else cash_ratio)

        # Convert sang numpy array và đảm bảo không có NaN
        obs = np.array(features, dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0)

    def _calculate_portfolio_value(self):
        """Calculate total portfolio value including cash."""
        if not hasattr(self, 'shares') or not hasattr(self, 'cash'):
            return self.initial_balance

        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        asset_values = np.sum(self.shares * current_prices)
        total_value = asset_values + self.cash
        return total_value

    def calculate_reward(self, old_value, new_value, info):
        """Calculate reward based on risk-adjusted returns."""
        # Get portfolio metrics
        portfolio_metrics = self._calculate_portfolio_risk_metrics()

        # Calculate base return
        base_return = (new_value - old_value) / old_value

        # Risk-adjusted components
        sharpe_component = portfolio_metrics['sharpe']
        volatility_penalty = -max(0, portfolio_metrics['volatility'] - self.max_volatility) ** 2
        var_penalty = -max(0, abs(portfolio_metrics['var_95']) - 0.02) ** 2  # Penalty for VaR > 2%

        # Diversification bonus
        hhi = np.sum(self.weights ** 2)  # Herfindahl-Hirschman Index
        diversification_bonus = 1 - hhi

        # Risk limit penalties
        risk_penalties = 0
        if portfolio_metrics['volatility'] > self.max_volatility:
            risk_penalties -= 0.2
        if abs(portfolio_metrics['beta']) > 1.5:
            risk_penalties -= 0.2

        # Drawdown penalty
        if abs(portfolio_metrics.get('max_drawdown', 0)) > self.max_drawdown_limit:
            risk_penalties -= 0.2

        # Transaction cost impact
        transaction_cost_impact = -info.get('transaction_costs', 0) / new_value

        # Combine all components
        reward = (
            0.4 * base_return +
            1 * sharpe_component +
            2 * volatility_penalty +
            1 * var_penalty +
            0.1 * diversification_bonus +
            1 * risk_penalties +
            transaction_cost_impact
        )

        # Store components in info
        info['reward_components'] = {
            'base_return': base_return,
            'sharpe_component': sharpe_component,
            'volatility_penalty': volatility_penalty,
            'var_penalty': var_penalty,
            'diversification_bonus': diversification_bonus,
            'risk_penalties': risk_penalties,
            'transaction_cost_impact': transaction_cost_impact,
            'total_reward': reward
        }

        return reward
    
    def _update_weights(self):
        """Update weights for both assets and cash to ensure they sum to 1."""
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        total_value = self._calculate_portfolio_value()

        # Calculate asset weights
        asset_values = self.shares * current_prices
        self.weights = asset_values / total_value

        # Calculate cash weight
        self.cash_weight = self.cash / total_value

        # Verify total weights sum to 1
        assert np.abs(np.sum(self.weights) + self.cash_weight - 1.0) < 1e-6, "Weights don't sum to 1"

    def _move_to_cash(self, target_cash_ratio):
        """Liquidate positions to increase cash holdings during crisis."""
        total_value = self._calculate_portfolio_value()
        current_cash_ratio = self.cash / total_value

        if current_cash_ratio >= target_cash_ratio:
            return []

        # Calculate how much more cash we need
        target_cash = total_value * target_cash_ratio
        cash_needed = target_cash - self.cash

        transactions = []
        total_transaction_costs = 0

        # Get current asset values
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        asset_values = current_prices * self.shares
        total_asset_value = np.sum(asset_values)

        # Calculate sell ratio to achieve target cash
        sell_ratio = cash_needed / total_asset_value

        # Proportionally sell assets
        for i, asset in enumerate(self.selected_assets):
            sell_value = asset_values[i] * sell_ratio

            if sell_value > 1:  # Add small threshold to avoid tiny trades
                transaction_cost = sell_value * self.transition_cost
                total_transaction_costs += transaction_cost
                actual_value = sell_value - transaction_cost

                # Update shares
                shares_to_sell = (sell_value / current_prices[i])
                self.shares[i] -= shares_to_sell

                # Update cash
                self.cash += actual_value

                transactions.append({
                    'asset': asset,
                    'type': 'sell',
                    'value': sell_value,
                    'cost': transaction_cost,
                    'shares_change': -shares_to_sell,
                    'reason': 'crisis_liquidation'
                })

        # Update weights after all transactions
        self._update_weights()

        return transactions

    def step(self, action):
        """Execute one step in the environment with improved cash management."""
        # Store old state for reward calculation
        old_step = self.current_step
        old_value = self._calculate_portfolio_value()
        old_weights = self.weights.copy()
        old_cash_weight = self.cash_weight

        # Update state
        self.current_step += 1
        self.days_since_reweight += 1
        self.days_since_rebalance += 1

        done = self.current_step >= len(self.df) - 1
        truncated = False

        # Initialize info dictionary
        info = {
            'rebalanced': False,
            'reweighted': False,
            'transactions': [],
            'metrics': {},
            'old_weights': old_weights,
            'old_cash_weight': old_cash_weight,
            'current_value': old_value,
            'transaction_costs': 0,
            'cash_position': self.cash,
            'trading_summary': {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_volume': 0
            },
            'market_conditions': {}
        }

        # Analyze market conditions and determine optimal cash ratio
        current_date = self.df.index[old_step]
        market_metrics = self.get_market_metrics(current_date)
        optimal_cash_ratio = self._determine_optimal_cash_ratio(market_metrics)
        info['market_conditions'] = market_metrics
        info['optimal_cash_ratio'] = optimal_cash_ratio

        # Handle portfolio rebalancing
        if self.days_since_rebalance >= self.rebalance_window:
            self.days_since_rebalance = 0

            if market_metrics['crisis_score'] < 0.7 and market_metrics['regime'] != 'bearish':
                transactions = self._rebalance_portfolio()
                info['transactions'].extend(transactions)
                info['rebalanced'] = True

                # Update trading summary
                for t in transactions:
                    info['trading_summary']['total_trades'] += 1
                    info['trading_summary']['total_volume'] += t['value']
                    if t['type'] == 'buy':
                        info['trading_summary']['buy_trades'] += 1
                    else:
                        info['trading_summary']['sell_trades'] += 1

                # Calculate total transaction costs
                total_costs = sum(t['cost'] for t in transactions)
                info['transaction_costs'] += total_costs

                # Update weights after rebalancing
                self._update_weights()

                if self.days_since_reweight >= self.reweight:
                    self.pending_reweight = True
                    self.pending_reweight_days = 2
            else:
                # In crisis conditions, move to cash if needed
                if self.cash_weight < optimal_cash_ratio:
                    liquidation_transactions = self._move_to_cash(optimal_cash_ratio)
                    info['transactions'].extend(liquidation_transactions)
                    info['crisis_liquidation'] = True

                    # Update trading summary
                    for t in liquidation_transactions:
                        info['trading_summary']['total_trades'] += 1
                        info['trading_summary']['total_volume'] += t['value']
                        info['trading_summary']['sell_trades'] += 1

                    info['transaction_costs'] += sum(t['cost'] for t in liquidation_transactions)

        # Handle weight adjustments (reweighting)
        elif (self.pending_reweight and self.pending_reweight_days <= 0) or \
            (self.days_since_reweight >= self.reweight and not self.pending_reweight):

            if market_metrics['crisis_score'] < 0.8:
                # Reset reweight timers
                self.days_since_reweight = 0
                self.pending_reweight = False

                # Calculate new weights
                new_weights, performance_scores = self._adjust_weights_based_on_risk(action)

                # Adjust for cash position
                adjusted_weights, transactions, costs = self._adjust_portfolio_for_cash(
                    new_weights, optimal_cash_ratio)

                info['transactions'].extend(transactions)
                info['reweighted'] = True
                info['transaction_costs'] += costs

                # Update trading summary
                for t in transactions:
                    info['trading_summary']['total_trades'] += 1
                    info['trading_summary']['total_volume'] += t['value']
                    if t['type'] == 'buy':
                        info['trading_summary']['buy_trades'] += 1
                    else:
                        info['trading_summary']['sell_trades'] += 1

        if self.pending_reweight:
            self.pending_reweight_days -= 1

        # Calculate new portfolio value and reward
        new_value = self._calculate_portfolio_value()
        reward = self.calculate_reward(old_value, new_value, info)

        # Update info with performance metrics
        info['portfolio_return'] = (new_value - old_value) / old_value
        info['portfolio_value'] = new_value
        info['cumulative_return'] = (new_value - self.initial_balance) / self.initial_balance
        info['cash_ratio'] = self.cash_weight

        # Add detailed asset information
        asset_info = {}
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        for i, asset in enumerate(self.selected_assets):
            asset_value = current_prices[i] * self.shares[i]
            asset_return = (current_prices[i] - self.df[asset].iloc[old_step]) / self.df[asset].iloc[old_step]

            asset_info[asset] = {
                'price': current_prices[i],
                'shares': self.shares[i],
                'weight': self.weights[i],
                'value': asset_value,
                'daily_return': asset_return,
                'contribution': (asset_value / new_value) * asset_return
            }
        info['assets'] = asset_info

        # Get observation
        observation = self._get_observation()

        return observation, reward, done, truncated, info

class CombineEnv(gym.Env):
    def __init__(self, df, precomputed_indicators, weinstein, n_assets=5, initial_assets=None, initial_balance=10_000_000, 
                 reweight=20, min_pctweight=0.05, limit=0.1, transaction_cost=0.01):
        super(CombineEnv, self).__init__()
        
        # Store parameters
        self.df = df
        self.n_assets = len(initial_assets) if initial_assets is not None else n_assets
        self.initial_assets = initial_assets
        self.initial_balance = initial_balance
        self.reweight = reweight
        self.min_pctweight = min_pctweight
        self.limit = limit
        self.transaction_cost = transaction_cost
        self.pending_reweight = False
        self.days_since_reweight = 0
        self.pending_reweight_days = 0
        self.precomputed_indicators = precomputed_indicators
        self.weinstein = weinstein
        self.all_assets = df.columns.tolist()
        
        assert all(asset in df.columns for asset in self.all_assets)
        
        # Set up observation space for n_features_per_asset (12) + weights
        n_features_per_asset = 13
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_assets * n_features_per_asset + self.n_assets,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
    def get_technical_indicators(self, asset, date):
        """Get precomputed technical indicators for an asset at a specific date."""
        if date in self.precomputed_indicators and asset in self.precomputed_indicators[date]:
            return self.precomputed_indicators[date][asset]
        else:
            # Return default values if data not available
            return {
                'ma_signals': {'price_over_ma50': 0, 'ma10_over_ma50': 0, 'ma50_over_ma200': 0, 
                               'golden_cross': 0, 'death_cross': 0},
                'rsi': {'value': 50, 'oversold': 0, 'overbought': 0},
                'macd': {'line': 0, 'signal': 0, 'hist': 0, 'hist_change': 0, 
                         'bullish_crossover': 0, 'bearish_crossover': 0},
                'bollinger': {'width': 0, 'position': 0.5, 'lower_breakout': 0, 'upper_breakout': 0},
                'atr': {'value': 0, 'norm_value': 0, 'volatility_change': 0},
                'adx': {'value': 25, 'strong_trend': 0, 'trend_direction': 0}
            }
    
    def get_weinstein_indicators(self, asset, date):
        """Lấy chỉ báo phân tích Stage của Weinstein cho một tài sản tại một ngày cụ thể."""
        if date in self.weinstein and asset in self.weinstein[date]:
            return self.weinstein[date][asset]
        else:
            # Trả về giá trị mặc định nếu không có dữ liệu
            return {
                'weinstein_stage': {
                    'current_stage': 0,
                    'stage_1_base': 0, 
                    'stage_2_advance': 0,
                    'stage_3_top': 0,
                    'stage_4_decline': 0,
                    'stage_1_to_2': 0,
                    'stage_2_to_3': 0,
                    'stage_3_to_4': 0,
                    'stage_4_to_1': 0,
                },
                'ma30': {'value': 0, 'slope': 0, 'price_to_ma30': 0, 'above_ma30': 0, 'ma30_rising': 0, 'ma30_flat': 0},
                'relative_strength': {'value': 1, 'rising': 0, 'strong': 0, 'weak': 0},
                'volume': {'vs_average': 1, 'high_volume': 0, 'low_volume': 0, 'declining_volume': 0},
                'price_patterns': {'near_resistance': 0, 'near_support': 0, 'breakout_potential': 0, 'breakdown_potential': 0},
                'weinstein_signals': {'buying_signal': 0, 'selling_signal': 0, 'hold_signal': 0, 'buy': 0, 'watch': 0, 'sell': 0}
            }
    
    def _calculate_scores(self, metrics):
        """Calculate technical scores for an asset based on its metrics."""
        trend_score = 0
        momentum_score = 0
        volatility_score = 0
        reversal_score = 0
        
        # Trend analysis
        if metrics['ma_signals']['golden_cross']:
            trend_score += 2
        elif metrics['ma_signals']['death_cross']:
            trend_score -= 2
        
        trend_score += metrics['ma_signals']['price_over_ma50'] * 3
        trend_score += metrics['ma_signals']['ma10_over_ma50'] * 2
        trend_score += metrics['ma_signals']['ma50_over_ma200'] * 3
        
        # ADX signals
        if metrics['adx']['strong_trend']:
            trend_multiplier = min(metrics['adx']['value'] / 25, 2)
            trend_score *= trend_multiplier if metrics['adx']['trend_direction'] > 0 else -trend_multiplier
        
        # Momentum analysis - MACD
        if metrics['macd']['bullish_crossover']:
            momentum_score += 2
        elif metrics['macd']['bearish_crossover']:
            momentum_score -= 2
            
        momentum_score += np.sign(metrics['macd']['hist']) * min(abs(metrics['macd']['hist']) * 5, 2)
        momentum_score += np.sign(metrics['macd']['hist_change']) * min(abs(metrics['macd']['hist_change']) * 10, 2)
        
        # Volatility analysis - Bollinger Bands & ATR
        bb_position = metrics['bollinger']['position']
        if bb_position < 0.2:
            reversal_score += 1
        elif bb_position > 0.8:
            reversal_score -= 1
            
        if metrics['atr']['volatility_change'] > 0.2:
            volatility_score += 1
        elif metrics['atr']['volatility_change'] < -0.2:
            volatility_score -= 1
        
        # Reversal signals - RSI & Bollinger breakouts
        if metrics['rsi']['oversold']:
            reversal_score += 2
        elif metrics['rsi']['overbought']:
            reversal_score -= 2
            
        if metrics['bollinger']['lower_breakout']:
            reversal_score += 1.5
        elif metrics['bollinger']['upper_breakout']:
            reversal_score -= 1.5
        
        # Combine scores with weights
        combined_score = (trend_score * 0.4) + (momentum_score * 0.3) + (reversal_score * 0.2) + (volatility_score * 0.1)
        
        # Determine market type
        if trend_score > 1 and momentum_score > 0:
            market_type = "bull"
        elif trend_score < -1 and momentum_score < 0:
            market_type = "bear"
        else:
            market_type = "neutral"
        
        return {
            'technical_score': combined_score,
            'market_type': market_type,
            'analysis': {
                'trend': trend_score, 'momentum': momentum_score,
                'reversal': reversal_score, 'volatility': volatility_score
            },
            'metrics': metrics
        }
    
    def _calculate_scores_w(self, metrics):
        """Calculate scores for an asset based on Weinstein stage analysis metrics."""
        trend_score = 0
        momentum_score = 0
        risk_score = 0
        opportunity_score = 0
        
        # Get current stage and transition signals
        current_stage = metrics['weinstein_stage']['current_stage']
        
        # Trend analysis based on Weinstein stages
        if metrics['weinstein_stage']['stage_2_advance']:
            trend_score += 3  # Strong uptrend in Stage 2
        elif metrics['weinstein_stage']['stage_1_to_2']:
            trend_score += 2  # Emerging uptrend (transition from Stage 1 to 2)
        elif metrics['weinstein_stage']['stage_4_decline']:
            trend_score -= 3  # Strong downtrend in Stage 4
        elif metrics['weinstein_stage']['stage_3_to_4']:
            trend_score -= 2  # Emerging downtrend (transition from Stage 3 to 4)
        elif metrics['weinstein_stage']['stage_1_base']:
            trend_score += 0.5  # Basing/accumulation - slightly positive
        elif metrics['weinstein_stage']['stage_3_top']:
            trend_score -= 0.5  # Distribution/topping - slightly negative
        
        # Add MA30 factors to trend score
        trend_score += metrics['ma30']['price_to_ma30'] * 2
        trend_score += metrics['ma30']['ma30_rising'] * 1.5
        
        # Momentum analysis based on Relative Strength
        if metrics['relative_strength']['strong']:
            momentum_score += 2
        elif metrics['relative_strength']['weak']:
            momentum_score -= 2
        
        momentum_score += (metrics['relative_strength']['value'] - 1) * 3
        
        # Risk analysis - considers stage transitions and price patterns
        if metrics['weinstein_stage']['stage_2_to_3'] or metrics['weinstein_stage']['stage_3_top']:
            risk_score -= 2  # Higher risk at top or transition to distribution phase
        elif metrics['weinstein_stage']['stage_4_decline']:
            risk_score -= 3  # Highest risk in declining phase
        elif metrics['weinstein_stage']['stage_1_base']:
            risk_score += 1  # Lower risk in base-building phase
        
        # Add volume considerations to risk score
        if metrics['volume']['high_volume'] and (metrics['weinstein_stage']['stage_1_to_2'] or metrics['weinstein_stage']['stage_2_advance']):
            risk_score += 1  # High volume in uptrend - positive
        elif metrics['volume']['high_volume'] and (metrics['weinstein_stage']['stage_3_to_4'] or metrics['weinstein_stage']['stage_4_decline']):
            risk_score -= 1  # High volume in downtrend - negative
        
        # Opportunity analysis - considers potential breakouts and transitions
        if metrics['weinstein_stage']['stage_1_to_2']:
            opportunity_score += 3  # Best opportunity at start of uptrend
        elif metrics['weinstein_stage']['stage_4_to_1']:
            opportunity_score += 2  # Good opportunity when downtrend ends
        elif metrics['price_patterns']['breakout_potential']:
            opportunity_score += 2  # Potential breakout
        
        # Penalize breakdown potential
        if metrics['price_patterns']['breakdown_potential']:
            opportunity_score -= 2
        
        # Incorporate specific Weinstein signals
        if metrics['weinstein_signals']['buying_signal']:
            opportunity_score += 2
        elif metrics['weinstein_signals']['selling_signal']:
            opportunity_score -= 2
        elif metrics['weinstein_signals']['hold_signal']:
            trend_score += 1
            
        # Combine scores with weights - adjusted to prioritize stage analysis
        combined_score = (trend_score * 0.35) + (momentum_score * 0.25) + (opportunity_score * 0.25) + (risk_score * 0.15)
        
        # Determine market type based on Weinstein stage
        if current_stage >= 1.5 and current_stage < 3:
            market_type = "bull"
        elif current_stage >= 3.5 or current_stage == 0 or current_stage == 4:
            market_type = "bear"
        else:
            market_type = "neutral"
        
        return {
            'technical_score': combined_score,
            'market_type': market_type,
            'analysis': {
                'trend': trend_score, 
                'momentum': momentum_score,
                'risk': risk_score, 
                'opportunity': opportunity_score
            },
            'stage': current_stage,
            'metrics': metrics
        }
    
    def _evaluate_all_assets(self):
        """Evaluate all assets using technical indicators."""
        all_metrics = {}
        current_date = self.df.index[self.current_step]
        
        for asset in self.all_assets:
            metrics = self.get_weinstein_indicators(asset, current_date)
            all_metrics[asset] = self._calculate_scores_w(metrics)
        
        return all_metrics
    
    def _evaluate_selected_assets(self):
        """Evaluate only the selected assets in portfolio."""
        evaluation = {}
        current_date = self.df.index[self.current_step]
        
        for asset in self.selected_assets:
            metrics = self.get_technical_indicators(asset, current_date)
            evaluation[asset] = self._calculate_scores(metrics)
        
        return evaluation

    def _rebalance_portfolio(self):
        """Change assets in portfolio based on Weinstein stage analysis when assets reach Stage 3+."""
        # Evaluate all assets using Weinstein stage metrics
        all_metrics = self._evaluate_all_assets()
        
        # Identify current portfolio assets that are in Stage 3 or beyond
        assets_to_replace = []
        for i, asset in enumerate(self.selected_assets):
            metrics = all_metrics[asset]
            stage = metrics['metrics']['weinstein_stage']['current_stage']
            stage_3_to_4 = metrics['metrics']['weinstein_stage']['stage_3_to_4'] > 0
            stage_3_top = metrics['metrics']['weinstein_stage']['stage_3_top'] > 0
            stage_4_decline = metrics['metrics']['weinstein_stage']['stage_4_decline'] > 0
            
            # Include assets that are in Stage 3 or higher, or transitioning to Stage 4
            if stage >= 3 or stage_3_to_4 or stage_3_top or stage_4_decline:
                assets_to_replace.append((asset, metrics['technical_score'], i, stage))
        
        # If no assets need replacement, exit early
        if not assets_to_replace:
            return []
        
        # Find potential replacement assets (Stage 1 or Stage 1-to-2 transition)
        available_assets = set(self.all_assets) - set(self.selected_assets)
        replacement_candidates = []
        
        for asset in available_assets:
            metrics = all_metrics[asset]
            stage = metrics['metrics']['weinstein_stage']['current_stage']
            stage_1_base = metrics['metrics']['weinstein_stage']['stage_1_base'] > 0
            stage_1_to_2 = metrics['metrics']['weinstein_stage']['stage_1_to_2'] > 0
            
            # Prioritize assets in Stage 1 or transitioning from Stage 1 to Stage 2
            if (stage < 2 or stage_1_base or stage_1_to_2):
                # Apply additional scoring for better selection
                score = metrics['technical_score']
                
                # Boost score for Stage 1-to-2 transitions (prioritize these)
                if stage_1_to_2:
                    score += 5.0
                
                # Boost for buying signals
                if metrics['metrics']['weinstein_signals']['buying_signal'] > 0:
                    score += 2.0
                    
                # Boost for breakout potential
                if metrics['metrics']['price_patterns']['breakout_potential'] > 0:
                    score += 2.0
                
                replacement_candidates.append((asset, score, stage))
        
        # Sort assets to replace (worst first) and replacement candidates (best first)
        assets_to_replace.sort(key=lambda x: x[1])  # Sort by technical score (ascending)
        replacement_candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by score (descending)
        
        # Execute asset changes
        transactions = []
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        total_transaction_costs = 0
        
        # Replace assets (limited by available replacements)
        num_replacements = min(len(assets_to_replace), len(replacement_candidates))
        
        for i in range(num_replacements):
            old_asset, old_score, old_idx, old_stage = assets_to_replace[i]
            new_asset, new_score, new_stage = replacement_candidates[i]
            
            # Calculate values and costs
            old_value = current_prices[old_idx] * self.shares[old_idx]
            
            # Sell transaction
            sell_cost = old_value * self.transaction_cost
            total_transaction_costs += sell_cost
                
            transactions.append({
                'asset': old_asset, 
                'type': 'sell', 
                'value': old_value,
                'cost': sell_cost, 
                'reason': 'rebalance',
                'old_stage': old_stage,
                'score': old_score
            })
            
            # Buy transaction
            remaining_value = old_value - sell_cost
            buy_cost = remaining_value * self.transaction_cost
            total_transaction_costs += buy_cost
            remaining_value -= buy_cost
            
            # Update assets and shares
            new_price = self.df[new_asset].iloc[self.current_step]
            new_shares = remaining_value / new_price
            
            self.selected_assets[old_idx] = new_asset
            self.shares[old_idx] = new_shares
            
            transactions.append({
                'asset': new_asset, 
                'type': 'buy', 
                'value': remaining_value,
                'cost': buy_cost, 
                'reason': 'rebalance',
                'new_stage': new_stage,
                'score': new_score
            })
        
        # Update portfolio value and weights
        self.portfolio_value = self._calculate_portfolio_value()
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        self.weights = (self.shares * current_prices) / self.portfolio_value
        
        return transactions
    
    def _adjust_weights_based_on_performance(self, action):
        """Adjust weights based on technical indicators and agent action."""
        technical_evaluation = self._evaluate_selected_assets()
        combined_scores = np.zeros(len(self.selected_assets))
        
        for i, asset in enumerate(self.selected_assets):
            tech_score = technical_evaluation[asset]['technical_score']
            market_bonus = 0.5 if technical_evaluation[asset]['market_type'] == "bull" else \
                          -0.5 if technical_evaluation[asset]['market_type'] == "bear" else 0
            
            # 70% technical + 30% market type
            combined_scores[i] = (tech_score * 0.7) + (market_bonus * 0.3)
        
        # Normalize scores to [-1, 1]
        min_score, max_score = np.min(combined_scores), np.max(combined_scores)
        if max_score != min_score:
            normalized_scores = 2 * (combined_scores - min_score) / (max_score - min_score) - 1
        else:
            normalized_scores = np.zeros_like(combined_scores)
        
        # Combine analysis impact (40%) with agent action (60%)
        performance_impact = normalized_scores * 0.4
        action_impact = action * 0.6
        total_adjustment = performance_impact + action_impact
        
        # Apply adjustments to current weights
        current_weights = self.weights.copy()
        proposed_weights = current_weights + total_adjustment
        
        # Ensure weights are at least limit
        proposed_weights = np.maximum(proposed_weights, self.limit)
        
        # Normalize to sum to 1
        proposed_weights = proposed_weights / np.sum(proposed_weights)
        
        # Apply min_pctweight logic
        weight_changes = proposed_weights - current_weights
        decreasing_assets = weight_changes < 0
        valid_changes = np.abs(weight_changes) >= self.min_pctweight
        need_adjustment = decreasing_assets & ~valid_changes
        
        if np.any(need_adjustment):
            # Keep original weights for assets not meeting min_pctweight
            proposed_weights[need_adjustment] = current_weights[need_adjustment]
            
            # Re-normalize remaining weights
            remaining_weight = 1 - np.sum(proposed_weights[need_adjustment])
            mask = ~need_adjustment
            if np.any(mask):
                proposed_weights[mask] = proposed_weights[mask] * remaining_weight / np.sum(proposed_weights[mask])
        
        return proposed_weights, normalized_scores
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.selected_assets = self.initial_assets if self.initial_assets is not None else \
            np.random.choice(self.all_assets, self.n_assets, replace=False)
        
        self.current_step = 0
        self.days_since_reweight = 0
        self.pending_reweight = False
        self.pending_reweight_days = 0
        self.portfolio_value = self.initial_balance
        
        self.weights = np.array([1/self.n_assets] * self.n_assets)
        self.shares = np.zeros(self.n_assets)
        
        initial_prices = self.df[self.selected_assets].iloc[self.current_step].values
        self.shares = (self.portfolio_value * self.weights) / initial_prices
        
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def _get_observation(self):
        """Tạo observation vector với các chỉ báo kỹ thuật được tối ưu."""
        features = []
        current_date = self.df.index[self.current_step]
        
        for asset in self.selected_assets:
            metrics = self.get_technical_indicators(asset, current_date)
            
            asset_features = [
                # MA signals - Xu hướng
                metrics['ma_signals']['price_over_ma50'],
                metrics['ma_signals']['ma10_over_ma50'], 
                metrics['ma_signals']['ma50_over_ma200'],
                metrics['ma_signals']['golden_cross'],
                
                # RSI - Tình trạng quá mua/quá bán
                metrics['rsi']['value'] / 100,  # Chuẩn hóa về range [0, 1]
                
                # MACD - Momentum
                np.sign(metrics['macd']['hist']),
                metrics['macd']['bullish_crossover'] - metrics['macd']['bearish_crossover'],
                
                # Bollinger Bands - Biến động và vị trí giá
                metrics['bollinger']['position'],
                metrics['bollinger']['width'],
                
                # ATR - Biến động
                metrics['atr']['norm_value'],
                
                # ADX - Sức mạnh xu hướng
                metrics['adx']['value'] / 100,  # Chuẩn hóa về range [0, 1]
                metrics['adx']['trend_direction'],
                
                # Giá của tài sản
                self.df[asset].iloc[self.current_step]
            ]
            
            features.extend(asset_features)
        
        # Thêm thông tin trọng số hiện tại
        features.extend(self.weights)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_portfolio_value(self):
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        return np.sum(self.shares * current_prices)
    
    def calculate_reward(self, old_value, new_value, info):
        """Calculate reward as portfolio return."""
        reward = (new_value - old_value) / old_value
        return reward * 100
    
    def _process_weight_change(self, new_weights, performance_scores):
        """Process weight changes with transactions."""
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        current_value = self._calculate_portfolio_value()
        
        old_values = current_prices * self.shares
        new_values = current_value * new_weights
        value_changes = new_values - old_values
        
        transactions = []
        total_transaction_costs = 0
        
        for i, asset in enumerate(self.selected_assets):
            if value_changes[i] != 0:
                transaction_value = abs(value_changes[i])
                transaction_cost = transaction_value * self.transaction_cost
                total_transaction_costs += transaction_cost
                
                actual_value = transaction_value - transaction_cost
                
                transactions.append({
                    'asset': asset,
                    'type': 'buy' if value_changes[i] > 0 else 'sell',
                    'value': transaction_value,
                    'cost': transaction_cost,
                    'actual_value': actual_value,
                    'old_weight': self.weights[i],
                    'new_weight': new_weights[i],
                    'performance_score': performance_scores[i],
                    'reason': 'reweight'
                })
        
        # Update portfolio after costs
        current_value -= total_transaction_costs
        self.portfolio_value = current_value
        
        # Recalculate shares based on new weights
        for i, asset in enumerate(self.selected_assets):
            self.shares[i] = (current_value * new_weights[i]) / current_prices[i]
        
        self.weights = new_weights
        
        return transactions, total_transaction_costs
    
    def step(self, action):
        """Execute one step in the environment."""
        # Save old state for reward calculation
        old_step = self.current_step
        old_value = self._calculate_portfolio_value()
        old_weights = self.weights.copy()
        
        # Update state
        self.current_step += 1
        self.days_since_reweight += 1
        
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Initialize info dictionary
        info = {
            'rebalanced': False,
            'reweighted': False,
            'transactions': [],
            'metrics': {},
            'old_weights': old_weights,
            'current_value': old_value,
            'transaction_costs': 0,
            'trading_summary': {
                'total_trades': 0, 'buy_trades': 0, 
                'sell_trades': 0, 'total_volume': 0
            }
        }
        
        # Check for rebalance based on asset stages
        # We now rebalance whenever assets reach Stage 3 or beyond
        transactions = self._rebalance_portfolio()
        if transactions:  # Only mark as rebalanced if transactions occurred
            info['transactions'].extend(transactions)
            info['rebalanced'] = True
            
            self._update_trading_summary(info, transactions)
            info['transaction_costs'] += sum(t['cost'] for t in transactions)
            
            # Delay reweight if needed after a rebalance
            if self.days_since_reweight >= self.reweight:
                self.pending_reweight = True
                self.pending_reweight_days = 2
        
        # Handle reweighting (same as before)
        should_reweight = (
            (self.pending_reweight and self.pending_reweight_days <= 0) or
            (self.days_since_reweight >= self.reweight and not self.pending_reweight)
        )
        
        if should_reweight:
            self.days_since_reweight = 0
            self.pending_reweight = False
            
            new_weights, performance_scores = self._adjust_weights_based_on_performance(action)
            reweight_transactions, costs = self._process_weight_change(new_weights, performance_scores)
            
            info['transactions'].extend(reweight_transactions)
            info['reweighted'] = True
            info['transaction_costs'] += costs
            
            self._update_trading_summary(info, reweight_transactions)
        
        # Decrement pending reweight days if needed
        if self.pending_reweight:
            self.pending_reweight_days -= 1
        
        # Calculate new portfolio value and reward
        new_value = self._calculate_portfolio_value()
        reward = self.calculate_reward(old_value, new_value, info)
        
        # Add performance info
        info['portfolio_return'] = (new_value - old_value) / old_value
        info['portfolio_value'] = new_value
        info['cumulative_return'] = (new_value - self.initial_balance) / self.initial_balance
        
        # Add detailed asset info
        asset_info = {}
        current_prices = self.df[self.selected_assets].iloc[self.current_step].values
        for i, asset in enumerate(self.selected_assets):
            asset_value = current_prices[i] * self.shares[i]
            asset_return = (current_prices[i] - self.df[asset].iloc[old_step]) / self.df[asset].iloc[old_step]
            
            asset_info[asset] = {
                'price': current_prices[i],
                'shares': self.shares[i],
                'weight': self.weights[i],
                'value': asset_value,
                'daily_return': asset_return,
                'contribution': (asset_value / new_value) * asset_return
            }
        info['assets'] = asset_info
        
        observation = self._get_observation()
        
        return observation, reward, done, truncated, info

    def _update_trading_summary(self, info, transactions):
        """Helper method to update trading summary info."""
        for t in transactions:
            info['trading_summary']['total_trades'] += 1
            info['trading_summary']['total_volume'] += t['value']
            info['trading_summary']['buy_trades' if t['type'] == 'buy' else 'sell_trades'] += 1



















