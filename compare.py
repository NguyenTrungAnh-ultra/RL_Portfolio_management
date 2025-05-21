import random
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
from vnstock3 import Vnstock
from pypfopt.efficient_frontier import EfficientFrontier


def calculate_metrics(portfolio_values):
    """Calculate various portfolio performance metrics."""
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    
    # Basic return metrics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    
    # Risk metrics
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # Drawdown analysis
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def compare(rl_results, mean_var_df, vn_index_df=None, how='sharpe'):
    """
    So sánh kết quả của agent RL với phương pháp Mean Variance Optimization và VN-Index.
    
    Parameters:
    -----------
    rl_results : list
        Kết quả từ hàm test_model, chứa thông tin về hiệu suất của agent RL
    mean_var_df : pandas.DataFrame
        DataFrame chứa giá trị portfolio của phương pháp Mean Variance Optimization,
        với index là datetime và cột 'Mean Var' chứa giá trị portfolio
    vn_index_df : pandas.DataFrame, optional
        DataFrame chứa dữ liệu OHLCV của VN-Index, với index là datetime
    how : str, default='sharpe'
        Tiêu chí để chọn episode tốt nhất:
        'return': tìm episode có tổng lợi nhuận cao nhất
        'risk': tìm episode có biến động (volatility) thấp nhất
        'sharpe': tìm episode có Sharpe Ratio cao nhất
        'random': chọn ngẫu nhiên một episode
        
    Returns:
    --------
    dict
        Dictionary chứa các biểu đồ so sánh và phân tích số liệu
    """
    # Chọn episode dựa trên tiêu chí
    if how == 'return':
        best_episode_idx = np.argmax([r['metrics']['total_return'] for r in rl_results])
        criterion = 'Total Return'
    elif how == 'risk':
        best_episode_idx = np.argmin([r['metrics']['volatility'] for r in rl_results])
        criterion = 'Lowest Volatility'
    elif how == 'sharpe':
        best_episode_idx = np.argmax([r['metrics']['sharpe_ratio'] for r in rl_results])
        criterion = 'Sharpe Ratio'
    elif how == 'random':
        best_episode_idx = random.randint(0, len(rl_results) - 1)
        criterion = 'Random Selection'
    else:
        raise ValueError("Parameter 'how' must be one of: 'return', 'risk', 'sharpe', 'random'")
    
    print(f"Selected episode {best_episode_idx} based on {criterion}")
    best_rl_result = rl_results[best_episode_idx]
    
    # Lấy metrics từ episode tốt nhất
    rl_metrics = best_rl_result['metrics']
    
    # Đảm bảo dữ liệu mean_var_df có index là datetime
    if not isinstance(mean_var_df.index[0], (datetime.datetime, np.datetime64, pd.Timestamp)):
        mean_var_df.index = pd.date_range(start='2020-01-01', periods=len(mean_var_df))
    
    # Chuẩn bị dữ liệu cho Mean Variance
    mean_var_returns = mean_var_df['Mean Var'].pct_change().dropna()
    
    # Tính toán các metrics cho Mean Variance
    portfolio_values = mean_var_df['Mean Var'].values
    mean_var_metrics = calculate_metrics(portfolio_values)
    
    # Chuẩn bị dữ liệu cho VN-Index nếu có
    vn_index_metrics = None
    if vn_index_df is not None:
        # Đảm bảo VN-Index có index là datetime
        if not isinstance(vn_index_df.index[0], (datetime.datetime, np.datetime64, pd.Timestamp)):
            vn_index_df.index = pd.date_range(start='2020-01-01', periods=len(vn_index_df))
        
        # Lấy giá đóng cửa (close) từ dữ liệu OHLCV
        if 'close' in vn_index_df.columns:
            vn_index_series = vn_index_df['close']
        else:
            # Sử dụng cột đầu tiên nếu không có cột 'close'
            vn_index_series = vn_index_df.iloc[:, 0]
        
        # Tính toán các metrics cho VN-Index (sử dụng cùng khoảng thời gian với mean_var_df)
        # Lấy dữ liệu VN-Index cho cùng khoảng thời gian với mean_var_df
        common_dates = mean_var_df.index.intersection(vn_index_df.index)
        if len(common_dates) > 0:
            vn_index_aligned = vn_index_series.loc[common_dates].values
            vn_index_metrics = calculate_metrics(vn_index_aligned)
            # Chuẩn bị dữ liệu returns cho các phép tính sau này
            vn_index_returns = pd.Series(vn_index_aligned).pct_change().dropna()
    
    # Tạo DataFrame để so sánh
    comparison_data = {
        'Metric': ['Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        'RL Agent': [rl_metrics['total_return'], rl_metrics['annual_return'], 
                     rl_metrics['volatility'], rl_metrics['sharpe_ratio'], 
                     rl_metrics['max_drawdown']],
        'Mean Variance': [mean_var_metrics['total_return'], mean_var_metrics['annual_return'],
                          mean_var_metrics['volatility'], mean_var_metrics['sharpe_ratio'],
                          mean_var_metrics['max_drawdown']]
    }
    
    if vn_index_metrics:
        comparison_data['VN-Index'] = [vn_index_metrics['total_return'], vn_index_metrics['annual_return'],
                                      vn_index_metrics['volatility'], vn_index_metrics['sharpe_ratio'],
                                      vn_index_metrics['max_drawdown']]
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Chuẩn bị dữ liệu cho biểu đồ so sánh giá trị danh mục
    rl_values = best_rl_result['portfolio_values']
    
    # Xác định khoảng thời gian chung
    min_length = min(len(rl_values), len(mean_var_df))
    dates = mean_var_df.index[:min_length]
    
    # Chuẩn hóa giá trị ban đầu về 1 để dễ so sánh
    normalized_rl = np.array(rl_values[:min_length]) / rl_values[0]
    normalized_mean_var = mean_var_df['Mean Var'].values[:min_length] / mean_var_df['Mean Var'].values[0]
    
    # Chuẩn bị DataFrame cho biểu đồ
    performance_data = {
        'Date': dates,
        'RL Agent': normalized_rl,
        'Mean Variance': normalized_mean_var
    }
    
    # Thêm VN-Index nếu có, đảm bảo khớp với khoảng thời gian
    if vn_index_df is not None and len(common_dates) > 0:
        # Cắt data để chỉ giữ lại những ngày khớp với mean_var_df
        common_dates_in_range = [d for d in dates if d in common_dates]
        
        if len(common_dates_in_range) > 0:
            # Lấy giá trị VN-Index cho các ngày khớp
            vn_index_values = vn_index_series.loc[common_dates_in_range].values
            
            # Chuẩn hóa giá trị
            if len(vn_index_values) > 0:
                normalized_vn_index = vn_index_values / vn_index_values[0]
                
                # Tạo một Series mới với index là tất cả các ngày trong dates
                temp_series = pd.Series(index=dates)
                
                # Gán giá trị chuẩn hóa cho những ngày khớp
                for i, date in enumerate(common_dates_in_range):
                    temp_series.loc[date] = normalized_vn_index[i]
                
                # Thêm vào performance_data
                # Chỉ sử dụng những giá trị không phải NaN
                valid_indices = ~temp_series.isna()
                if valid_indices.any():
                    performance_data['VN-Index'] = temp_series.fillna(method='ffill').values
    
    performance_df = pd.DataFrame(performance_data)
    
    # Tạo biểu đồ hiệu suất so sánh
    fig1 = px.line(
        performance_df, 
        x='Date', 
        y=[col for col in performance_df.columns if col != 'Date'],
        title=f'So sánh hiệu suất: RL Agent vs Mean Variance' + (' vs VN-Index' if 'VN-Index' in performance_df.columns else ''),
        labels={'value': 'Giá trị danh mục (chuẩn hóa)', 'variable': 'Phương pháp'},
        template='plotly_white'
    )
    
    # Tạo biểu đồ so sánh các metrics
    fig3 = px.bar(
        comparison_df,
        x='Metric',
        y=[col for col in comparison_df.columns if col != 'Metric'],
        barmode='group',
        title=f'So sánh các chỉ số hiệu suất',
        labels={'value': 'Giá trị', 'variable': 'Phương pháp'},
        template='plotly_white'
    )
    
    # Tính drawdowns
    # RL Agent
    rl_returns = pd.Series(best_rl_result['portfolio_values']).pct_change().dropna()
    rl_cum_returns = (1 + rl_returns).cumprod()
    rl_rolling_max = rl_cum_returns.expanding().max()
    rl_drawdowns = (rl_cum_returns / rl_rolling_max - 1) * 100
    
    # Mean Variance
    mean_var_cum_returns = (1 + mean_var_returns).cumprod()
    mean_var_rolling_max = mean_var_cum_returns.expanding().max()
    mean_var_drawdowns = (mean_var_cum_returns / mean_var_rolling_max - 1) * 100
    
    # Chuẩn bị dữ liệu drawdown
    min_drawdown_length = min(len(rl_drawdowns), len(mean_var_drawdowns), min_length-1)
    drawdown_data = {
        'Date': dates[1:min_drawdown_length+1],
        'RL Agent': rl_drawdowns.values[:min_drawdown_length],
        'Mean Variance': mean_var_drawdowns.values[:min_drawdown_length]
    }
    
    # Thêm VN-Index drawdown nếu có
    if vn_index_df is not None and 'VN-Index' in performance_df.columns:
        # Tính drawdown cho VN-Index sử dụng dữ liệu đã căn chỉnh
        try:
            vn_index_aligned_returns = performance_df['VN-Index'].pct_change().dropna()
            vn_cum_returns = (1 + vn_index_aligned_returns).cumprod()
            vn_rolling_max = vn_cum_returns.expanding().max()
            vn_drawdowns = (vn_cum_returns / vn_rolling_max - 1) * 100
            
            # Đảm bảo độ dài tương thích
            if len(vn_drawdowns) >= min_drawdown_length:
                drawdown_data['VN-Index'] = vn_drawdowns.values[:min_drawdown_length]
        except Exception as e:
            print(f"Không thể tính drawdown cho VN-Index: {e}")
    
    drawdown_df = pd.DataFrame(drawdown_data)
    
    # Tạo biểu đồ drawdown
    fig4 = px.line(
        drawdown_df,
        x='Date',
        y=[col for col in drawdown_df.columns if col != 'Date'],
        title=f'So sánh Drawdown (%)',
        labels={'value': 'Drawdown (%)', 'variable': 'Phương pháp'},
        template='plotly_white'
    )
    
    # Thay đổi màu sắc cho thống nhất
    colors = {'RL Agent': '#2e7cee', 'Mean Variance': '#fc6955', 'VN-Index': '#00CC96'}
    for fig in [fig1, fig4]:
        for i, trace in enumerate(fig.data):
            name = trace.name
            if name in colors:
                fig.data[i].line.color = colors[name]
    
    # Tạo biểu đồ phân phối lợi nhuận
    # Chuẩn bị dữ liệu
    rl_daily_returns = rl_returns * 100  # Chuyển thành phần trăm
    mean_var_daily_returns = mean_var_returns * 100  # Chuyển thành phần trăm
    
    # Giới hạn số lượng để tương đồng
    min_returns_length = min(len(rl_daily_returns), len(mean_var_daily_returns))
    rl_daily_returns = rl_daily_returns[:min_returns_length]
    mean_var_daily_returns = mean_var_daily_returns[:min_returns_length]
    
    # Tính số cột cần dùng trong subplot
    num_columns = 2
    subplot_titles = ['RL Agent', 'Mean Variance']
    
    if vn_index_df is not None and 'VN-Index' in performance_df.columns:
        num_columns = 3
        subplot_titles.append('VN-Index')
        # Sử dụng dữ liệu VN-Index đã căn chỉnh
        vn_daily_returns = performance_df['VN-Index'].pct_change().dropna() * 100
        if len(vn_daily_returns) > min_returns_length:
            vn_daily_returns = vn_daily_returns[:min_returns_length]
    
    # Tạo subplot
    fig5 = make_subplots(rows=1, cols=num_columns, subplot_titles=subplot_titles)
    
    # Thêm histogram cho RL Agent
    fig5.add_trace(
        go.Histogram(
            x=rl_daily_returns,
            name='RL Agent',
            marker_color=colors['RL Agent'],
            opacity=0.7,
            nbinsx=30
        ),
        row=1, col=1
    )
    
    # Thêm histogram cho Mean Variance
    fig5.add_trace(
        go.Histogram(
            x=mean_var_daily_returns,
            name='Mean Variance',
            marker_color=colors['Mean Variance'],
            opacity=0.7,
            nbinsx=30
        ),
        row=1, col=2
    )
    
    # Thêm histogram cho VN-Index nếu có
    if vn_index_df is not None and 'VN-Index' in performance_df.columns and len(vn_daily_returns) > 0:
        fig5.add_trace(
            go.Histogram(
                x=vn_daily_returns,
                name='VN-Index',
                marker_color=colors['VN-Index'],
                opacity=0.7,
                nbinsx=30
            ),
            row=1, col=3
        )
    
    fig5.update_layout(
        title_text=f'Phân phối lợi nhuận hàng ngày (%)',
        template='plotly_white',
        showlegend=False
    )
    
    # Tóm tắt kết quả so sánh
    comparison_summary = {
        'performance_plot': fig1,
        'metrics_comparison_plot': fig3,
        'drawdown_comparison_plot': fig4,
        'return_distribution_plot': fig5,
        'metrics_table': comparison_df,
        'best_episode_number': best_episode_idx,
        'selection_criterion': criterion
    }
    
    return comparison_summary

def get_mean_variance(train_df, test_df, balance=100_000_000):
    def StockReturnsComputing(StockPrice, Rows, Columns):
        StockReturn = np.zeros([Rows-1, Columns])
        for j in range(Columns):
            for i in range(Rows-1):
                StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100

        return StockReturn

    #compute asset returns
    arStockPrices = np.asarray(train_df)
    [Rows, Cols]=arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

    #compute mean returns and variance covariance matrix of returns
    meanReturns = np.mean(arReturns, axis = 0)
    covReturns = np.cov(arReturns, rowvar=False)

    #set precision for printing results
    np.set_printoptions(precision=3, suppress = True)

    ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()
    mvo_weights = np.array([balance * cleaned_weights_mean[i] for i in range(len(test_df.columns))])

    LastPrice = np.array([1/p for p in train_df.tail(1).to_numpy()[0]])
    Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

    Portfolio_Assets = test_df @ Initial_Portfolio
    MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
    
    return MVO_result

def get_benchmark(start='2018-08-03', market='VNINDEX'):
    stock = Vnstock().stock(symbol=market, source='VCI')
    vn_index = stock.quote.history(symbol=market, start=start, end=str(datetime.date.today())).set_index('time')
    vn_index.index.name = 'date'
    
    return vn_index


