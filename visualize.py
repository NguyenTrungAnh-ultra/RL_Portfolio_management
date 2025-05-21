import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
import random

def test_model(model, env, episodes=100, cash=False):
    """
    Test the trained model over multiple episodes and collect detailed metrics.
    
    Args:
        model: The trained model to test
        env: The environment to test in
        episodes: Number of episodes to run (default: 100)
        cash: Whether to save cash from environment (default: True)
    
    Returns:
        List of results for all episodes
    """
    all_episodes_results = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        
        # Track various metrics
        portfolio_values = []
        weight_history = []
        asset_history = []  # Track which assets are in portfolio
        rebalance_history = []  # Track rebalance events
        reweight_history = []   # Track reweight events
        cash_history = []  # Track cash positions
        info_history = []  # Store all info dictionaries
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Record basic metrics
            portfolio_values.append(info['portfolio_value'])
            info_history.append(info)  # Store complete info dictionary
            
            # Correctly extract weights as float values
            current_weights = {k: float(v['weight']) for k, v in info['assets'].items()}
            weight_history.append(current_weights)
            asset_history.append(list(info['assets'].keys()))
            
            # Record cash position (only if cash=True)
            if cash:
                cash_ratio = info.get('cash_ratio', env.min_cash_ratio)  # Use min_cash_ratio as fallback
                cash_history.append(cash_ratio)
            
            # Record rebalance/reweight events with full portfolio state
            if info['rebalanced']:
                rebalance_event = {
                    'day': len(portfolio_values),
                    'old_portfolio': asset_history[-2],  # Portfolio before change
                    'new_portfolio': asset_history[-1],  # Portfolio after change
                    'transactions': [t for t in info['transactions'] if t['reason'] == 'rebalance'],
                    'weights': current_weights,  # Add current weights
                }
                
                # Only add cash ratio if cash=True
                if cash:
                    rebalance_event['cash_ratio'] = cash_ratio  # Add cash ratio at rebalance
                
                rebalance_history.append(rebalance_event)
                
            if info['reweighted']:
                reweight_event = {
                    'day': len(portfolio_values),
                    'weights': current_weights,
                    'transactions': [t for t in info['transactions'] if t['reason'] == 'cash_adjustment'],
                }
                
                # Only add cash ratio if cash=True
                if cash:
                    reweight_event['cash_ratio'] = cash_ratio  # Add cash ratio at reweight
                
                reweight_history.append(reweight_event)
            
            done = done or truncated
        
        # Calculate episode metrics
        episode_metrics = calculate_metrics(portfolio_values)
        
        result = {
            'episode': episode,
            'portfolio_values': portfolio_values,
            'weight_history': weight_history,
            'asset_history': asset_history,
            'rebalance_history': rebalance_history,
            'reweight_history': reweight_history,
            'metrics': episode_metrics
        }
        
        # Only add cash history if cash=True
        if cash:
            result['cash_history'] = cash_history  # Add cash history
        
        result['info'] = info_history  # Add complete info history
        
        all_episodes_results.append(result)
    
    return all_episodes_results

def analyze_results(test_results, dates, cash=False):
    """
    Analyze and visualize portfolio performance with improved weight visualization.
    If cash=True, show cash allocation from environment.
    
    Args:
        test_results: Results from test_model function
        dates: Array of dates corresponding to the test period
        cash: Whether to include cash visualization (default: True)
    """
    # Create figure with secondary y-axis
    subtitle = 'Asset & Cash Allocation Over Time' if cash else 'Asset Allocation Over Time'
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Portfolio Value', subtitle),
                       row_heights=[0.4, 0.6],
                       vertical_spacing=0.15)
    
    # Plot portfolio value
    portfolio_values = pd.Series(test_results['portfolio_values'], 
                               index=dates[:len(test_results['portfolio_values'])])
    fig.add_trace(
        go.Scatter(x=portfolio_values.index, y=portfolio_values.values,
                  name='Portfolio Value', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add rebalance event markers
    for event in test_results['rebalance_history']:
        day = event['day']
        date = dates[day] if cash else dates[day]
        value = test_results['portfolio_values'][day-1] if cash else test_results['portfolio_values'][day]
        fig.add_trace(
            go.Scatter(x=[date], y=[value],
                      mode='markers',
                      marker=dict(symbol='star', size=12, color='red'),
                      name='Rebalance Event',
                      showlegend=False),
            row=1, col=1
        )
    
    # Convert weight history to DataFrame
    weight_df = pd.DataFrame(test_results['weight_history'])
    weight_df.index = dates[:len(test_results['weight_history'])]
    
    # Add cash weight from cash_history if cash=True
    if cash and 'cash_history' in test_results:
        weight_df['CASH'] = pd.Series(test_results['cash_history'], 
                                     index=dates[:len(test_results['cash_history'])])
    
    # Fill NaN values with 0
    weight_df = weight_df.fillna(0)
    
    # Resample weights to weekly data for clearer visualization
    weight_df_weekly = weight_df.resample('W').last()
    
    # Create a stacked bar chart for weights
    colors = px.colors.qualitative.Set3  # Using a distinct color palette
    cash_color = '#2E8B57'  # Special color for cash (Sea Green)
    
    if cash and 'CASH' in weight_df_weekly.columns:
        # First plot all assets except cash
        for idx, asset in enumerate(weight_df_weekly.columns[:-1]):  # Exclude CASH
            fig.add_trace(
                go.Bar(
                    x=weight_df_weekly.index,
                    y=weight_df_weekly[asset] * 100,  # Convert to percentage
                    name=asset,
                    marker_color=colors[idx % len(colors)],
                    text=[f"{value:.1f}%" if value > 5 else "" 
                          for value in weight_df_weekly[asset] * 100],
                    textposition="inside",
                ),
                row=2, col=1
            )
        
        # Then plot cash weight with a distinct color
        fig.add_trace(
            go.Bar(
                x=weight_df_weekly.index,
                y=weight_df_weekly['CASH'] * 100,
                name='CASH',
                marker_color=cash_color,
                text=[f"{value:.1f}%" if value > 5 else "" 
                      for value in weight_df_weekly['CASH'] * 100],
                textposition="inside",
            ),
            row=2, col=1
        )
    else:
        # Plot all assets (no special handling for cash)
        for idx, asset in enumerate(weight_df_weekly.columns):
            fig.add_trace(
                go.Bar(
                    x=weight_df_weekly.index,
                    y=weight_df_weekly[asset] * 100,  # Convert to percentage
                    name=asset,
                    marker_color=colors[idx % len(colors)],
                    text=[f"{value:.1f}%" if value > 5 else "" 
                          for value in weight_df_weekly[asset] * 100],
                    textposition="inside",
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='stack',
    )
    
    # Update axes labels and format
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value (VND)", row=1, col=1)
    fig.update_yaxes(title_text="Allocation (%)", range=[0, 100], row=2, col=1)
    
    # Add gridlines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Update bar chart layout
    fig.update_layout(
        bargap=0.1,  # gap between bars
        bargroupgap=0.1  # gap between bar groups
    )
    
    fig.show()

def multitest_analysis(test_results, dates):
    """Analyze results across multiple episodes and create visualizations."""
    # Extract metrics from all episodes
    returns = [result['metrics']['total_return'] for result in test_results]
    risks = [result['metrics']['volatility'] for result in test_results]
    sharpe_ratios = [result['metrics']['sharpe_ratio'] for result in test_results]
    max_drawdowns = [result['metrics']['max_drawdown'] for result in test_results]
    
    # Calculate aggregate statistics
    mean_return = np.mean(returns)
    mean_risk = np.mean(risks)
    mean_sharpe = np.mean(sharpe_ratios)
    mean_max_drawdown = np.mean(max_drawdowns)
    
    # Print aggregate statistics
    print("\n=== Multi-Episode Analysis ===")
    print(f"Number of episodes: {len(test_results)}")
    print(f"Mean Return: {mean_return:.2%}")
    print(f"Mean Risk (Volatility): {mean_risk:.2%}")
    print(f"Mean Sharpe Ratio: {mean_sharpe:.2f}")
    print(f"Mean Maximum Drawdown: {mean_max_drawdown:.2%}")
    print("\nDistribution Statistics:")
    print(f"Return Std Dev: {np.std(returns):.2%}")
    print(f"Risk Std Dev: {np.std(risks):.2%}")
    
    # Create distribution plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Return Distribution',
            'Risk Distribution',
            'MaxDrawdown Distribution',  # Changed from Return vs Risk Scatter
            'Sharpe Ratio Distribution'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Return distribution
    fig.add_trace(
        go.Histogram(
            x=[r * 100 for r in returns],
            name='Returns',
            nbinsx=50,
            marker_color='blue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Risk distribution
    fig.add_trace(
        go.Histogram(
            x=[r * 100 for r in risks],
            name='Risks',
            nbinsx=50,
            marker_color='red',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # MaxDrawdown distribution (replacing Return vs Risk scatter)
    fig.add_trace(
        go.Histogram(
            x=[r * 100 for r in max_drawdowns],
            name='MaxDrawdowns',
            nbinsx=50,
            marker_color='purple',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Sharpe ratio distribution
    fig.add_trace(
        go.Histogram(
            x=sharpe_ratios,
            name='Sharpe Ratios',
            nbinsx=50,
            marker_color='green',
            opacity=0.7
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=False,
        title_text="Portfolio Performance Distribution Analysis",
        title_x=0.5
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Return (%)", row=1, col=1)
    fig.update_xaxes(title_text="Risk (%)", row=1, col=2)
    fig.update_xaxes(title_text="MaxDrawdown (%)", row=2, col=1)  # Updated
    fig.update_xaxes(title_text="Sharpe Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)  # Updated
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    # Add mean lines to histograms
    fig.add_vline(x=mean_return * 100, line_dash="dash", line_color="blue", row=1, col=1)
    fig.add_vline(x=mean_risk * 100, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_vline(x=mean_max_drawdown * 100, line_dash="dash", line_color="purple", row=2, col=1)  # Added
    fig.add_vline(x=mean_sharpe, line_dash="dash", line_color="green", row=2, col=2)
    
    fig.show()
    
    # Create box plots with MaxDrawdown added
    fig2 = make_subplots(
        rows=1, cols=4,  # Changed to 4 columns to include MaxDrawdown
        subplot_titles=('Return Distribution', 'Risk Distribution', 'Sharpe Ratio Distribution', 'MaxDrawdown Distribution'),
        specs=[[{'type': 'box'}, {'type': 'box'}, {'type': 'box'}, {'type': 'box'}]]
    )
    
    fig2.add_trace(
        go.Box(y=[r * 100 for r in returns], name='Returns', marker_color='blue'),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Box(y=[r * 100 for r in risks], name='Risks', marker_color='red'),
        row=1, col=2
    )
    
    fig2.add_trace(
        go.Box(y=sharpe_ratios, name='Sharpe Ratios', marker_color='green'),
        row=1, col=3
    )
    
    fig2.add_trace(
        go.Box(y=[r * 100 for r in max_drawdowns], name='MaxDrawdowns', marker_color='purple'),
        row=1, col=4
    )
    
    fig2.update_layout(
        height=400,
        width=1200,
        title_text="Distribution Box Plots",
        title_x=0.5,
        showlegend=False
    )
    
    fig2.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig2.update_yaxes(title_text="Risk (%)", row=1, col=2)
    fig2.update_yaxes(title_text="Sharpe Ratio", row=1, col=3)
    fig2.update_yaxes(title_text="MaxDrawdown (%)", row=1, col=4)
    
    fig2.show()

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
