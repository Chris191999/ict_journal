import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_basic_metrics(df):
    """Calculate basic trading metrics from trade DataFrame"""
    if df.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'net_pnl': 0,
            'total_pnl': 0,
            'profit_factor': 0
        }
    
    # Filter for completed trades
    completed = df[df['status'] == 'Completed'].copy()
    
    # Basic counts
    total_trades = len(completed)
    winning_trades = len(completed[completed['pnl'] > 0])
    losing_trades = len(completed[completed['pnl'] < 0])
    
    # Profit metrics
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        net_pnl = completed['pnl'].sum()
        
        # Average metrics
        avg_win = completed[completed['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = completed[completed['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Extreme values
        largest_win = completed['pnl'].max() if not completed.empty else 0
        largest_loss = completed['pnl'].min() if not completed.empty else 0
        
        # Total positive and negative PnL
        total_profit = completed[completed['pnl'] > 0]['pnl'].sum()
        total_loss = abs(completed[completed['pnl'] < 0]['pnl'].sum())
        
        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'net_pnl': net_pnl,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor
        }
    else:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'net_pnl': 0,
            'total_profit': 0,
            'total_loss': 0,
            'profit_factor': 0
        }

def calculate_advanced_metrics(df):
    """Calculate advanced trading metrics including ICT-specific metrics"""
    if df.empty:
        return {
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'avg_rrr': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'expectancy': 0,
            'recovery_factor': 0,
            'kelly_percentage': 0,
            'efficiency_ratio': 0
        }
    
    # Filter for completed trades
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return {
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'avg_rrr': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'expectancy': 0,
            'recovery_factor': 0,
            'kelly_percentage': 0,
            'efficiency_ratio': 0
        }
    
    # Ensure data is sorted by date
    completed = completed.sort_values('date')
    
    # Calculate daily returns (if date frequency allows)
    if 'date' in completed.columns:
        completed['date'] = pd.to_datetime(completed['date'])
        
        # Create a cumulative equity curve
        completed['cumulative_pnl'] = completed['pnl'].cumsum()
        
        # Calculate drawdown
        completed['peak'] = completed['cumulative_pnl'].cummax()
        completed['drawdown'] = completed['cumulative_pnl'] - completed['peak']
        
        max_drawdown = abs(completed['drawdown'].min()) if not completed.empty else 0
        
        # Drawdown percentage (relative to peak)
        if not completed.empty and completed['peak'].max() > 0:
            max_drawdown_pct = (abs(completed['drawdown'].min()) / completed['peak'].max()) * 100
        else:
            max_drawdown_pct = 0
            
        # Sharpe ratio (simplified without risk-free rate)
        if len(completed) > 1:
            daily_returns = completed.groupby(completed['date'].dt.date)['pnl'].sum()
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
    else:
        max_drawdown = 0
        max_drawdown_pct = 0
        sharpe_ratio = 0
    
    # Risk-Reward Ratio (if risk data is available)
    if 'risk' in completed.columns and completed['risk'].sum() != 0:
        completed['rrr'] = completed['pnl'] / completed['risk'].abs()
        avg_rrr = completed['rrr'].mean()
    else:
        avg_rrr = 0
    
    # Consecutive wins/losses
    if not completed.empty:
        completed['win'] = completed['pnl'] > 0
        result_changes = completed['win'].diff().ne(0).cumsum()
        streak_groups = completed.groupby(result_changes)
        
        max_win_streak = 0
        max_loss_streak = 0
        
        for _, group in streak_groups:
            if group['win'].iloc[0]:
                max_win_streak = max(max_win_streak, len(group))
            else:
                max_loss_streak = max(max_loss_streak, len(group))
    else:
        max_win_streak = 0
        max_loss_streak = 0
    
    # Expectancy
    total_trades = len(completed)
    if total_trades > 0:
        win_rate = len(completed[completed['pnl'] > 0]) / total_trades
        avg_win = completed[completed['pnl'] > 0]['pnl'].mean() if len(completed[completed['pnl'] > 0]) > 0 else 0
        avg_loss = abs(completed[completed['pnl'] < 0]['pnl'].mean()) if len(completed[completed['pnl'] < 0]) > 0 else 0
        
        if avg_loss > 0:
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        else:
            expectancy = win_rate * avg_win
    else:
        expectancy = 0
    
    # Recovery Factor
    if max_drawdown > 0:
        recovery_factor = completed['pnl'].sum() / max_drawdown
    else:
        recovery_factor = 0 if completed['pnl'].sum() <= 0 else float('inf')
    
    # Kelly Percentage
    if avg_loss > 0:
        kelly_percentage = (win_rate - ((1 - win_rate) / (avg_win / avg_loss))) * 100
    else:
        kelly_percentage = 0
    
    # Efficiency Ratio (for ICT specific)
    # This measures how efficiently the trader captures the move
    if 'planned_take_profit' in completed.columns and 'planned_stop_loss' in completed.columns:
        valid_entries = completed.dropna(subset=['entry', 'exit', 'planned_take_profit', 'planned_stop_loss'])
        
        if not valid_entries.empty:
            valid_entries['potential_profit'] = abs(valid_entries['planned_take_profit'] - valid_entries['entry'])
            valid_entries['potential_loss'] = abs(valid_entries['planned_stop_loss'] - valid_entries['entry'])
            
            # Calculate efficiency for long trades
            long_trades = valid_entries[valid_entries['direction'] == 'Long']
            if not long_trades.empty:
                long_trades['actual_profit'] = long_trades['exit'] - long_trades['entry']
                long_trades['efficiency'] = long_trades['actual_profit'] / long_trades['potential_profit']
            
            # Calculate efficiency for short trades
            short_trades = valid_entries[valid_entries['direction'] == 'Short']
            if not short_trades.empty:
                short_trades['actual_profit'] = short_trades['entry'] - short_trades['exit']
                short_trades['efficiency'] = short_trades['actual_profit'] / short_trades['potential_profit']
            
            # Combine and calculate average efficiency
            efficiency_trades = pd.concat([long_trades, short_trades]) if not short_trades.empty and not long_trades.empty else long_trades if not long_trades.empty else short_trades
            efficiency_ratio = efficiency_trades['efficiency'].mean() * 100 if not efficiency_trades.empty else 0
        else:
            efficiency_ratio = 0
    else:
        efficiency_ratio = 0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_rrr': avg_rrr,
        'consecutive_wins': max_win_streak,
        'consecutive_losses': max_loss_streak,
        'expectancy': expectancy,
        'recovery_factor': recovery_factor,
        'kelly_percentage': kelly_percentage,
        'efficiency_ratio': efficiency_ratio
    }

def calculate_market_metrics(df):
    """Calculate market-specific metrics"""
    if df.empty:
        return {
            'top_markets': {},
            'market_pnl': {},
            'market_win_rates': {},
            'market_trade_counts': {}
        }
    
    # Filter for completed trades
    completed = df[df['status'] == 'Completed']
    
    if completed.empty:
        return {
            'top_markets': {},
            'market_pnl': {},
            'market_win_rates': {},
            'market_trade_counts': {}
        }
    
    # Group by market/symbol
    market_groups = completed.groupby('symbol')
    
    # Market PnL
    market_pnl = market_groups['pnl'].sum().to_dict()
    
    # Market trade counts
    market_trade_counts = market_groups.size().to_dict()
    
    # Market win rates
    market_win_rates = {}
    for market, group in market_groups:
        wins = (group['pnl'] > 0).sum()
        total = len(group)
        market_win_rates[market] = (wins / total) * 100 if total > 0 else 0
    
    # Top performing markets by PnL
    top_markets = dict(sorted(market_pnl.items(), key=lambda x: x[1], reverse=True)[:5])
    
    return {
        'top_markets': top_markets,
        'market_pnl': market_pnl,
        'market_win_rates': market_win_rates,
        'market_trade_counts': market_trade_counts
    }

def calculate_ict_specific_metrics(df):
    """Calculate ICT-specific trading metrics"""
    if df.empty:
        return {
            'ob_hit_rate': 0,
            'average_ob_deviation': 0,
            'liquidity_grab_success': 0,
            'fvg_efficiency': 0,
            'breaker_block_success': 0,
            'premium_discount_ratio': 0,
            'inducement_success_rate': 0,
            'imbalance_efficiency': 0
        }
    
    # Filter for completed trades
    completed = df[df['status'] == 'Completed']
    
    if completed.empty:
        return {
            'ob_hit_rate': 0,
            'average_ob_deviation': 0,
            'liquidity_grab_success': 0,
            'fvg_efficiency': 0,
            'breaker_block_success': 0,
            'premium_discount_ratio': 0,
            'inducement_success_rate': 0,
            'imbalance_efficiency': 0
        }
    
    # Order Block Hit Rate (if such fields exist)
    ob_hit_rate = 0
    if 'order_block_entry' in completed.columns and 'entry_hit_ob' in completed.columns:
        ob_entries = completed[completed['order_block_entry'] == True]
        if len(ob_entries) > 0:
            ob_hit_rate = (ob_entries['entry_hit_ob'].sum() / len(ob_entries)) * 100
    
    # Average Order Block Deviation
    avg_ob_deviation = 0
    if 'ob_deviation' in completed.columns:
        valid_ob_entries = completed.dropna(subset=['ob_deviation'])
        if not valid_ob_entries.empty:
            avg_ob_deviation = valid_ob_entries['ob_deviation'].mean()
    
    # Liquidity Grab Success Rate
    liquidity_grab_success = 0
    if 'liquidity_grab' in completed.columns and 'liquidity_grab_success' in completed.columns:
        liq_grab_trades = completed[completed['liquidity_grab'] == True]
        if len(liq_grab_trades) > 0:
            liquidity_grab_success = (liq_grab_trades['liquidity_grab_success'].sum() / len(liq_grab_trades)) * 100
    
    # Fair Value Gap (FVG) Efficiency
    fvg_efficiency = 0
    if 'fvg_entry' in completed.columns and 'fvg_filled' in completed.columns:
        fvg_trades = completed[completed['fvg_entry'] == True]
        if len(fvg_trades) > 0:
            fvg_efficiency = (fvg_trades['fvg_filled'].sum() / len(fvg_trades)) * 100
    
    # Breaker Block Success Rate
    breaker_block_success = 0
    if 'breaker_block' in completed.columns and 'breaker_respected' in completed.columns:
        breaker_trades = completed[completed['breaker_block'] == True]
        if len(breaker_trades) > 0:
            breaker_block_success = (breaker_trades['breaker_respected'].sum() / len(breaker_trades)) * 100
    
    # Premium/Discount Trading Ratio
    premium_discount_ratio = 0
    if 'market_condition' in completed.columns:
        premium_trades = len(completed[completed['market_condition'] == 'Premium'])
        discount_trades = len(completed[completed['market_condition'] == 'Discount'])
        
        if discount_trades > 0:
            premium_discount_ratio = premium_trades / discount_trades
        elif premium_trades > 0:
            premium_discount_ratio = float('inf')
    
    # Inducement Success Rate
    inducement_success_rate = 0
    if 'inducement_identified' in completed.columns and 'inducement_worked' in completed.columns:
        inducement_trades = completed[completed['inducement_identified'] == True]
        if len(inducement_trades) > 0:
            inducement_success_rate = (inducement_trades['inducement_worked'].sum() / len(inducement_trades)) * 100
    
    # Imbalance Efficiency (how well imbalances are traded)
    imbalance_efficiency = 0
    if 'imbalance_entry' in completed.columns and 'imbalance_filled' in completed.columns:
        imbalance_trades = completed[completed['imbalance_entry'] == True]
        if len(imbalance_trades) > 0:
            imbalance_efficiency = (imbalance_trades['imbalance_filled'].sum() / len(imbalance_trades)) * 100
    
    return {
        'ob_hit_rate': ob_hit_rate,
        'average_ob_deviation': avg_ob_deviation,
        'liquidity_grab_success': liquidity_grab_success,
        'fvg_efficiency': fvg_efficiency,
        'breaker_block_success': breaker_block_success,
        'premium_discount_ratio': premium_discount_ratio,
        'inducement_success_rate': inducement_success_rate,
        'imbalance_efficiency': imbalance_efficiency
    }

def calculate_drawdowns(df):
    """Calculate drawdown periods and statistics"""
    if df.empty:
        return [], 0, 0, 0
    
    # Filter for completed trades
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return [], 0, 0, 0
    
    # Sort by date
    completed = completed.sort_values('date')
    
    # Calculate running balance
    completed['cumulative_pnl'] = completed['pnl'].cumsum()
    
    # Calculate drawdown
    completed['peak'] = completed['cumulative_pnl'].cummax()
    completed['drawdown'] = completed['cumulative_pnl'] - completed['peak']
    
    # Identify drawdown periods
    drawdown_start = None
    drawdown_periods = []
    
    for i, row in completed.iterrows():
        # Check if we're at a peak
        if row['cumulative_pnl'] == row['peak'] and drawdown_start is not None:
            # End of drawdown
            end_date = row['date']
            prev_peak = completed.loc[drawdown_start, 'peak']
            deepest_point = completed.loc[drawdown_start:i, 'drawdown'].min()
            recovery_time = (end_date - completed.loc[drawdown_start, 'date']).days
            
            drawdown_periods.append({
                'start_date': completed.loc[drawdown_start, 'date'],
                'end_date': end_date,
                'previous_peak': prev_peak,
                'drawdown_amount': abs(deepest_point),
                'drawdown_pct': (abs(deepest_point) / prev_peak) * 100 if prev_peak > 0 else 0,
                'recovery_days': recovery_time
            })
            
            drawdown_start = None
        
        # Check if we're starting a drawdown
        elif row['drawdown'] < 0 and drawdown_start is None:
            drawdown_start = i
    
    # If we're still in a drawdown at the end
    if drawdown_start is not None:
        end_date = completed.iloc[-1]['date']
        prev_peak = completed.loc[drawdown_start, 'peak']
        deepest_point = completed.loc[drawdown_start:, 'drawdown'].min()
        recovery_time = (end_date - completed.loc[drawdown_start, 'date']).days
        
        drawdown_periods.append({
            'start_date': completed.loc[drawdown_start, 'date'],
            'end_date': "Ongoing",
            'previous_peak': prev_peak,
            'drawdown_amount': abs(deepest_point),
            'drawdown_pct': (abs(deepest_point) / prev_peak) * 100 if prev_peak > 0 else 0,
            'recovery_days': recovery_time
        })
    
    # Calculate max drawdown info
    max_drawdown = abs(completed['drawdown'].min()) if not completed.empty else 0
    max_drawdown_pct = (max_drawdown / completed['peak'].max()) * 100 if completed['peak'].max() > 0 else 0
    
    # Calculate average recovery time
    avg_recovery_time = np.mean([period['recovery_days'] for period in drawdown_periods]) if drawdown_periods else 0
    
    return drawdown_periods, max_drawdown, max_drawdown_pct, avg_recovery_time

def calculate_backtest_vs_forward_metrics(df):
    """Compare backtesting and forward testing performance"""
    if df.empty:
        return {
            'backtest_count': 0,
            'forward_count': 0,
            'backtest_win_rate': 0,
            'forward_win_rate': 0,
            'backtest_pnl': 0,
            'forward_pnl': 0,
            'backtest_avg_trade': 0,
            'forward_avg_trade': 0,
            'backtest_profit_factor': 0,
            'forward_profit_factor': 0
        }
    
    # Filter for completed trades
    completed = df[df['status'] == 'Completed']
    
    if completed.empty or 'test_type' not in completed.columns:
        return {
            'backtest_count': 0,
            'forward_count': 0,
            'backtest_win_rate': 0,
            'forward_win_rate': 0,
            'backtest_pnl': 0,
            'forward_pnl': 0,
            'backtest_avg_trade': 0,
            'forward_avg_trade': 0,
            'backtest_profit_factor': 0,
            'forward_profit_factor': 0
        }
    
    # Split into backtest and forward test
    backtest = completed[completed['test_type'] == 'Backtest']
    forward = completed[completed['test_type'] == 'Forward']
    
    # Calculate metrics for backtest
    backtest_count = len(backtest)
    backtest_wins = len(backtest[backtest['pnl'] > 0])
    backtest_win_rate = (backtest_wins / backtest_count) * 100 if backtest_count > 0 else 0
    backtest_pnl = backtest['pnl'].sum() if not backtest.empty else 0
    backtest_avg_trade = backtest['pnl'].mean() if not backtest.empty else 0
    
    backtest_profit = backtest[backtest['pnl'] > 0]['pnl'].sum() if not backtest.empty else 0
    backtest_loss = abs(backtest[backtest['pnl'] < 0]['pnl'].sum()) if not backtest.empty else 0
    backtest_profit_factor = backtest_profit / backtest_loss if backtest_loss > 0 else float('inf') if backtest_profit > 0 else 0
    
    # Calculate metrics for forward test
    forward_count = len(forward)
    forward_wins = len(forward[forward['pnl'] > 0])
    forward_win_rate = (forward_wins / forward_count) * 100 if forward_count > 0 else 0
    forward_pnl = forward['pnl'].sum() if not forward.empty else 0
    forward_avg_trade = forward['pnl'].mean() if not forward.empty else 0
    
    forward_profit = forward[forward['pnl'] > 0]['pnl'].sum() if not forward.empty else 0
    forward_loss = abs(forward[forward['pnl'] < 0]['pnl'].sum()) if not forward.empty else 0
    forward_profit_factor = forward_profit / forward_loss if forward_loss > 0 else float('inf') if forward_profit > 0 else 0
    
    return {
        'backtest_count': backtest_count,
        'forward_count': forward_count,
        'backtest_win_rate': backtest_win_rate,
        'forward_win_rate': forward_win_rate,
        'backtest_pnl': backtest_pnl,
        'forward_pnl': forward_pnl,
        'backtest_avg_trade': backtest_avg_trade,
        'forward_avg_trade': forward_avg_trade,
        'backtest_profit_factor': backtest_profit_factor,
        'forward_profit_factor': forward_profit_factor
    }

def calculate_psychology_metrics(psychology_df, trades_df):
    """Calculate psychology-related metrics from psychology entries and trades"""
    if psychology_df.empty:
        return {
            'avg_mental_state': 0,
            'avg_focus': 0,
            'avg_discipline': 0,
            'avg_confidence': 0,
            'avg_patience': 0,
            'best_mental_state': 0,
            'corr_mental_state_pnl': 0,
            'days_tracked': 0
        }
    
    # Basic averages
    avg_mental_state = psychology_df['mental_state'].mean() if 'mental_state' in psychology_df.columns else 0
    avg_focus = psychology_df['focus'].mean() if 'focus' in psychology_df.columns else 0
    avg_discipline = psychology_df['discipline'].mean() if 'discipline' in psychology_df.columns else 0
    avg_confidence = psychology_df['confidence'].mean() if 'confidence' in psychology_df.columns else 0
    avg_patience = psychology_df['patience'].mean() if 'patience' in psychology_df.columns else 0
    
    # Best mental state
    best_mental_state = psychology_df['mental_state'].max() if 'mental_state' in psychology_df.columns else 0
    
    # Days tracked
    days_tracked = psychology_df['date'].nunique() if 'date' in psychology_df.columns else 0
    
    # Correlation between mental state and trade performance
    corr_mental_state_pnl = 0
    
    if not psychology_df.empty and not trades_df.empty and 'mental_state' in psychology_df.columns:
        # Prepare trade data
        trades_with_date = trades_df[['date', 'pnl']].copy()
        trades_with_date['date'] = pd.to_datetime(trades_with_date['date']).dt.date
        
        # Group trades by date
        daily_pnl = trades_with_date.groupby('date')['pnl'].sum().reset_index()
        
        # Prepare psychology data
        psychology_with_date = psychology_df[['date', 'mental_state']].copy()
        psychology_with_date['date'] = pd.to_datetime(psychology_with_date['date']).dt.date
        
        # Merge data
        merged_data = pd.merge(daily_pnl, psychology_with_date, on='date', how='inner')
        
        if len(merged_data) > 1:
            # Calculate correlation
            corr_mental_state_pnl = merged_data['mental_state'].corr(merged_data['pnl'])
    
    return {
        'avg_mental_state': avg_mental_state,
        'avg_focus': avg_focus,
        'avg_discipline': avg_discipline,
        'avg_confidence': avg_confidence,
        'avg_patience': avg_patience,
        'best_mental_state': best_mental_state,
        'corr_mental_state_pnl': corr_mental_state_pnl,
        'days_tracked': days_tracked
    }

def calculate_timeframe_metrics(df):
    """Calculate metrics across different timeframes"""
    if df.empty or 'date' not in df.columns:
        return {
            'day_of_week': {},
            'month': {},
            'hour_of_day': {}
        }
    
    # Filter for completed trades
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return {
            'day_of_week': {},
            'month': {},
            'hour_of_day': {}
        }
    
    # Convert date to datetime
    completed['date'] = pd.to_datetime(completed['date'])
    
    # Extract time components
    completed['day_of_week'] = completed['date'].dt.day_name()
    completed['month'] = completed['date'].dt.month_name()
    
    # Extract hour if time component is available
    if completed['date'].dt.hour.nunique() > 1:
        completed['hour'] = completed['date'].dt.hour
        hour_metrics = {}
        
        for hour, group in completed.groupby('hour'):
            hour_pnl = group['pnl'].sum()
            hour_trades = len(group)
            hour_wins = len(group[group['pnl'] > 0])
            hour_win_rate = (hour_wins / hour_trades) * 100 if hour_trades > 0 else 0
            
            hour_metrics[hour] = {
                'trades': hour_trades,
                'pnl': hour_pnl,
                'win_rate': hour_win_rate
            }
    else:
        hour_metrics = {}
    
    # Metrics by day of week
    day_metrics = {}
    for day, group in completed.groupby('day_of_week'):
        day_pnl = group['pnl'].sum()
        day_trades = len(group)
        day_wins = len(group[group['pnl'] > 0])
        day_win_rate = (day_wins / day_trades) * 100 if day_trades > 0 else 0
        
        day_metrics[day] = {
            'trades': day_trades,
            'pnl': day_pnl,
            'win_rate': day_win_rate
        }
    
    # Metrics by month
    month_metrics = {}
    for month, group in completed.groupby('month'):
        month_pnl = group['pnl'].sum()
        month_trades = len(group)
        month_wins = len(group[group['pnl'] > 0])
        month_win_rate = (month_wins / month_trades) * 100 if month_trades > 0 else 0
        
        month_metrics[month] = {
            'trades': month_trades,
            'pnl': month_pnl,
            'win_rate': month_win_rate
        }
    
    return {
        'day_of_week': day_metrics,
        'month': month_metrics,
        'hour_of_day': hour_metrics
    }
