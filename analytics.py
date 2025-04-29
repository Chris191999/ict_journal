import streamlit as st
import pandas as pd
import numpy as np
from utils.data_manager import get_trades_df, filter_trades
from utils.metrics import (
    calculate_basic_metrics, calculate_advanced_metrics, 
    calculate_market_metrics, calculate_ict_specific_metrics,
    calculate_drawdowns, calculate_backtest_vs_forward_metrics,
    calculate_timeframe_metrics
)
from utils.visualization import (
    create_equity_curve, create_drawdown_chart, create_win_loss_chart,
    create_pnl_distribution, create_market_performance_chart,
    create_ict_concept_performance, create_backtest_vs_forward_chart,
    create_monthly_performance_chart, create_day_of_week_chart
)

def show_analytics_page():
    """Display the analytics page with performance metrics and visualizations"""
    st.title("Trading Performance Analytics")
    
    # Get trades dataframe
    trades_df = get_trades_df()
    
    if trades_df.empty:
        st.info("No trades recorded yet. Add trades in the Journal Entries page or import data first.")
        return
        
    # Filter options
    with st.expander("Filter Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Date range filter
            date_range = st.date_input(
                "Date Range",
                value=(
                    pd.to_datetime(trades_df['date']).min().date() if 'date' in trades_df.columns else pd.Timestamp.now().date() - pd.Timedelta(days=30),
                    pd.to_datetime(trades_df['date']).max().date() if 'date' in trades_df.columns else pd.Timestamp.now().date()
                ),
                key='analytics_date_range'
            )
            st.session_state.filters['date_range'] = date_range
        
        with col2:
            # Symbol filter
            all_symbols = trades_df['symbol'].unique().tolist() if 'symbol' in trades_df.columns else []
            selected_symbols = st.multiselect("Symbols", options=all_symbols, default=[], key='analytics_symbols')
            st.session_state.filters['symbols'] = selected_symbols
        
        with col3:
            # Test type filter
            test_types = trades_df['test_type'].unique().tolist() if 'test_type' in trades_df.columns else []
            selected_test_types = st.multiselect("Test Type", options=test_types, default=[], key='analytics_test_types')
            # Create a test_type filter in session state
            if 'test_type' not in st.session_state.filters:
                st.session_state.filters['test_type'] = []
            st.session_state.filters['test_type'] = selected_test_types
        
        # Direction and outcome
        dir_col, outcome_col, status_col = st.columns(3)
        
        with dir_col:
            selected_directions = st.multiselect(
                "Direction", 
                options=["Long", "Short"],
                default=[],
                key='analytics_direction'
            )
            st.session_state.filters['direction'] = selected_directions
        
        with outcome_col:
            selected_outcomes = st.multiselect(
                "Outcome",
                options=["Win", "Loss"],
                default=[],
                key='analytics_outcome'
            )
            st.session_state.filters['outcome'] = selected_outcomes
            
        with status_col:
            statuses = trades_df['status'].unique().tolist() if 'status' in trades_df.columns else []
            default_statuses = ["Completed"] if "Completed" in statuses else []
            selected_statuses = st.multiselect("Status", options=statuses, default=default_statuses, key='analytics_status')
            st.session_state.filters['status'] = selected_statuses
    
    # Apply filters
    filtered_df = filter_trades(trades_df, st.session_state.filters)
    
    if filtered_df.empty:
        st.warning("No trades match the selected filters. Please adjust your filter criteria.")
        return
    
    # Display metrics and visualizations in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Advanced Metrics", "Market Analysis", 
        "ICT Analysis", "Backtest vs. Forward"
    ])
    
    with tab1:
        show_overview_tab(filtered_df)
    
    with tab2:
        show_advanced_metrics_tab(filtered_df)
    
    with tab3:
        show_market_analysis_tab(filtered_df)
    
    with tab4:
        show_ict_analysis_tab(filtered_df)
    
    with tab5:
        show_backtest_comparison_tab(filtered_df)

def show_overview_tab(df):
    """Display overview metrics and charts"""
    st.subheader("Performance Overview")
    
    # Calculate basic metrics
    basic_metrics = calculate_basic_metrics(df)
    
    # Display key metrics at the top
    metric_cols = st.columns(4)
    
    metric_cols[0].metric("Total Trades", basic_metrics['total_trades'])
    metric_cols[1].metric("Win Rate", f"{basic_metrics['win_rate']:.2f}%")
    metric_cols[2].metric("Net P&L", f"${basic_metrics['net_pnl']:.2f}")
    metric_cols[3].metric("Profit Factor", f"{basic_metrics['profit_factor']:.2f}")
    
    # Equity curve
    st.subheader("Equity Curve")
    equity_chart = create_equity_curve(df)
    if equity_chart:
        st.plotly_chart(equity_chart, use_container_width=True)
    else:
        st.info("Not enough data to display equity curve.")
    
    # Win/Loss and P&L distribution side by side
    wl_col, pnl_col = st.columns(2)
    
    with wl_col:
        win_loss_chart = create_win_loss_chart(df)
        if win_loss_chart:
            st.plotly_chart(win_loss_chart, use_container_width=True)
    
    with pnl_col:
        pnl_chart = create_pnl_distribution(df)
        if pnl_chart:
            st.plotly_chart(pnl_chart, use_container_width=True)
    
    # Drawdown chart
    st.subheader("Drawdown Analysis")
    drawdown_chart = create_drawdown_chart(df)
    if drawdown_chart:
        st.plotly_chart(drawdown_chart, use_container_width=True)
    
    # Monthly performance
    st.subheader("Monthly Performance")
    monthly_chart = create_monthly_performance_chart(df)
    if monthly_chart:
        st.plotly_chart(monthly_chart, use_container_width=True)
    else:
        st.info("Not enough data to display monthly performance.")
    
    # Additional statistics
    st.subheader("Trade Statistics")
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.write("**Win/Loss Metrics**")
        st.write(f"Winning Trades: {basic_metrics['winning_trades']}")
        st.write(f"Losing Trades: {basic_metrics['losing_trades']}")
        st.write(f"Win Rate: {basic_metrics['win_rate']:.2f}%")
        st.write(f"Average Win: ${basic_metrics['avg_win']:.2f}")
        st.write(f"Average Loss: ${basic_metrics['avg_loss']:.2f}")
        st.write(f"Largest Win: ${basic_metrics['largest_win']:.2f}")
        st.write(f"Largest Loss: ${basic_metrics['largest_loss']:.2f}")
    
    with stats_col2:
        st.write("**Profit Metrics**")
        st.write(f"Net P&L: ${basic_metrics['net_pnl']:.2f}")
        st.write(f"Gross Profit: ${basic_metrics['total_profit']:.2f}")
        st.write(f"Gross Loss: ${basic_metrics['total_loss']:.2f}")
        st.write(f"Profit Factor: {basic_metrics['profit_factor']:.2f}")
        st.write(f"Average Trade P&L: ${basic_metrics['net_pnl'] / basic_metrics['total_trades'] if basic_metrics['total_trades'] > 0 else 0:.2f}")

def show_advanced_metrics_tab(df):
    """Display advanced trading metrics"""
    st.subheader("Advanced Performance Metrics")
    
    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(df)
    
    # Display key advanced metrics at the top
    adv_metric_cols = st.columns(4)
    
    adv_metric_cols[0].metric("Sharpe Ratio", f"{advanced_metrics['sharpe_ratio']:.2f}")
    adv_metric_cols[1].metric("Max Drawdown", f"${advanced_metrics['max_drawdown']:.2f}")
    adv_metric_cols[2].metric("Avg RRR", f"{advanced_metrics['avg_rrr']:.2f}")
    adv_metric_cols[3].metric("Expectancy", f"${advanced_metrics['expectancy']:.2f}")
    
    # Drawdown periods
    st.subheader("Drawdown Periods")
    drawdown_periods, max_drawdown, max_drawdown_pct, avg_recovery_time = calculate_drawdowns(df)
    
    if drawdown_periods:
        drawdown_df = pd.DataFrame(drawdown_periods)
        st.dataframe(drawdown_df, use_container_width=True)
        
        # Drawdown summary
        dd_col1, dd_col2, dd_col3 = st.columns(3)
        dd_col1.metric("Maximum Drawdown", f"${max_drawdown:.2f}")
        dd_col2.metric("Maximum Drawdown %", f"{max_drawdown_pct:.2f}%")
        dd_col3.metric("Avg Recovery Time", f"{avg_recovery_time:.1f} days")
    else:
        st.info("No significant drawdown periods detected.")
    
    # Timeframe analysis
    st.subheader("Performance by Timeframe")
    
    # Calculate timeframe metrics
    timeframe_metrics = calculate_timeframe_metrics(df)
    
    # Day of week chart
    st.write("**Performance by Day of Week**")
    dow_chart = create_day_of_week_chart(df)
    if dow_chart:
        st.plotly_chart(dow_chart, use_container_width=True)
    else:
        st.info("Not enough data to analyze performance by day of week.")
    
    # Detailed advanced metrics
    st.subheader("Detailed Advanced Metrics")
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        st.write("**Risk Metrics**")
        st.write(f"Sharpe Ratio: {advanced_metrics['sharpe_ratio']:.2f}")
        st.write(f"Max Drawdown: ${advanced_metrics['max_drawdown']:.2f}")
        st.write(f"Max Drawdown %: {advanced_metrics['max_drawdown_pct']:.2f}%")
        st.write(f"Average RRR: {advanced_metrics['avg_rrr']:.2f}")
        st.write(f"Recovery Factor: {advanced_metrics['recovery_factor']:.2f}")
    
    with adv_col2:
        st.write("**Statistical Metrics**")
        st.write(f"Expectancy: ${advanced_metrics['expectancy']:.2f}")
        st.write(f"Consecutive Wins: {advanced_metrics['consecutive_wins']}")
        st.write(f"Consecutive Losses: {advanced_metrics['consecutive_losses']}")
        st.write(f"Kelly Percentage: {advanced_metrics['kelly_percentage']:.2f}%")
        st.write(f"Efficiency Ratio: {advanced_metrics['efficiency_ratio']:.2f}%")

def show_market_analysis_tab(df):
    """Display market-specific analysis"""
    st.subheader("Market Analysis")
    
    # Calculate market metrics
    market_metrics = calculate_market_metrics(df)
    
    # Display market performance chart
    market_chart = create_market_performance_chart(df)
    if market_chart:
        st.plotly_chart(market_chart, use_container_width=True)
    else:
        st.info("Not enough data to display market performance.")
    
    # Top markets
    st.subheader("Top Performing Markets")
    
    if market_metrics['top_markets']:
        top_markets_df = pd.DataFrame({
            'Market': list(market_metrics['top_markets'].keys()),
            'P&L': list(market_metrics['top_markets'].values())
        })
        
        st.dataframe(top_markets_df, use_container_width=True)
    else:
        st.info("No market performance data available.")
    
    # Market statistics
    if market_metrics['market_trade_counts']:
        st.subheader("Market Statistics")
        
        market_stats = []
        for market in market_metrics['market_trade_counts']:
            market_stats.append({
                'Market': market,
                'Trades': market_metrics['market_trade_counts'].get(market, 0),
                'P&L': market_metrics['market_pnl'].get(market, 0),
                'Win Rate': f"{market_metrics['market_win_rates'].get(market, 0):.2f}%",
                'Avg P&L': market_metrics['market_pnl'].get(market, 0) / market_metrics['market_trade_counts'].get(market, 1)
            })
        
        market_stats_df = pd.DataFrame(market_stats)
        market_stats_df = market_stats_df.sort_values('P&L', ascending=False)
        
        st.dataframe(market_stats_df, use_container_width=True)

def show_ict_analysis_tab(df):
    """Display ICT-specific analysis"""
    st.subheader("ICT Concept Analysis")
    
    # Calculate ICT-specific metrics
    ict_metrics = calculate_ict_specific_metrics(df)
    
    # Display ICT concept performance chart
    ict_chart = create_ict_concept_performance(df)
    if ict_chart:
        st.plotly_chart(ict_chart, use_container_width=True)
    else:
        st.info("Not enough ICT-specific data available. Tag your trades with ICT concepts for detailed analysis.")
    
    # Display ICT metrics
    st.subheader("ICT Trading Metrics")
    
    ict_col1, ict_col2 = st.columns(2)
    
    with ict_col1:
        st.write("**Order Block & Liquidity Metrics**")
        st.write(f"Order Block Hit Rate: {ict_metrics['ob_hit_rate']:.2f}%")
        st.write(f"Average OB Deviation: {ict_metrics['average_ob_deviation']:.4f}")
        st.write(f"Liquidity Grab Success: {ict_metrics['liquidity_grab_success']:.2f}%")
        st.write(f"FVG Efficiency: {ict_metrics['fvg_efficiency']:.2f}%")
    
    with ict_col2:
        st.write("**Advanced ICT Metrics**")
        st.write(f"Breaker Block Success: {ict_metrics['breaker_block_success']:.2f}%")
        st.write(f"Premium/Discount Ratio: {ict_metrics['premium_discount_ratio']:.2f}")
        st.write(f"Inducement Success Rate: {ict_metrics['inducement_success_rate']:.2f}%")
        st.write(f"Imbalance Efficiency: {ict_metrics['imbalance_efficiency']:.2f}%")
    
    # ICT Trading Patterns Analysis
    st.subheader("ICT Trading Pattern Performance")
    
    # Analyze trade patterns based on combinations of ICT concepts
    ict_patterns = []
    
    # Check if necessary columns exist
    pattern_columns = ['order_block_entry', 'liquidity_grab', 'fvg_entry', 
                      'breaker_block', 'imbalance_entry', 'inducement_identified']
    
    existing_columns = [col for col in pattern_columns if col in df.columns]
    
    if existing_columns:
        # For completed trades only
        completed = df[df['status'] == 'Completed'].copy()
        
        if not completed.empty:
            # Define common ICT patterns
            patterns = {
                'OB + Liquidity': ['order_block_entry', 'liquidity_grab'],
                'FVG + Imbalance': ['fvg_entry', 'imbalance_entry'],
                'Breaker + Inducement': ['breaker_block', 'inducement_identified'],
                'OB + FVG': ['order_block_entry', 'fvg_entry'],
                'Premium Trading': ['market_condition']  # Special case for market condition
            }
            
            for pattern_name, pattern_cols in patterns.items():
                # Check if all required columns exist
                if not all(col in df.columns for col in pattern_cols):
                    continue
                
                if pattern_name == 'Premium Trading':
                    # Special case for market condition
                    pattern_trades = completed[completed['market_condition'] == 'Premium']
                else:
                    # Regular pattern where all conditions must be True
                    pattern_condition = True
                    for col in pattern_cols:
                        pattern_condition = pattern_condition & (completed[col] == True)
                    
                    pattern_trades = completed[pattern_condition]
                
                if len(pattern_trades) > 0:
                    win_rate = (pattern_trades['pnl'] > 0).mean() * 100
                    avg_pnl = pattern_trades['pnl'].mean()
                    max_pnl = pattern_trades['pnl'].max()
                    
                    ict_patterns.append({
                        'Pattern': pattern_name,
                        'Trades': len(pattern_trades),
                        'Win Rate': f"{win_rate:.2f}%",
                        'Avg P&L': f"${avg_pnl:.2f}",
                        'Max P&L': f"${max_pnl:.2f}"
                    })
        
        if ict_patterns:
            ict_patterns_df = pd.DataFrame(ict_patterns)
            st.dataframe(ict_patterns_df, use_container_width=True)
        else:
            st.info("No ICT pattern data available. Tag your trades with multiple ICT concepts to see pattern analysis.")
    else:
        st.info("No ICT-specific columns found in your trade data. Add ICT concepts to your trades for detailed analysis.")

def show_backtest_comparison_tab(df):
    """Display backtest vs. forward test comparison"""
    st.subheader("Backtest vs. Forward Test Comparison")
    
    if 'test_type' not in df.columns:
        st.info("Test type information is missing. Please tag your trades as 'Backtest' or 'Forward'.")
        return
    
    # Calculate comparison metrics
    comparison_metrics = calculate_backtest_vs_forward_metrics(df)
    
    # Display comparison chart
    comparison_chart = create_backtest_vs_forward_chart(df)
    if comparison_chart:
        st.plotly_chart(comparison_chart, use_container_width=True)
    else:
        st.info("Not enough data to compare backtest and forward test results. Ensure you have trades tagged with both test types.")
    
    # Display metrics side by side
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.write("**Backtest Results**")
        st.write(f"Total Trades: {comparison_metrics['backtest_count']}")
        st.write(f"Win Rate: {comparison_metrics['backtest_win_rate']:.2f}%")
        st.write(f"Total P&L: ${comparison_metrics['backtest_pnl']:.2f}")
        st.write(f"Average Trade: ${comparison_metrics['backtest_avg_trade']:.2f}")
        st.write(f"Profit Factor: {comparison_metrics['backtest_profit_factor']:.2f}")
    
    with comp_col2:
        st.write("**Forward Test Results**")
        st.write(f"Total Trades: {comparison_metrics['forward_count']}")
        st.write(f"Win Rate: {comparison_metrics['forward_win_rate']:.2f}%")
        st.write(f"Total P&L: ${comparison_metrics['forward_pnl']:.2f}")
        st.write(f"Average Trade: ${comparison_metrics['forward_avg_trade']:.2f}")
        st.write(f"Profit Factor: {comparison_metrics['forward_profit_factor']:.2f}")
    
    # Display implementation gap analysis
    if comparison_metrics['backtest_count'] > 0 and comparison_metrics['forward_count'] > 0:
        st.subheader("Implementation Gap Analysis")
        
        # Calculate performance differences
        win_rate_diff = comparison_metrics['forward_win_rate'] - comparison_metrics['backtest_win_rate']
        avg_trade_diff = comparison_metrics['forward_avg_trade'] - comparison_metrics['backtest_avg_trade']
        profit_factor_diff = comparison_metrics['forward_profit_factor'] - comparison_metrics['backtest_profit_factor']
        
        # Display differences
        diff_col1, diff_col2, diff_col3 = st.columns(3)
        
        diff_col1.metric(
            "Win Rate Gap", 
            f"{win_rate_diff:.2f}%",
            delta=f"{win_rate_diff:.2f}%"
        )
        
        diff_col2.metric(
            "Avg Trade Gap", 
            f"${avg_trade_diff:.2f}",
            delta=f"${avg_trade_diff:.2f}"
        )
        
        diff_col3.metric(
            "Profit Factor Gap", 
            f"{profit_factor_diff:.2f}",
            delta=f"{profit_factor_diff:.2f}"
        )
        
        # Analysis text
        st.write("**Gap Analysis**")
        
        if win_rate_diff < -5:
            st.warning("Your forward testing win rate is significantly lower than backtesting. Consider reviewing your trade execution or strategy adaptation.")
        elif win_rate_diff > 5:
            st.success("Your forward testing win rate is higher than backtesting. Your real-world implementation is outperforming your testing.")
        
        if avg_trade_diff < 0:
            st.warning(f"Your average forward trade P&L is ${abs(avg_trade_diff):.2f} lower than backtesting. Review position sizing or exit management.")
        else:
            st.success(f"Your average forward trade P&L is ${avg_trade_diff:.2f} higher than backtesting.")
        
        # Overall assessment
        overall_gap = (win_rate_diff + (avg_trade_diff * 10) + profit_factor_diff) / 3
        
        st.write("**Overall Implementation Assessment**")
        
        if overall_gap < -5:
            st.error("Significant negative implementation gap. Your forward testing results are substantially underperforming your backtesting expectations.")
        elif overall_gap < 0:
            st.warning("Slight negative implementation gap. Your forward testing results aren't fully matching your backtesting expectations.")
        elif overall_gap < 5:
            st.success("Strong implementation. Your forward testing closely matches or slightly exceeds your backtesting expectations.")
        else:
            st.success("Exceptional implementation. Your forward testing significantly outperforms your backtesting, indicating strong execution or improved strategy.")
