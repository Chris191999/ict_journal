import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def create_equity_curve(df):
    """
    Create an equity curve chart using trade data
    """
    if df.empty or 'date' not in df.columns or 'pnl' not in df.columns:
        return None
    
    # Filter for completed trades only
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return None
    
    # Ensure date is datetime
    completed['date'] = pd.to_datetime(completed['date'])
    completed = completed.sort_values('date')
    
    # Calculate cumulative P&L
    completed['cumulative_pnl'] = completed['pnl'].cumsum()
    
    # Create figure
    fig = px.line(
        completed,
        x='date',
        y='cumulative_pnl',
        title='Equity Curve',
        labels={'cumulative_pnl': 'Cumulative P&L', 'date': 'Date'},
        template='plotly_white'
    )
    
    # Add profit/loss markers
    fig.add_trace(
        go.Scatter(
            x=completed[completed['pnl'] > 0]['date'],
            y=completed[completed['pnl'] > 0]['cumulative_pnl'],
            mode='markers',
            marker=dict(color='green', size=8, symbol='circle'),
            name='Winning Trade'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=completed[completed['pnl'] < 0]['date'],
            y=completed[completed['pnl'] < 0]['cumulative_pnl'],
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle'),
            name='Losing Trade'
        )
    )
    
    # Add hover information
    fig.update_traces(
        hovertemplate='<b>Date</b>: %{x}<br><b>Cumulative P&L</b>: $%{y:.2f}<extra></extra>'
    )
    
    # Add regression line to show trend
    fig.add_trace(
        go.Scatter(
            x=completed['date'],
            y=completed['cumulative_pnl'].rolling(window=min(5, len(completed))).mean(),
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.5)', width=2, dash='dash'),
            name='Moving Average (5 trades)'
        )
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Cumulative P&L ($)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    return fig

def create_drawdown_chart(df):
    """
    Create a drawdown chart to visualize equity drawdowns
    """
    if df.empty or 'date' not in df.columns or 'pnl' not in df.columns:
        return None
    
    # Filter for completed trades only
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return None
    
    # Ensure date is datetime
    completed['date'] = pd.to_datetime(completed['date'])
    completed = completed.sort_values('date')
    
    # Calculate running balance and drawdown
    completed['cumulative_pnl'] = completed['pnl'].cumsum()
    completed['peak'] = completed['cumulative_pnl'].cummax()
    completed['drawdown'] = ((completed['cumulative_pnl'] - completed['peak']) / completed['peak']) * 100
    completed['drawdown'] = completed['drawdown'].fillna(0)
    
    # Create figure
    fig = px.area(
        completed,
        x='date',
        y='drawdown',
        title='Equity Drawdown (%)',
        labels={'drawdown': 'Drawdown (%)', 'date': 'Date'},
        color_discrete_sequence=['rgba(255, 0, 0, 0.5)']
    )
    
    # Mark maximum drawdown
    max_dd_idx = completed['drawdown'].idxmin()
    if not pd.isna(max_dd_idx):
        max_dd_date = completed.loc[max_dd_idx, 'date']
        max_dd_value = completed.loc[max_dd_idx, 'drawdown']
        
        fig.add_trace(
            go.Scatter(
                x=[max_dd_date],
                y=[max_dd_value],
                mode='markers',
                marker=dict(color='darkred', size=10, symbol='circle'),
                name=f'Max Drawdown: {max_dd_value:.2f}%'
            )
        )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        yaxis=dict(autorange="reversed"),  # Invert y-axis for better visualization of drawdowns
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_win_loss_chart(df):
    """
    Create a win/loss distribution chart
    """
    if df.empty or 'pnl' not in df.columns:
        return None
    
    # Filter for completed trades only
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return None
    
    # Classify trades
    completed['result'] = completed['pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss')
    
    # Count wins and losses
    results = completed['result'].value_counts().reset_index()
    results.columns = ['Result', 'Count']
    
    # Create pie chart
    fig = px.pie(
        results,
        names='Result',
        values='Count',
        title='Win/Loss Distribution',
        color='Result',
        color_discrete_map={'Win': '#5cb85c', 'Loss': '#d9534f'},
        hole=0.4
    )
    
    # Win rate annotation
    win_rate = (results[results['Result'] == 'Win']['Count'].sum() / results['Count'].sum()) * 100 if not results.empty else 0
    
    fig.add_annotation(
        text=f"{win_rate:.1f}%<br>Win Rate",
        x=0.5,
        y=0.5,
        font_size=20,
        showarrow=False
    )
    
    # Customize layout
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=400
    )
    
    return fig

def create_pnl_distribution(df):
    """
    Create a P&L distribution histogram
    """
    if df.empty or 'pnl' not in df.columns:
        return None
    
    # Filter for completed trades only
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty or len(completed) < 2:
        return None
    
    # Create figure
    fig = px.histogram(
        completed,
        x='pnl',
        nbins=min(20, len(completed)),
        title='P&L Distribution',
        labels={'pnl': 'Profit/Loss', 'count': 'Number of Trades'},
        color_discrete_sequence=['rgba(53, 133, 255, 0.7)']
    )
    
    # Add mean and median markers
    mean_pnl = completed['pnl'].mean()
    median_pnl = completed['pnl'].median()
    
    fig.add_vline(
        x=mean_pnl,
        line_width=2,
        line_dash="solid",
        line_color="green",
        annotation_text=f"Mean: ${mean_pnl:.2f}",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=median_pnl,
        line_width=2,
        line_dash="dash",
        line_color="purple",
        annotation_text=f"Median: ${median_pnl:.2f}",
        annotation_position="top left"
    )
    
    fig.add_vline(
        x=0,
        line_width=1,
        line_dash="dot",
        line_color="black"
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Profit/Loss ($)',
        yaxis_title='Number of Trades',
        bargap=0.1,
        height=400
    )
    
    return fig

def create_market_performance_chart(df):
    """
    Create a market performance chart
    """
    if df.empty or 'symbol' not in df.columns or 'pnl' not in df.columns:
        return None
    
    # Filter for completed trades only
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return None
    
    # Group by symbol
    symbol_pnl = completed.groupby('symbol')['pnl'].sum().reset_index()
    symbol_pnl = symbol_pnl.sort_values('pnl', ascending=False)
    
    # Calculate win rate
    symbol_stats = []
    for symbol in symbol_pnl['symbol']:
        symbol_trades = completed[completed['symbol'] == symbol]
        win_rate = (symbol_trades['pnl'] > 0).mean() * 100
        total_trades = len(symbol_trades)
        avg_pnl = symbol_trades['pnl'].mean()
        
        symbol_stats.append({
            'symbol': symbol,
            'win_rate': win_rate,
            'trades': total_trades,
            'avg_pnl': avg_pnl,
            'total_pnl': symbol_trades['pnl'].sum()
        })
    
    symbol_stats_df = pd.DataFrame(symbol_stats)
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total P&L by Market', 'Win Rate by Market'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Add P&L bar chart
    fig.add_trace(
        go.Bar(
            x=symbol_stats_df['symbol'],
            y=symbol_stats_df['total_pnl'],
            marker_color=['green' if x > 0 else 'red' for x in symbol_stats_df['total_pnl']],
            name='Total P&L',
            text=symbol_stats_df['total_pnl'].apply(lambda x: f'${x:.2f}'),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Add win rate chart
    fig.add_trace(
        go.Bar(
            x=symbol_stats_df['symbol'],
            y=symbol_stats_df['win_rate'],
            marker_color='rgba(55, 128, 191, 0.7)',
            name='Win Rate',
            text=symbol_stats_df['win_rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Customize layout
    fig.update_layout(
        title_text='Market Performance Analysis',
        showlegend=False,
        height=450
    )
    
    fig.update_xaxes(title_text='Market', row=1, col=1)
    fig.update_xaxes(title_text='Market', row=1, col=2)
    fig.update_yaxes(title_text='Total P&L ($)', row=1, col=1)
    fig.update_yaxes(title_text='Win Rate (%)', row=1, col=2)
    
    return fig

def create_ict_concept_performance(df):
    """
    Create a performance chart for various ICT concepts
    """
    if df.empty:
        return None
    
    # Filter for completed trades only
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return None
    
    # ICT concept columns to analyze (if they exist)
    ict_concepts = [
        'order_block_entry', 'liquidity_grab', 'fvg_entry', 
        'breaker_block', 'imbalance_entry', 'inducement_identified'
    ]
    
    # Filter to only include columns that exist in the dataframe
    existing_concepts = [col for col in ict_concepts if col in completed.columns]
    
    if not existing_concepts:
        return None
    
    # Analyze performance for each concept
    concept_stats = []
    
    for concept in existing_concepts:
        # Trades using this concept
        concept_trades = completed[completed[concept] == True]
        
        if len(concept_trades) > 0:
            win_rate = (concept_trades['pnl'] > 0).mean() * 100
            avg_pnl = concept_trades['pnl'].mean()
            total_trades = len(concept_trades)
            
            # Make concept name more readable
            concept_name = concept.replace('_', ' ').title()
            
            concept_stats.append({
                'concept': concept_name,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'trades': total_trades
            })
    
    if not concept_stats:
        return None
    
    concept_stats_df = pd.DataFrame(concept_stats)
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Win Rate by ICT Concept', 'Average P&L by ICT Concept'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Add win rate bar chart
    fig.add_trace(
        go.Bar(
            x=concept_stats_df['concept'],
            y=concept_stats_df['win_rate'],
            marker_color='rgba(55, 128, 191, 0.7)',
            name='Win Rate',
            text=concept_stats_df['win_rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Add average P&L chart
    fig.add_trace(
        go.Bar(
            x=concept_stats_df['concept'],
            y=concept_stats_df['avg_pnl'],
            marker_color=['green' if x > 0 else 'red' for x in concept_stats_df['avg_pnl']],
            name='Average P&L',
            text=concept_stats_df['avg_pnl'].apply(lambda x: f'${x:.2f}'),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Customize layout
    fig.update_layout(
        title_text='ICT Concept Performance Analysis',
        showlegend=False,
        height=450
    )
    
    fig.update_xaxes(title_text='ICT Concept', row=1, col=1)
    fig.update_xaxes(title_text='ICT Concept', row=1, col=2)
    fig.update_yaxes(title_text='Win Rate (%)', row=1, col=1)
    fig.update_yaxes(title_text='Average P&L ($)', row=1, col=2)
    
    return fig

def create_backtest_vs_forward_chart(df):
    """
    Create a comparison chart between backtest and forward test results
    """
    if df.empty or 'test_type' not in df.columns:
        return None
    
    # Filter for completed trades only
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return None
    
    # Separate backtest and forward test data
    backtest = completed[completed['test_type'] == 'Backtest']
    forward = completed[completed['test_type'] == 'Forward']
    
    if backtest.empty or forward.empty:
        return None
    
    # Calculate metrics
    metrics = [
        {
            'metric': 'Win Rate',
            'Backtest': (backtest['pnl'] > 0).mean() * 100,
            'Forward': (forward['pnl'] > 0).mean() * 100,
            'format': '{:.1f}%'
        },
        {
            'metric': 'Average P&L',
            'Backtest': backtest['pnl'].mean(),
            'Forward': forward['pnl'].mean(),
            'format': '${:.2f}'
        },
        {
            'metric': 'Total P&L',
            'Backtest': backtest['pnl'].sum(),
            'Forward': forward['pnl'].sum(),
            'format': '${:.2f}'
        }
    ]
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(metrics)
    
    # Create figure
    fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=[m['metric'] for m in metrics],
        vertical_spacing=0.1
    )
    
    # Add bar charts for each metric
    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=['Backtest', 'Forward'],
                y=[metric['Backtest'], metric['Forward']],
                text=[
                    metric['format'].format(metric['Backtest']),
                    metric['format'].format(metric['Forward'])
                ],
                textposition='auto',
                marker_color=['rgba(55, 128, 191, 0.7)', 'rgba(219, 64, 82, 0.7)'],
                name=metric['metric']
            ),
            row=i+1, col=1
        )
    
    # Customize layout
    fig.update_layout(
        title_text='Backtest vs. Forward Test Comparison',
        showlegend=False,
        height=150 * len(metrics)
    )
    
    return fig

def create_psychology_correlation_chart(psychology_df, trades_df):
    """
    Create a chart showing correlation between psychological factors and trading performance
    """
    if psychology_df.empty or trades_df.empty:
        return None
    
    # Filter for completed trades only
    completed_trades = trades_df[trades_df['status'] == 'Completed'].copy()
    
    if completed_trades.empty:
        return None
    
    # Ensure dates are datetime
    psychology_df['date'] = pd.to_datetime(psychology_df['date'])
    completed_trades['date'] = pd.to_datetime(completed_trades['date'])
    
    # Extract date part only
    psychology_df['date_only'] = psychology_df['date'].dt.date
    completed_trades['date_only'] = completed_trades['date'].dt.date
    
    # Group trades by date
    daily_pnl = completed_trades.groupby('date_only')['pnl'].sum().reset_index()
    
    # Psychology factors to analyze
    psych_factors = ['mental_state', 'focus', 'discipline', 'confidence', 'patience']
    
    # Filter to only include columns that exist in the dataframe
    existing_factors = [col for col in psych_factors if col in psychology_df.columns]
    
    if not existing_factors:
        return None
    
    # Prepare merged data for each factor
    correlations = []
    
    for factor in existing_factors:
        # Get daily average for the factor
        factor_data = psychology_df.groupby('date_only')[factor].mean().reset_index()
        
        # Merge with PnL data
        merged = pd.merge(daily_pnl, factor_data, on='date_only', how='inner')
        
        if len(merged) > 1:
            corr = merged[factor].corr(merged['pnl'])
            correlations.append({
                'factor': factor.replace('_', ' ').title(),
                'correlation': corr
            })
    
    if not correlations:
        return None
    
    corr_df = pd.DataFrame(correlations)
    
    # Create figure
    fig = px.bar(
        corr_df,
        x='factor',
        y='correlation',
        title='Correlation: Psychological Factors vs. Trading Performance',
        labels={'factor': 'Psychological Factor', 'correlation': 'Correlation with Daily P&L'},
        color='correlation',
        color_continuous_scale=['red', 'white', 'green'],
        range_color=[-1, 1]
    )
    
    # Add reference line at zero
    fig.add_hline(
        y=0,
        line_width=1,
        line_dash="dash",
        line_color="black"
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Psychological Factor',
        yaxis_title='Correlation Coefficient',
        coloraxis_showscale=False,
        height=400
    )
    
    return fig

def create_monthly_performance_chart(df):
    """
    Create a monthly performance chart
    """
    if df.empty or 'date' not in df.columns:
        return None
    
    # Filter for completed trades only
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return None
    
    # Ensure date is datetime
    completed['date'] = pd.to_datetime(completed['date'])
    
    # Extract year and month
    completed['year_month'] = completed['date'].dt.strftime('%Y-%m')
    
    # Group by year-month
    monthly_pnl = completed.groupby('year_month')['pnl'].sum().reset_index()
    monthly_pnl = monthly_pnl.sort_values('year_month')
    
    # Calculate cumulative P&L
    monthly_pnl['cumulative_pnl'] = monthly_pnl['pnl'].cumsum()
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly P&L', 'Cumulative P&L'),
        vertical_spacing=0.15
    )
    
    # Add monthly P&L bar chart
    fig.add_trace(
        go.Bar(
            x=monthly_pnl['year_month'],
            y=monthly_pnl['pnl'],
            marker_color=['green' if x > 0 else 'red' for x in monthly_pnl['pnl']],
            name='Monthly P&L',
            text=monthly_pnl['pnl'].apply(lambda x: f'${x:.2f}'),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Add cumulative P&L line chart
    fig.add_trace(
        go.Scatter(
            x=monthly_pnl['year_month'],
            y=monthly_pnl['cumulative_pnl'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Customize layout
    fig.update_layout(
        title_text='Monthly Performance Analysis',
        showlegend=False,
        height=600
    )
    
    fig.update_xaxes(title_text='Month', row=1, col=1)
    fig.update_xaxes(title_text='Month', row=2, col=1)
    fig.update_yaxes(title_text='P&L ($)', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative P&L ($)', row=2, col=1)
    
    return fig

def create_day_of_week_chart(df):
    """
    Create a day of week performance chart
    """
    if df.empty or 'date' not in df.columns:
        return None
    
    # Filter for completed trades only
    completed = df[df['status'] == 'Completed'].copy()
    
    if completed.empty:
        return None
    
    # Ensure date is datetime
    completed['date'] = pd.to_datetime(completed['date'])
    
    # Extract day of week
    completed['day_of_week'] = completed['date'].dt.day_name()
    
    # Define correct order of days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Group by day of week
    day_stats = []
    for day in day_order:
        day_trades = completed[completed['day_of_week'] == day]
        
        if len(day_trades) > 0:
            day_stats.append({
                'day': day,
                'pnl': day_trades['pnl'].sum(),
                'win_rate': (day_trades['pnl'] > 0).mean() * 100,
                'trades': len(day_trades)
            })
    
    if not day_stats:
        return None
    
    day_stats_df = pd.DataFrame(day_stats)
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('P&L by Day of Week', 'Win Rate by Day of Week'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Add P&L bar chart
    fig.add_trace(
        go.Bar(
            x=day_stats_df['day'],
            y=day_stats_df['pnl'],
            marker_color=['green' if x > 0 else 'red' for x in day_stats_df['pnl']],
            name='P&L',
            text=day_stats_df['pnl'].apply(lambda x: f'${x:.2f}'),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Add win rate chart
    fig.add_trace(
        go.Bar(
            x=day_stats_df['day'],
            y=day_stats_df['win_rate'],
            marker_color='rgba(55, 128, 191, 0.7)',
            name='Win Rate',
            text=day_stats_df['win_rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Customize layout
    fig.update_layout(
        title_text='Day of Week Performance',
        showlegend=False,
        height=450
    )
    
    fig.update_xaxes(title_text='Day', row=1, col=1, categoryorder='array', categoryarray=day_order)
    fig.update_xaxes(title_text='Day', row=1, col=2, categoryorder='array', categoryarray=day_order)
    fig.update_yaxes(title_text='P&L ($)', row=1, col=1)
    fig.update_yaxes(title_text='Win Rate (%)', row=1, col=2)
    
    return fig
