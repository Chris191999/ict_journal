import streamlit as st
import pandas as pd
import os
import datetime
from utils.data_manager import initialize_session_state, save_session_data, load_session_data

# Initialize the app
st.set_page_config(
    page_title="ICT Trading Journal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for the app
initialize_session_state()

# Sidebar navigation
st.sidebar.title("ICT Trading Journal")
st.sidebar.image("https://img.icons8.com/fluency/96/000000/investment-portfolio.png", width=80)

pages = {
    "Dashboard": "dashboard",
    "Journal Entries": "journal",
    "Performance Analytics": "analytics",
    "Psychology Tracker": "psychology",
    "Import/Export": "import_export"
}

selection = st.sidebar.radio("Navigate", list(pages.keys()))

# Session status indicator
if st.session_state.trades:
    st.sidebar.success(f"Journal Active: {len(st.session_state.trades)} trades recorded")
else:
    st.sidebar.warning("No trade data loaded. Add entries or import data.")

# Load external pages based on selection
if selection == "Dashboard":
    # Dashboard page (main page)
    st.title("ICT Trading Journal Dashboard")
    
    # Summary metrics row
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    if st.session_state.trades:
        # Calculate summary metrics
        trades_df = pd.DataFrame(st.session_state.trades)
        
        # Filter for completed trades only for certain metrics
        completed_trades = trades_df[trades_df['status'] == 'Completed']
        
        # Calculate basic metrics
        total_trades = len(trades_df)
        completed_count = len(completed_trades)
        
        if completed_count > 0:
            winning_trades = len(completed_trades[completed_trades['pnl'] > 0])
            win_rate = winning_trades / completed_count * 100
            
            # Calculate profit metrics
            total_profit = completed_trades[completed_trades['pnl'] > 0]['pnl'].sum()
            total_loss = abs(completed_trades[completed_trades['pnl'] < 0]['pnl'].sum())
            
            if total_loss > 0:
                profit_factor = total_profit / total_loss
            else:
                profit_factor = float('inf') if total_profit > 0 else 0
                
            # Display metrics
            metrics_col1.metric("Total Trades", total_trades)
            metrics_col2.metric("Win Rate", f"{win_rate:.2f}%")
            metrics_col3.metric("Profit Factor", f"{profit_factor:.2f}")
            
            # Calculate net P&L
            net_pnl = completed_trades['pnl'].sum()
            metrics_col4.metric("Net P&L", f"${net_pnl:.2f}", 
                               delta=f"{net_pnl:.2f}")
        else:
            metrics_col1.metric("Total Trades", total_trades)
            metrics_col2.metric("Win Rate", "N/A")
            metrics_col3.metric("Profit Factor", "N/A")
            metrics_col4.metric("Net P&L", "N/A")
        
        # Recent trades
        st.subheader("Recent Trades")
        if not trades_df.empty:
            # Sort by date (most recent first)
            recent_trades = trades_df.sort_values(by='date', ascending=False).head(5)
            # Display only relevant columns
            display_cols = ['date', 'symbol', 'direction', 'order_type', 'entry', 'exit', 'pnl', 'status']
            st.dataframe(recent_trades[display_cols], use_container_width=True)
        else:
            st.info("No trades recorded yet.")
            
        # Market distribution chart
        st.subheader("Trading Activity by Market")
        market_counts = trades_df['symbol'].value_counts()
        if len(market_counts) > 0:
            st.bar_chart(market_counts)
        else:
            st.info("No market data available.")
            
        # Trading performance over time
        st.subheader("Cumulative P&L Over Time")
        if not completed_trades.empty:
            # Ensure the date is in datetime format
            completed_trades['date'] = pd.to_datetime(completed_trades['date'])
            completed_trades = completed_trades.sort_values(by='date')
            
            # Create cumulative P&L
            completed_trades['cumulative_pnl'] = completed_trades['pnl'].cumsum()
            
            # Plot the cumulative P&L
            st.line_chart(completed_trades.set_index('date')['cumulative_pnl'])
        else:
            st.info("No completed trades to display performance over time.")
            
    else:
        # No data state
        metrics_col1.metric("Total Trades", 0)
        metrics_col2.metric("Win Rate", "N/A")
        metrics_col3.metric("Profit Factor", "N/A")
        metrics_col4.metric("Net P&L", "N/A")
        
        st.info("Your journal is empty. Start by adding trade entries or import existing data.")
        st.markdown("""
        ## Getting Started
        1. Go to the **Journal Entries** page to add individual trades
        2. Use the **Import/Export** page to load your existing trade data
        3. Review your performance in the **Analytics** section
        4. Track trading psychology in the **Psychology Tracker**
        """)

    # Quick Actions
    st.subheader("Quick Actions")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("Add New Trade", use_container_width=True):
            st.session_state.page = "journal"
            st.rerun()
            
    with quick_col2:
        if st.button("View Analytics", use_container_width=True):
            st.session_state.page = "analytics"
            st.rerun()
            
    with quick_col3:
        if st.button("Import/Export Data", use_container_width=True):
            st.session_state.page = "import_export"
            st.rerun()

elif selection == "Journal Entries":
    # Import the journal page
    from pages.journal import show_journal_page
    show_journal_page()
    
elif selection == "Performance Analytics":
    # Import the analytics page
    from pages.analytics import show_analytics_page
    show_analytics_page()
    
elif selection == "Psychology Tracker":
    # Import the psychology page
    from pages.psychology import show_psychology_page
    show_psychology_page()
    
elif selection == "Import/Export":
    # Import the import/export page
    from pages.import_export import show_import_export_page
    show_import_export_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("ICT Trading Journal v1.0")
