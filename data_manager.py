import streamlit as st
import pandas as pd
import json
import os
import datetime
import base64
from io import BytesIO

def initialize_session_state():
    """Initialize all required session state variables"""
    if 'trades' not in st.session_state:
        st.session_state.trades = []
        
    if 'psychology_entries' not in st.session_state:
        st.session_state.psychology_entries = []
        
    if 'current_trade' not in st.session_state:
        st.session_state.current_trade = {}
        
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
        
    if 'edit_index' not in st.session_state:
        st.session_state.edit_index = -1
        
    if 'filters' not in st.session_state:
        st.session_state.filters = {
            'date_range': None,
            'symbols': [],
            'direction': [],
            'outcome': [],
            'status': []
        }

def save_session_data():
    """Save the current session data to a JSON file for download"""
    session_data = {
        'trades': st.session_state.trades,
        'psychology_entries': st.session_state.psychology_entries
    }
    
    json_str = json.dumps(session_data, indent=4, default=str)
    
    # Convert to bytes for download
    b64 = base64.b64encode(json_str.encode()).decode()
    
    # Generate a download link
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ict_journal_session_{now}.json"
    
    return b64, filename

def load_session_data(uploaded_file):
    """Load session data from an uploaded JSON file"""
    try:
        content = uploaded_file.read().decode()
        data = json.loads(content)
        
        # Validate data structure
        if 'trades' not in data or 'psychology_entries' not in data:
            st.error("Invalid session file format.")
            return False
            
        # Update session state
        st.session_state.trades = data['trades']
        st.session_state.psychology_entries = data['psychology_entries']
        
        return True
    except Exception as e:
        st.error(f"Error loading session data: {str(e)}")
        return False

def add_trade(trade_data):
    """Add a new trade to the session state"""
    # Generate a unique ID for the trade
    if 'id' not in trade_data:
        # Use timestamp as ID if not provided
        trade_data['id'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    st.session_state.trades.append(trade_data)
    return True

def update_trade(index, trade_data):
    """Update an existing trade at the given index"""
    if 0 <= index < len(st.session_state.trades):
        # Preserve the original ID
        original_id = st.session_state.trades[index]['id']
        trade_data['id'] = original_id
        
        st.session_state.trades[index] = trade_data
        return True
    return False

def delete_trade(index):
    """Delete a trade at the given index"""
    if 0 <= index < len(st.session_state.trades):
        del st.session_state.trades[index]
        return True
    return False

def get_trades_df():
    """Convert trades list to a pandas DataFrame for analysis"""
    if not st.session_state.trades:
        return pd.DataFrame()
        
    df = pd.DataFrame(st.session_state.trades)
    
    # Convert date strings to datetime objects
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    return df

def add_psychology_entry(entry_data):
    """Add a new psychology entry to the session state"""
    if 'id' not in entry_data:
        entry_data['id'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
    st.session_state.psychology_entries.append(entry_data)
    return True

def get_psychology_df():
    """Convert psychology entries to a pandas DataFrame"""
    if not st.session_state.psychology_entries:
        return pd.DataFrame()
        
    df = pd.DataFrame(st.session_state.psychology_entries)
    
    # Convert date strings to datetime objects
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    return df

def filter_trades(df, filters):
    """Apply filters to the trades DataFrame"""
    filtered_df = df.copy()
    
    # Apply date range filter
    if filters['date_range'] and len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        filtered_df = filtered_df[(filtered_df['date'].dt.date >= start_date) & 
                                 (filtered_df['date'].dt.date <= end_date)]
    
    # Apply symbol filter
    if filters['symbols']:
        filtered_df = filtered_df[filtered_df['symbol'].isin(filters['symbols'])]
    
    # Apply direction filter
    if filters['direction']:
        filtered_df = filtered_df[filtered_df['direction'].isin(filters['direction'])]
    
    # Apply outcome filter (win/loss)
    if filters['outcome']:
        if 'Win' in filters['outcome']:
            win_condition = filtered_df['pnl'] > 0
            filtered_df = filtered_df[win_condition if 'Loss' not in filters['outcome'] 
                                     else win_condition | (filtered_df['pnl'] < 0)]
        elif 'Loss' in filters['outcome']:
            filtered_df = filtered_df[filtered_df['pnl'] < 0]
    
    # Apply status filter
    if filters['status']:
        filtered_df = filtered_df[filtered_df['status'].isin(filters['status'])]
    
    # Apply test_type filter
    if 'test_type' in filters and filters['test_type']:
        filtered_df = filtered_df[filtered_df['test_type'].isin(filters['test_type'])]
    
    return filtered_df

def export_trades_to_csv(include_images=False):
    """Export trades to a CSV file (with optional image data)"""
    trades_df = get_trades_df()
    
    if trades_df.empty:
        return None, None
    
    # Convert DataFrame to CSV
    csv_buffer = BytesIO()
    
    # If not including images, remove image columns for cleaner CSV
    export_df = trades_df.copy()
    if not include_images and 'images' in export_df.columns:
        export_df = export_df.drop(columns=['images'])
    
    export_df.to_csv(csv_buffer, index=False)
    
    # Convert to base64 for download
    b64 = base64.b64encode(csv_buffer.getvalue()).decode()
    
    # Generate filename
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ict_trades_export_{now}.csv"
    
    return b64, filename

def import_trades_from_csv(df, replace=False):
    """Import trades from a DataFrame (parsed from CSV)"""
    try:
        # Validate required columns
        required_columns = ['symbol', 'direction', 'entry', 'exit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Process the DataFrame
        trades_list = df.to_dict('records')
        
        # Clear existing trades if replace is True
        if replace:
            st.session_state.trades = []
        
        # Add the imported trades
        for trade in trades_list:
            # Generate IDs for new trades
            if 'id' not in trade:
                trade['id'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + str(len(st.session_state.trades))
            
            st.session_state.trades.append(trade)
        
        return True, f"Successfully imported {len(trades_list)} trades."
    
    except Exception as e:
        return False, f"Error importing trades: {str(e)}"
        
def import_psychology_from_csv(df, replace=False):
    """Import psychology entries from a DataFrame (parsed from CSV)"""
    try:
        # Validate that this is a psychology dataframe
        psychology_columns = ['mental_state', 'focus', 'discipline', 'confidence']
        valid_dataframe = any(col in df.columns for col in psychology_columns)
        
        if not valid_dataframe:
            return False, "This doesn't appear to be a psychology data file. Missing required columns."
        
        # Process the DataFrame
        psychology_list = df.to_dict('records')
        
        # Clear existing entries if replace is True
        if replace:
            st.session_state.psychology_entries = []
        
        # Add the imported entries
        for entry in psychology_list:
            # Generate IDs for new entries if needed
            if 'id' not in entry:
                entry['id'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + str(len(st.session_state.psychology_entries))
            
            st.session_state.psychology_entries.append(entry)
        
        return True, f"Successfully imported {len(psychology_list)} psychology entries."
    
    except Exception as e:
        return False, f"Error importing psychology data: {str(e)}"
