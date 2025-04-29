import streamlit as st
import pandas as pd
import datetime
import base64
from io import BytesIO
import json
import zipfile
from utils.data_manager import get_trades_df, get_psychology_df, import_trades_from_csv, import_psychology_from_csv, save_session_data, load_session_data
from utils.image_handler import export_with_images, import_with_images

def show_import_export_page():
    """Display the import/export page for trade data"""
    st.title("Import & Export Trade Data")
    
    # Create tabs for import and export
    import_tab, export_tab, session_tab = st.tabs(["Import Data", "Export Data", "Session Management"])
    
    with import_tab:
        show_import_section()
    
    with export_tab:
        show_export_section()
    
    with session_tab:
        show_session_management()

def show_import_section():
    """Display the import section"""
    st.subheader("Import Trade Data")
    
    # Instructions
    st.write("""
    Import your trade data from CSV files or previous exports. 
    You can import basic trade data from CSV or full data including images from ZIP files.
    """)
    
    # Import options
    import_option = st.radio(
        "Select Import Format",
        options=["CSV File (Basic Trade Data)", "ZIP File (Complete Data with Images)", "Psychology Data CSV"]
    )
    
    if import_option == "CSV File (Basic Trade Data)":
        csv_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if csv_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(csv_file)
                
                # Display preview
                st.write("Preview of data to be imported:")
                st.dataframe(df.head())
                
                # Import options
                replace_existing = st.checkbox("Replace existing trades", value=False)
                
                # Import button
                if st.button("Import CSV Data"):
                    # Try to import as trade data
                    success, message = import_trades_from_csv(df, replace=replace_existing)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    elif import_option == "Psychology Data CSV":
        psych_csv_file = st.file_uploader("Upload Psychology CSV File", type=["csv"], key="psychology_csv")
        
        if psych_csv_file is not None:
            try:
                # Read CSV
                psych_df = pd.read_csv(psych_csv_file)
                
                # Display preview
                st.write("Preview of psychology data to be imported:")
                st.dataframe(psych_df.head())
                
                # Import options
                replace_existing = st.checkbox("Replace existing psychology entries", value=False, key="psych_replace")
                
                # Import button
                if st.button("Import Psychology Data"):
                    # Import as psychology data
                    success, message = import_psychology_from_csv(psych_df, replace=replace_existing)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            except Exception as e:
                st.error(f"Error reading psychology CSV file: {str(e)}")
    
    else:  # ZIP File import
        zip_file = st.file_uploader("Upload ZIP File", type=["zip"])
        
        if zip_file is not None:
            st.info("Importing ZIP file with images. This may take a moment for large files.")
            
            # Import options
            replace_existing = st.checkbox("Replace existing trades", value=False)
            
            # Import button
            if st.button("Import ZIP Data"):
                try:
                    # Import with images
                    imported_df = import_with_images(zip_file)
                    
                    if imported_df is not None:
                        # Import trades
                        success, message = import_trades_from_csv(imported_df, replace=replace_existing)
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.error("Failed to import ZIP file. The file format may be invalid.")
                except Exception as e:
                    st.error(f"Error importing ZIP file: {str(e)}")

def show_export_section():
    """Display the export section"""
    st.subheader("Export Trade Data")
    
    # Get trades dataframe
    trades_df = get_trades_df()
    
    if trades_df.empty:
        st.info("No trades to export. Add trades in the Journal Entries page first.")
        return
    
    # Display count of trades to export
    st.write(f"Found {len(trades_df)} trades to export.")
    
    # Export options
    export_option = st.radio(
        "Select Export Format",
        options=["CSV File (Basic Trade Data)", "ZIP File (Complete Data with Images)", "Psychology Data CSV"]
    )
    
    # Filter options
    with st.expander("Export Filters", expanded=False):
        # Date range
        st.write("**Date Range**")
        date_range = st.date_input(
            "Filter by date",
            value=(
                pd.to_datetime(trades_df['date']).min().date() if 'date' in trades_df.columns else datetime.date.today() - datetime.timedelta(days=30),
                pd.to_datetime(trades_df['date']).max().date() if 'date' in trades_df.columns else datetime.date.today()
            )
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Symbol filter
            all_symbols = trades_df['symbol'].unique().tolist() if 'symbol' in trades_df.columns else []
            selected_symbols = st.multiselect("Symbols", options=all_symbols, default=[])
        
        with col2:
            # Test type filter
            all_test_types = trades_df['test_type'].unique().tolist() if 'test_type' in trades_df.columns else []
            selected_test_types = st.multiselect("Test Types", options=all_test_types, default=[])
    
    # Apply filters
    filtered_df = trades_df.copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['date']).dt.date >= start_date) & 
                               (pd.to_datetime(filtered_df['date']).dt.date <= end_date)]
    
    if selected_symbols:
        filtered_df = filtered_df[filtered_df['symbol'].isin(selected_symbols)]
    
    if selected_test_types:
        filtered_df = filtered_df[filtered_df['test_type'].isin(selected_test_types)]
    
    st.write(f"Exporting {len(filtered_df)} trades after filtering.")
    
    # Generate export
    if export_option == "CSV File (Basic Trade Data)":
        if st.button("Export as CSV"):
            # Export to CSV
            csv_buffer = BytesIO()
            filtered_df.to_csv(csv_buffer, index=False)
            
            # Get psychology data
            psychology_df = get_psychology_df()
            
            # Create download link
            b64 = base64.b64encode(csv_buffer.getvalue()).decode()
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ict_trades_export_{now}.csv"
            
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.success(f"Generated CSV export with {len(filtered_df)} trades.")
            
            # Also export psychology data
            if not psychology_df.empty:
                psych_csv_buffer = BytesIO()
                psychology_df.to_csv(psych_csv_buffer, index=False)
                psych_b64 = base64.b64encode(psych_csv_buffer.getvalue()).decode()
                psych_filename = f"ict_psychology_export_{now}.csv"
                psych_href = f'<a href="data:file/csv;base64,{psych_b64}" download="{psych_filename}">Download Psychology CSV File</a>'
                st.markdown(psych_href, unsafe_allow_html=True)
                st.success(f"Also generated psychology data export with {len(psychology_df)} entries.")
    
    elif export_option == "ZIP File (Complete Data with Images)":
        include_images = "images" in filtered_df.columns
        
        if st.button("Export as ZIP"):
            st.info("Generating ZIP export. This may take a moment for large datasets with images.")
            
            try:
                # Export with images
                zip_buffer = export_with_images(filtered_df)
                
                if zip_buffer:
                    # Create download link
                    b64 = base64.b64encode(zip_buffer.getvalue()).decode()
                    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ict_trades_with_images_{now}.zip"
                    
                    href = f'<a href="data:application/zip;base64,{b64}" download="{filename}">Download ZIP File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    image_count = sum(len(row.get('images', [])) for _, row in filtered_df.iterrows() 
                                     if isinstance(row.get('images'), list))
                    st.success(f"Generated ZIP export with {len(filtered_df)} trades and {image_count} images.")
                    
                    # Also export psychology data
                    psychology_df = get_psychology_df()
                    if not psychology_df.empty:
                        psych_csv_buffer = BytesIO()
                        psychology_df.to_csv(psych_csv_buffer, index=False)
                        psych_b64 = base64.b64encode(psych_csv_buffer.getvalue()).decode()
                        psych_filename = f"ict_psychology_export_{now}.csv"
                        psych_href = f'<a href="data:file/csv;base64,{psych_b64}" download="{psych_filename}">Download Psychology CSV File</a>'
                        st.markdown(psych_href, unsafe_allow_html=True)
                        st.success(f"Also generated psychology data export with {len(psychology_df)} entries.")
                else:
                    st.error("Failed to generate ZIP export.")
            except Exception as e:
                st.error(f"Error creating ZIP export: {str(e)}")
    
    elif export_option == "Psychology Data CSV":
        psychology_df = get_psychology_df()
        
        if psychology_df.empty:
            st.warning("No psychology data available to export.")
        else:
            # Date range filter for psychology data
            if len(date_range) == 2:
                start_date, end_date = date_range
                psychology_df['date'] = pd.to_datetime(psychology_df['date'])
                filtered_psychology_df = psychology_df[
                    (psychology_df['date'].dt.date >= start_date) & 
                    (psychology_df['date'].dt.date <= end_date)
                ]
            else:
                filtered_psychology_df = psychology_df.copy()
            
            st.write(f"Exporting {len(filtered_psychology_df)} psychology entries.")
            
            if st.button("Export Psychology Data"):
                # Export to CSV
                csv_buffer = BytesIO()
                filtered_psychology_df.to_csv(csv_buffer, index=False)
                
                # Create download link
                b64 = base64.b64encode(csv_buffer.getvalue()).decode()
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ict_psychology_export_{now}.csv"
                
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Psychology CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.success(f"Generated CSV export with {len(filtered_psychology_df)} psychology entries.")

def show_session_management():
    """Display session management options"""
    st.subheader("Session Management")
    
    st.write("""
    Save or load your entire trading journal session, including all trades, psychology entries, and images.
    This allows you to back up your data or transfer it to another device.
    """)
    
    # Save session
    save_col, load_col = st.columns(2)
    
    with save_col:
        st.write("**Save Current Session**")
        
        if st.button("Save Session"):
            b64, filename = save_session_data()
            
            # Create download link
            href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download Session File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.success(f"Session data saved to {filename}")
    
    with load_col:
        st.write("**Load Saved Session**")
        
        session_file = st.file_uploader("Upload Session File", type=["json"])
        
        if session_file is not None:
            replace_session = st.checkbox("Replace current session (rather than merge)", value=True)
            
            if st.button("Load Session"):
                try:
                    if load_session_data(session_file):
                        st.success("Session loaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to load session data.")
                except Exception as e:
                    st.error(f"Error loading session: {str(e)}")
    
    # Session statistics
    st.subheader("Current Session Statistics")
    
    trades_df = get_trades_df()
    psychology_df = get_psychology_df()
    
    trade_count = len(trades_df) if not trades_df.empty else 0
    psychology_count = len(psychology_df) if not psychology_df.empty else 0
    
    # Count images
    image_count = 0
    if not trades_df.empty and 'images' in trades_df.columns:
        for _, row in trades_df.iterrows():
            if isinstance(row.get('images'), list):
                image_count += len(row['images'])
    
    # Display stats
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    stats_col1.metric("Total Trades", trade_count)
    stats_col2.metric("Psychology Entries", psychology_count)
    stats_col3.metric("Trade Images", image_count)
    
    # Clear session
    st.subheader("Clear Session Data")
    st.warning("This will delete all your current trading data. This action cannot be undone.")
    
    confirm_clear = st.checkbox("I understand this will delete all my trading data")
    
    if confirm_clear and st.button("Clear All Session Data"):
        st.session_state.trades = []
        st.session_state.psychology_entries = []
        st.session_state.current_trade = {}
        st.session_state.edit_index = -1
        
        st.success("All session data has been cleared.")
        st.rerun()

