import streamlit as st
import pandas as pd
import datetime
from utils.data_manager import add_trade, update_trade, delete_trade, get_trades_df, add_psychology_entry, get_psychology_df
from utils.image_handler import encode_image, display_image

def show_journal_page():
    """Display the journal entries page"""
    st.title("ICT Trading Journal Entries")
    
    # Create tabs for Add/Edit, Psychology, and View Journal
    add_tab, psychology_tab, view_tab = st.tabs(["Add/Edit Trade", "Log Psychology", "View Journal"])
    
    with add_tab:
        show_trade_form()
    
    with psychology_tab:
        show_psychology_form()
    
    with view_tab:
        show_journal_table()

def show_trade_form():
    """Display the form for adding or editing trades"""
    # Determine if we're editing or adding a trade
    is_editing = st.session_state.edit_index >= 0
    
    # Form title
    if is_editing:
        st.subheader("Edit Trade Entry")
    else:
        st.subheader("Add New Trade Entry")
    
    # Create form
    with st.form(key="trade_form"):
        # Main trade details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input("Symbol/Market", 
                                   value=st.session_state.current_trade.get('symbol', ''))
            
            direction = st.selectbox("Direction", 
                                     options=["Long", "Short"],
                                     index=0 if not st.session_state.current_trade.get('direction') 
                                     or st.session_state.current_trade.get('direction') == "Long" else 1)
            
            test_type = st.selectbox("Test Type", 
                                    options=["Forward", "Backtest"],
                                    index=0 if not st.session_state.current_trade.get('test_type') 
                                    or st.session_state.current_trade.get('test_type') == "Forward" else 1)
        
        with col2:
            date = st.date_input("Date", 
                                value=pd.to_datetime(st.session_state.current_trade.get('date', 
                                                                                      datetime.datetime.now().strftime("%Y-%m-%d"))))
            
            time = st.time_input("Time", 
                                value=datetime.datetime.now().time() if 'date' not in st.session_state.current_trade 
                                else pd.to_datetime(st.session_state.current_trade.get('date')).time())
            
            status = st.selectbox("Status", 
                                 options=["Planned", "Active", "Completed", "Cancelled"],
                                 index=2 if st.session_state.current_trade.get('status') == "Completed" 
                                 else 1 if st.session_state.current_trade.get('status') == "Active"
                                 else 3 if st.session_state.current_trade.get('status') == "Cancelled" else 0)
        
        with col3:
            order_type = st.selectbox("Order Type", 
                                     options=["Market", "Limit", "Stop", "Stop Limit"],
                                     index=0 if not st.session_state.current_trade.get('order_type') 
                                     or st.session_state.current_trade.get('order_type') == "Market" 
                                     else 1 if st.session_state.current_trade.get('order_type') == "Limit"
                                     else 2 if st.session_state.current_trade.get('order_type') == "Stop" else 3)
            
            entry = st.number_input("Entry Price", 
                                   value=float(st.session_state.current_trade.get('entry', 0.0)),
                                   format="%.4f")
            
            exit = st.number_input("Exit Price", 
                                  value=float(st.session_state.current_trade.get('exit', 0.0)),
                                  format="%.4f")
        
        # Trade metrics
        st.subheader("Trade Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            size = st.number_input("Position Size", 
                                  value=float(st.session_state.current_trade.get('size', 0.0)))
            
            risk = st.number_input("Risk Amount ($)", 
                                  value=float(st.session_state.current_trade.get('risk', 0.0)))
        
        with col2:
            planned_rrr = st.number_input("Planned RRR", 
                                         value=float(st.session_state.current_trade.get('planned_rrr', 0.0)))
            
            stop_loss = st.number_input("Stop Loss", 
                                       value=float(st.session_state.current_trade.get('planned_stop_loss', 0.0)),
                                       format="%.4f")
        
        with col3:
            take_profit = st.number_input("Take Profit", 
                                         value=float(st.session_state.current_trade.get('planned_take_profit', 0.0)),
                                         format="%.4f")
            
            actual_rrr = st.number_input("Actual RRR", 
                                        value=float(st.session_state.current_trade.get('actual_rrr', 0.0)))
        
        with col4:
            pnl = st.number_input("P&L", 
                                 value=float(st.session_state.current_trade.get('pnl', 0.0)))
            
            fees = st.number_input("Fees", 
                                  value=float(st.session_state.current_trade.get('fees', 0.0)))
        
        # ICT-specific details
        st.subheader("ICT Trade Elements")
        
        ict_col1, ict_col2, ict_col3 = st.columns(3)
        
        with ict_col1:
            market_condition = st.selectbox("Market Condition", 
                                           options=["", "Premium", "Discount", "Fair Value", "Range"],
                                           index=0 if not st.session_state.current_trade.get('market_condition')
                                           else 1 if st.session_state.current_trade.get('market_condition') == "Premium"
                                           else 2 if st.session_state.current_trade.get('market_condition') == "Discount"
                                           else 3 if st.session_state.current_trade.get('market_condition') == "Fair Value"
                                           else 4)
            
            order_block_entry = st.checkbox("Order Block Entry", 
                                          value=st.session_state.current_trade.get('order_block_entry', False))
            
            liquidity_grab = st.checkbox("Liquidity Grab", 
                                        value=st.session_state.current_trade.get('liquidity_grab', False))
        
        with ict_col2:
            fvg_entry = st.checkbox("Fair Value Gap Entry", 
                                   value=st.session_state.current_trade.get('fvg_entry', False))
            
            breaker_block = st.checkbox("Breaker Block", 
                                       value=st.session_state.current_trade.get('breaker_block', False))
            
            inducement_identified = st.checkbox("Inducement Identified", 
                                              value=st.session_state.current_trade.get('inducement_identified', False))
        
        with ict_col3:
            imbalance_entry = st.checkbox("Imbalance Entry", 
                                         value=st.session_state.current_trade.get('imbalance_entry', False))
            
            entry_hit_ob = st.checkbox("Entry Hit OB", 
                                      value=st.session_state.current_trade.get('entry_hit_ob', False))
            
            liquidity_grabbed = st.checkbox("Liquidity Grab Success", 
                                          value=st.session_state.current_trade.get('liquidity_grab_success', False))
        
        # Notes and image
        st.subheader("Notes & Images")
        
        notes = st.text_area("Trade Notes", 
                            value=st.session_state.current_trade.get('notes', ''), 
                            height=150)
        
        # Image upload
        st.caption("Upload chart images or trade setup screenshots")
        uploaded_images = st.file_uploader("Add Images to Trade", 
                                          type=["jpg", "jpeg", "png", "gif"],
                                          accept_multiple_files=True)
        
        # Display existing images
        existing_images = st.session_state.current_trade.get('images', [])
        if existing_images:
            st.write("Existing Images:")
            for i, img in enumerate(existing_images):
                display_image(img)
                
                # Option to remove image
                if st.checkbox(f"Remove Image {i+1}", key=f"remove_img_{i}"):
                    existing_images[i] = None
            
            # Filter out None values (removed images)
            existing_images = [img for img in existing_images if img is not None]
        
        # Submit button
        submit_label = "Update Trade" if is_editing else "Add Trade"
        submit_button = st.form_submit_button(label=submit_label)
        
        if submit_button:
            # Process form data
            datetime_combined = datetime.datetime.combine(date, time)
            
            # Calculate P&L if not provided
            calculated_pnl = pnl
            if status == "Completed" and pnl == 0 and entry != 0 and exit != 0 and size != 0:
                if direction == "Long":
                    calculated_pnl = (exit - entry) * size - fees
                else:  # Short
                    calculated_pnl = (entry - exit) * size - fees
            
            # Encode new uploaded images
            new_images = []
            for uploaded_file in uploaded_images:
                encoded_image = encode_image(uploaded_file)
                if encoded_image:
                    new_images.append(encoded_image)
            
            # Combine existing and new images
            combined_images = existing_images + new_images
            
            # Create trade data
            trade_data = {
                'symbol': symbol,
                'direction': direction,
                'date': datetime_combined.strftime("%Y-%m-%d %H:%M:%S"),
                'status': status,
                'test_type': test_type,
                'order_type': order_type,
                'entry': entry,
                'exit': exit,
                'size': size,
                'risk': risk,
                'planned_rrr': planned_rrr,
                'planned_stop_loss': stop_loss,
                'planned_take_profit': take_profit,
                'actual_rrr': actual_rrr,
                'pnl': calculated_pnl if calculated_pnl != 0 else pnl,
                'fees': fees,
                'market_condition': market_condition,
                'order_block_entry': order_block_entry,
                'liquidity_grab': liquidity_grab,
                'fvg_entry': fvg_entry,
                'breaker_block': breaker_block,
                'inducement_identified': inducement_identified,
                'imbalance_entry': imbalance_entry, 
                'entry_hit_ob': entry_hit_ob,
                'liquidity_grab_success': liquidity_grabbed,
                'notes': notes,
                'images': combined_images if combined_images else []
            }
            
            # Add or update trade
            if is_editing:
                success = update_trade(st.session_state.edit_index, trade_data)
                if success:
                    st.success(f"Updated trade for {symbol}")
                    # Reset edit state
                    st.session_state.edit_index = -1
                    st.session_state.current_trade = {}
                else:
                    st.error("Failed to update trade. Please try again.")
            else:
                success = add_trade(trade_data)
                if success:
                    st.success(f"Added trade for {symbol}")
                    # Clear form
                    st.session_state.current_trade = {}
                else:
                    st.error("Failed to add trade. Please try again.")
            
            # Rerun to update the interface
            st.rerun()

def show_psychology_form():
    """Display the form for adding psychology entries"""
    st.subheader("Log Your Trading Psychology")
    
    with st.form(key="trade_psychology_form"):
        # Date and time
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", value=datetime.datetime.now().date(), key="psych_date")
        
        with col2:
            time = st.time_input("Time", value=datetime.datetime.now().time(), key="psych_time")
            
        # Associate with a trade option
        trades_df = get_trades_df()
        if not trades_df.empty:
            st.write("**Associate with Trade**")
            all_trades = []
            
            for idx, row in trades_df.iterrows():
                trade_date = pd.to_datetime(row['date']).strftime("%Y-%m-%d")
                trade_desc = f"{trade_date} | {row['symbol']} | {row['direction']} | {row.get('status', 'Unknown')}"
                all_trades.append((idx, trade_desc))
            
            selected_trade_idx = st.selectbox(
                "Link to Trade (Optional)",
                options=[-1] + [t[0] for t in all_trades],
                format_func=lambda x: "None" if x == -1 else [t[1] for t in all_trades if t[0] == x][0],
            )
        else:
            selected_trade_idx = -1
        
        # Psychology ratings
        st.write("**Rate your trading psychology factors (1-10)**")
        
        rating_col1, rating_col2 = st.columns(2)
        
        with rating_col1:
            mental_state = st.slider("Overall Mental State", 1, 10, 5, 
                                   help="Your overall mental and emotional state", key="psych_mental")
            
            focus = st.slider("Focus", 1, 10, 5, 
                            help="Your ability to concentrate on trading tasks", key="psych_focus")
            
            discipline = st.slider("Discipline", 1, 10, 5, 
                                 help="Your ability to stick to your trading plan", key="psych_discipline")
        
        with rating_col2:
            confidence = st.slider("Confidence", 1, 10, 5, 
                                 help="Your belief in your trading decisions", key="psych_confidence")
            
            patience = st.slider("Patience", 1, 10, 5, 
                               help="Your ability to wait for optimal setups", key="psych_patience")
            
            adaptability = st.slider("Adaptability", 1, 10, 5, 
                                   help="Your ability to adjust to changing market conditions", key="psych_adapt")
        
        # ICT-specific psychological factors
        st.write("**ICT-Specific Psychology Factors**")
        
        ict_col1, ict_col2 = st.columns(2)
        
        with ict_col1:
            fomo_rating = st.slider("FOMO Level", 1, 10, 5, 
                                  help="Fear of missing out on trades (lower is better)", key="psych_fomo")
            
            patience_for_setups = st.slider("Patience for ICT Setups", 1, 10, 5, 
                                          help="Ability to wait for high-probability ICT setups", key="psych_patience_ict")
        
        with ict_col2:
            market_bias = st.selectbox("Market Bias", 
                                     options=["Neutral", "Bullish", "Bearish", "Confused"],
                                     index=0,
                                     help="Your overall bias toward the market", key="psych_bias")
            
            trading_fidelity = st.slider("Trading Plan Fidelity", 1, 10, 5, 
                                       help="How closely you followed your ICT trading plan", key="psych_fidelity")
        
        # Journal notes
        st.write("**Trading Psychology Notes**")
        
        mood = st.text_input("Current Mood", 
                           help="Describe your current emotional state", key="psych_mood")
        
        external_factors = st.text_area("External Factors", 
                                      help="Any external factors affecting your trading (e.g., life events, news)", 
                                      key="psych_external", height=80)
        
        trading_notes = st.text_area("Trading Psychology Notes", 
                                   help="Observations about your trading psychology today", 
                                   key="psych_notes", height=80)
        
        improvement_notes = st.text_area("Improvement Plan", 
                                       help="How you plan to improve your trading psychology", 
                                       key="psych_improve", height=80)
        
        # Submit button
        submit_button = st.form_submit_button(label="Save Psychology Entry")
        
        if submit_button:
            # Process form data
            datetime_combined = datetime.datetime.combine(date, time)
            
            # Create entry data
            entry_data = {
                'date': datetime_combined.strftime("%Y-%m-%d %H:%M:%S"),
                'mental_state': mental_state,
                'focus': focus,
                'discipline': discipline,
                'confidence': confidence,
                'patience': patience,
                'adaptability': adaptability,
                'fomo_rating': fomo_rating,
                'patience_for_setups': patience_for_setups,
                'market_bias': market_bias,
                'trading_fidelity': trading_fidelity,
                'mood': mood,
                'external_factors': external_factors,
                'trading_notes': trading_notes,
                'improvement_notes': improvement_notes,
                'linked_trade_idx': selected_trade_idx
            }
            
            # Add entry
            success = add_psychology_entry(entry_data)
            
            if success:
                st.success("Psychology entry saved successfully!")
                st.rerun()
            else:
                st.error("Failed to save psychology entry. Please try again.")

def show_journal_table():
    """Display the table of all trade entries"""
    # Get trades dataframe
    trades_df = get_trades_df()
    
    if trades_df.empty:
        st.info("No trades recorded yet. Add your first trade in the 'Add/Edit Trade' tab.")
        return
    
    # Filter options
    st.subheader("Filter Trades")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=(
                pd.to_datetime(trades_df['date']).min().date() if 'date' in trades_df.columns else datetime.date.today() - datetime.timedelta(days=30),
                pd.to_datetime(trades_df['date']).max().date() if 'date' in trades_df.columns else datetime.date.today()
            ),
            key='journal_date_range'
        )
    
    with col2:
        # Symbol filter
        all_symbols = trades_df['symbol'].unique().tolist() if 'symbol' in trades_df.columns else []
        selected_symbols = st.multiselect("Symbols", options=all_symbols, default=[], key='journal_symbols')
    
    with col3:
        # Status filter
        all_statuses = trades_df['status'].unique().tolist() if 'status' in trades_df.columns else []
        selected_statuses = st.multiselect("Status", options=all_statuses, default=[], key='journal_status')
    
    # Direction filter
    direction_col, outcome_col = st.columns(2)
    
    with direction_col:
        selected_directions = st.multiselect(
            "Direction", 
            options=["Long", "Short"],
            default=[],
            key='journal_direction'
        )
    
    with outcome_col:
        selected_outcomes = st.multiselect(
            "Outcome",
            options=["Win", "Loss"],
            default=[],
            key='journal_outcome'
        )
    
    # Apply filters
    filtered_df = trades_df.copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(pd.to_datetime(filtered_df['date']).dt.date >= start_date) & 
                                 (pd.to_datetime(filtered_df['date']).dt.date <= end_date)]
    
    if selected_symbols:
        filtered_df = filtered_df[filtered_df['symbol'].isin(selected_symbols)]
    
    if selected_directions:
        filtered_df = filtered_df[filtered_df['direction'].isin(selected_directions)]
    
    if selected_statuses:
        filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]
    
    if selected_outcomes:
        if 'Win' in selected_outcomes:
            win_condition = filtered_df['pnl'] > 0
            filtered_df = filtered_df[win_condition if 'Loss' not in selected_outcomes else win_condition | (filtered_df['pnl'] < 0)]
        elif 'Loss' in selected_outcomes:
            filtered_df = filtered_df[filtered_df['pnl'] < 0]
    
    # Sort by date (most recent first)
    if 'date' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('date', ascending=False)
    
    # Display filtered trades
    st.subheader(f"Trade Journal ({len(filtered_df)} entries)")
    
    # Display either the full table or a preview depending on count
    display_cols = ['date', 'symbol', 'direction', 'entry', 'exit', 'pnl', 'status']
    
    # Show table with all columns
    st.dataframe(filtered_df[display_cols], use_container_width=True)
    
    # Select and view a specific trade
    if not filtered_df.empty:
        st.subheader("Trade Details")
        
        # Have user select a trade to view
        trade_indices = filtered_df.index.tolist()
        trade_labels = [f"{idx}: {row['symbol']} {row['direction']} on {row['date']}" 
                      for idx, row in filtered_df.iterrows()]
        
        selected_trade_label = st.selectbox(
            "Select a trade to view details",
            options=trade_labels,
            key='view_trade_selector'
        )
        
        if selected_trade_label:
            # Extract the index from the label
            selected_index = int(selected_trade_label.split(':')[0])
            
            # Get the selected trade
            selected_trade = filtered_df.loc[selected_index]
            
            # Display trade details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Basic Info**")
                st.write(f"Symbol: {selected_trade['symbol']}")
                st.write(f"Direction: {selected_trade['direction']}")
                st.write(f"Date: {selected_trade['date']}")
                st.write(f"Status: {selected_trade['status']}")
                st.write(f"Test Type: {selected_trade.get('test_type', 'N/A')}")
                st.write(f"Order Type: {selected_trade.get('order_type', 'N/A')}")
            
            with col2:
                st.write("**Trade Metrics**")
                st.write(f"Entry Price: {selected_trade['entry']}")
                st.write(f"Exit Price: {selected_trade['exit']}")
                st.write(f"Position Size: {selected_trade.get('size', 'N/A')}")
                st.write(f"P&L: ${selected_trade.get('pnl', 0):.2f}")
                st.write(f"Risk: ${selected_trade.get('risk', 'N/A')}")
                st.write(f"RRR: {selected_trade.get('actual_rrr', 'N/A')}")
            
            with col3:
                st.write("**ICT Elements**")
                ict_elements = []
                
                if selected_trade.get('order_block_entry'):
                    ict_elements.append("Order Block Entry")
                if selected_trade.get('liquidity_grab'):
                    ict_elements.append("Liquidity Grab")
                if selected_trade.get('fvg_entry'):
                    ict_elements.append("Fair Value Gap")
                if selected_trade.get('breaker_block'):
                    ict_elements.append("Breaker Block")
                if selected_trade.get('inducement_identified'):
                    ict_elements.append("Inducement")
                if selected_trade.get('imbalance_entry'):
                    ict_elements.append("Imbalance")
                
                if ict_elements:
                    for element in ict_elements:
                        st.write(f"- {element}")
                else:
                    st.write("No ICT elements tagged")
                
                st.write(f"Market Condition: {selected_trade.get('market_condition', 'N/A')}")
            
            # Notes
            if selected_trade.get('notes'):
                st.write("**Notes**")
                st.write(selected_trade['notes'])
            
            # Images
            if selected_trade.get('images') and len(selected_trade['images']) > 0:
                st.write("**Trade Images**")
                
                for img in selected_trade['images']:
                    display_image(img)
            
            # Actions
            st.write("**Actions**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Edit This Trade", key="edit_selected_trade"):
                    # Set up edit state
                    st.session_state.edit_index = selected_index
                    st.session_state.current_trade = selected_trade.to_dict()
                    
                    # Switch to the Add/Edit tab
                    st.rerun()
            
            with col2:
                if st.button("Delete This Trade", key="delete_selected_trade"):
                    if delete_trade(selected_index):
                        st.success(f"Deleted trade for {selected_trade['symbol']}")
                        st.rerun()
                    else:
                        st.error("Failed to delete trade")
