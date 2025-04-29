import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.data_manager import get_trades_df, get_psychology_df, add_psychology_entry
from utils.metrics import calculate_psychology_metrics
from utils.visualization import create_psychology_correlation_chart

def show_psychology_page():
    """Display the psychology tracker page"""
    st.title("Trading Psychology Tracker")
    
    # Create tabs for psychology tracking
    add_tab, view_tab, analysis_tab = st.tabs([
        "Log Psychology Entry", "View Psychology Journal", "Psychology Analysis"
    ])
    
    with add_tab:
        show_psychology_form()
    
    with view_tab:
        show_psychology_journal()
    
    with analysis_tab:
        show_psychology_analysis()

def show_psychology_form():
    """Display the form for adding psychology entries"""
    st.subheader("Log Your Trading Psychology")
    
    with st.form(key="psychology_form"):
        # Date and time
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", value=datetime.now().date())
        
        with col2:
            time = st.time_input("Time", value=datetime.now().time())
        
        # Psychology ratings
        st.write("**Rate your trading psychology factors (1-10)**")
        
        rating_col1, rating_col2 = st.columns(2)
        
        with rating_col1:
            mental_state = st.slider("Overall Mental State", 1, 10, 5, 
                                    help="Your overall mental and emotional state")
            
            focus = st.slider("Focus", 1, 10, 5, 
                             help="Your ability to concentrate on trading tasks")
            
            discipline = st.slider("Discipline", 1, 10, 5, 
                                  help="Your ability to stick to your trading plan")
        
        with rating_col2:
            confidence = st.slider("Confidence", 1, 10, 5, 
                                  help="Your belief in your trading decisions")
            
            patience = st.slider("Patience", 1, 10, 5, 
                                help="Your ability to wait for optimal setups")
            
            adaptability = st.slider("Adaptability", 1, 10, 5, 
                                    help="Your ability to adjust to changing market conditions")
        
        # ICT-specific psychological factors
        st.write("**ICT-Specific Psychology Factors**")
        
        ict_col1, ict_col2 = st.columns(2)
        
        with ict_col1:
            fomo_rating = st.slider("FOMO Level", 1, 10, 5, 
                                   help="Fear of missing out on trades (lower is better)")
            
            patience_for_setups = st.slider("Patience for ICT Setups", 1, 10, 5, 
                                           help="Ability to wait for high-probability ICT setups")
        
        with ict_col2:
            market_bias = st.selectbox("Market Bias", 
                                      options=["Neutral", "Bullish", "Bearish", "Confused"],
                                      index=0,
                                      help="Your overall bias toward the market")
            
            trading_fidelity = st.slider("Trading Plan Fidelity", 1, 10, 5, 
                                        help="How closely you followed your ICT trading plan")
        
        # Journal notes
        st.write("**Trading Psychology Notes**")
        
        mood = st.text_input("Current Mood", 
                            help="Describe your current emotional state")
        
        external_factors = st.text_area("External Factors", 
                                       help="Any external factors affecting your trading (e.g., life events, news)")
        
        trading_notes = st.text_area("Trading Psychology Notes", 
                                    help="Observations about your trading psychology today")
        
        improvement_notes = st.text_area("Improvement Plan", 
                                        help="How you plan to improve your trading psychology")
        
        # Submit button
        submit_button = st.form_submit_button(label="Save Psychology Entry")
        
        if submit_button:
            # Process form data
            datetime_combined = datetime.combine(date, time)
            
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
                'improvement_notes': improvement_notes
            }
            
            # Add entry
            success = add_psychology_entry(entry_data)
            
            if success:
                st.success("Psychology entry saved successfully!")
                st.rerun()
            else:
                st.error("Failed to save psychology entry. Please try again.")

def show_psychology_journal():
    """Display the psychology journal entries"""
    st.subheader("Psychology Journal Entries")
    
    # Get psychology dataframe
    psychology_df = get_psychology_df()
    
    if psychology_df.empty:
        st.info("No psychology entries recorded yet. Add your first entry in the 'Log Psychology Entry' tab.")
        return
    
    # Sort by date (most recent first)
    psychology_df = psychology_df.sort_values('date', ascending=False)
    
    # Display entries
    for i, row in psychology_df.iterrows():
        with st.expander(f"Entry: {pd.to_datetime(row['date']).strftime('%Y-%m-%d %H:%M')} | Mental State: {row['mental_state']}/10"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Psychological Ratings**")
                st.write(f"Overall Mental State: {row['mental_state']}/10")
                st.write(f"Focus: {row['focus']}/10")
                st.write(f"Discipline: {row['discipline']}/10")
                st.write(f"Confidence: {row['confidence']}/10")
                st.write(f"Patience: {row['patience']}/10")
                st.write(f"Adaptability: {row['adaptability']}/10")
            
            with col2:
                st.write("**ICT-Specific Factors**")
                st.write(f"FOMO Level: {row['fomo_rating']}/10")
                st.write(f"Patience for ICT Setups: {row['patience_for_setups']}/10")
                st.write(f"Market Bias: {row['market_bias']}")
                st.write(f"Trading Plan Fidelity: {row['trading_fidelity']}/10")
            
            st.write("**Notes**")
            st.write(f"Mood: {row['mood']}")
            
            if row['external_factors']:
                st.write("**External Factors**")
                st.write(row['external_factors'])
            
            if row['trading_notes']:
                st.write("**Trading Psychology Notes**")
                st.write(row['trading_notes'])
            
            if row['improvement_notes']:
                st.write("**Improvement Plan**")
                st.write(row['improvement_notes'])

def show_psychology_analysis():
    """Display psychology analysis and correlations"""
    st.subheader("Trading Psychology Analysis")
    
    # Get psychology and trades dataframes
    psychology_df = get_psychology_df()
    trades_df = get_trades_df()
    
    if psychology_df.empty:
        st.info("No psychology entries recorded yet. Add entries in the 'Log Psychology Entry' tab to see analysis.")
        return
    
    # Calculate psychology metrics
    psychology_metrics = calculate_psychology_metrics(psychology_df, trades_df)
    
    # Display summary metrics
    metric_cols = st.columns(4)
    
    metric_cols[0].metric("Avg Mental State", f"{psychology_metrics['avg_mental_state']:.1f}/10")
    metric_cols[1].metric("Avg Discipline", f"{psychology_metrics['avg_discipline']:.1f}/10")
    metric_cols[2].metric("Avg Focus", f"{psychology_metrics['avg_focus']:.1f}/10")
    metric_cols[3].metric("Days Tracked", psychology_metrics['days_tracked'])
    
    # Correlation chart
    st.subheader("Psychology vs. Performance Correlation")
    
    if not trades_df.empty:
        corr_chart = create_psychology_correlation_chart(psychology_df, trades_df)
        if corr_chart:
            st.plotly_chart(corr_chart, use_container_width=True)
            
            # Interpretation of correlation
            correlation = psychology_metrics['corr_mental_state_pnl']
            
            if abs(correlation) > 0.7:
                if correlation > 0:
                    st.success(f"Strong positive correlation ({correlation:.2f}) between mental state and trading performance. Your psychological state is strongly linked to your results.")
                else:
                    st.error(f"Strong negative correlation ({correlation:.2f}) between mental state and trading performance. This unusual pattern suggests you may perform better under pressure or might be rating your mental state incorrectly.")
            elif abs(correlation) > 0.3:
                if correlation > 0:
                    st.info(f"Moderate positive correlation ({correlation:.2f}) between mental state and trading performance. Your psychological state appears to influence your trading.")
                else:
                    st.warning(f"Moderate negative correlation ({correlation:.2f}) between mental state and trading performance. This could indicate counterintuitive patterns in your trading psychology.")
            else:
                st.write(f"Weak correlation ({correlation:.2f}) between mental state and trading performance. Your trading results may be more influenced by other factors than your reported mental state.")
        else:
            st.info("Not enough data to analyze correlations. Continue logging both trades and psychology entries.")
    else:
        st.info("No trade data available to correlate with psychology. Add trades in the Journal Entries page.")
    
    # Psychological factors tracking over time
    st.subheader("Psychological Factors Over Time")
    
    # Prepare data
    psychology_df['date'] = pd.to_datetime(psychology_df['date'])
    psychology_df = psychology_df.sort_values('date')
    
    # Create time series chart
    psych_factors = ['mental_state', 'focus', 'discipline', 'confidence', 'patience']
    
    # Check which factors exist in the dataframe
    available_factors = [f for f in psych_factors if f in psychology_df.columns]
    
    if available_factors:
        fig = go.Figure()
        
        for factor in available_factors:
            fig.add_trace(go.Scatter(
                x=psychology_df['date'],
                y=psychology_df[factor],
                mode='lines+markers',
                name=factor.replace('_', ' ').title()
            ))
        
        fig.update_layout(
            title="Psychological Factors Trend",
            xaxis_title="Date",
            yaxis_title="Rating (1-10)",
            yaxis=dict(range=[0, 11]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No psychological factor data available.")
    
    # ICT-specific psychology analysis
    st.subheader("ICT-Specific Psychology Analysis")
    
    # Check if we have ICT-specific psychology data
    ict_factors = ['fomo_rating', 'patience_for_setups', 'trading_fidelity']
    available_ict_factors = [f for f in ict_factors if f in psychology_df.columns]
    
    if available_ict_factors:
        # Create ICT psychology radar chart
        recent_entries = psychology_df.tail(min(5, len(psychology_df)))
        
        fig = go.Figure()
        
        for i, row in recent_entries.iterrows():
            entry_date = pd.to_datetime(row['date']).strftime("%Y-%m-%d")
            
            values = []
            for factor in available_ict_factors:
                values.append(row[factor])
            
            # Add an extra value to close the loop
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=[f.replace('_', ' ').title() for f in available_ict_factors] + [available_ict_factors[0].replace('_', ' ').title()],
                fill='toself',
                name=entry_date
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True,
            title="ICT Psychology Profile (Recent Entries)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Compare ICT psychology with trade performance
        if not trades_df.empty and 'date' in trades_df.columns:
            st.subheader("ICT Psychology vs. Trading Performance")
            
            # Prepare trades data
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            
            # For each psychology entry, find trades within ±1 day
            psychology_vs_performance = []
            
            for _, psych_row in psychology_df.iterrows():
                psych_date = pd.to_datetime(psych_row['date'])
                start_date = psych_date - timedelta(days=1)
                end_date = psych_date + timedelta(days=1)
                
                # Find trades within the date range
                nearby_trades = trades_df[(trades_df['date'] >= start_date) & 
                                         (trades_df['date'] <= end_date) & 
                                         (trades_df['status'] == 'Completed')]
                
                if not nearby_trades.empty:
                    win_rate = (nearby_trades['pnl'] > 0).mean() * 100
                    total_pnl = nearby_trades['pnl'].sum()
                    trade_count = len(nearby_trades)
                    
                    # Extract relevant psychology metrics
                    psychology_vs_performance.append({
                        'date': psych_date,
                        'trading_fidelity': psych_row.get('trading_fidelity', None),
                        'patience_for_setups': psych_row.get('patience_for_setups', None),
                        'fomo_rating': psych_row.get('fomo_rating', None),
                        'win_rate': win_rate,
                        'pnl': total_pnl,
                        'trade_count': trade_count
                    })
            
            if psychology_vs_performance:
                perf_df = pd.DataFrame(psychology_vs_performance)
                
                # Create scatter plot for ICT psychology vs. performance
                if 'trading_fidelity' in perf_df.columns and not perf_df['trading_fidelity'].isnull().all():
                    fig = px.scatter(
                        perf_df,
                        x='trading_fidelity',
                        y='win_rate',
                        size='trade_count',
                        color='pnl',
                        hover_name='date',
                        color_continuous_scale=['red', 'yellow', 'green'],
                        title='Trading Plan Fidelity vs. Win Rate',
                        labels={
                            'trading_fidelity': 'Trading Plan Fidelity (1-10)',
                            'win_rate': 'Win Rate (%)',
                            'pnl': 'P&L',
                            'trade_count': 'Number of Trades'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show findings
                    fidelity_corr = perf_df['trading_fidelity'].corr(perf_df['win_rate'])
                    
                    if fidelity_corr > 0.5:
                        st.success(f"Strong positive correlation ({fidelity_corr:.2f}) between trading plan fidelity and win rate. Following your ICT trading plan is clearly beneficial.")
                    elif fidelity_corr > 0.2:
                        st.info(f"Moderate positive correlation ({fidelity_corr:.2f}) between trading plan fidelity and win rate. Following your ICT trading plan appears helpful.")
                    elif fidelity_corr > -0.2:
                        st.write(f"Weak correlation ({fidelity_corr:.2f}) between trading plan fidelity and win rate. The relationship is not clear yet.")
                    else:
                        st.warning(f"Negative correlation ({fidelity_corr:.2f}) between trading plan fidelity and win rate. Consider reviewing your trading plan or how you rate fidelity.")
                else:
                    st.info("Not enough trading fidelity data available.")
            else:
                st.info("No matching time periods between psychology entries and trades.")
        else:
            st.info("No trade data available for comparison.")
    else:
        st.info("No ICT-specific psychology data available. Use the 'Log Psychology Entry' tab to record ICT-specific factors.")
    
    # Trading psychology improvement recommendations
    if not psychology_df.empty:
        st.subheader("Trading Psychology Improvement Recommendations")
        
        # Calculate average scores for recent entries (last 5)
        recent_df = psychology_df.sort_values('date', ascending=False).head(min(5, len(psychology_df)))
        
        recent_scores = {}
        for factor in ['mental_state', 'focus', 'discipline', 'confidence', 'patience', 
                      'adaptability', 'fomo_rating', 'patience_for_setups', 'trading_fidelity']:
            if factor in recent_df.columns:
                recent_scores[factor] = recent_df[factor].mean()
        
        # Generate recommendations based on scores
        recommendations = []
        
        if 'focus' in recent_scores and recent_scores['focus'] < 6:
            recommendations.append({
                'factor': 'Focus',
                'score': recent_scores['focus'],
                'recommendation': "Your focus scores are below average. Consider implementing meditation or mindfulness practices before trading sessions. Eliminate distractions and create a dedicated trading environment."
            })
        
        if 'discipline' in recent_scores and recent_scores['discipline'] < 6:
            recommendations.append({
                'factor': 'Discipline',
                'score': recent_scores['discipline'],
                'recommendation': "Your discipline scores need improvement. Create a structured trading checklist with ICT concepts for each trade. Set clear rules for entries and exits based on order blocks and liquidity levels."
            })
        
        if 'fomo_rating' in recent_scores and recent_scores['fomo_rating'] > 5:
            recommendations.append({
                'factor': 'FOMO',
                'score': recent_scores['fomo_rating'],
                'recommendation': "Your FOMO levels are high. Practice patience by waiting for confirmed ICT setups only. Focus on quality over quantity and track the trades you didn't take that would have failed."
            })
        
        if 'patience_for_setups' in recent_scores and recent_scores['patience_for_setups'] < 6:
            recommendations.append({
                'factor': 'Patience for Setups',
                'score': recent_scores['patience_for_setups'],
                'recommendation': "You need more patience for proper ICT setups. Create a visual checklist of ideal setups with order blocks, fair value gaps, and liquidity sweeps. Only take trades that match your predefined criteria."
            })
        
        if 'trading_fidelity' in recent_scores and recent_scores['trading_fidelity'] < 6:
            recommendations.append({
                'factor': 'Trading Plan Fidelity',
                'score': recent_scores['trading_fidelity'],
                'recommendation': "You're not following your trading plan consistently. Review your plan to ensure it's clear and actionable. Break down ICT concepts into specific rules and verify each criterion before entering trades."
            })
        
        if recommendations:
            for rec in recommendations:
                st.warning(f"**{rec['factor']} (Score: {rec['score']:.1f}/10)**")
                st.write(rec['recommendation'])
        else:
            st.success("Your recent psychology scores look good! Continue monitoring and maintaining your trading discipline and focus.")
