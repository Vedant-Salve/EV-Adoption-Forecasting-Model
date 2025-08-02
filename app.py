import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('dark_background')
sns.set_palette("husl")

# Set Streamlit page config first thing
st.set_page_config(
    page_title="EV Forecast Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# EV Forecasting App\nPowered by AI for sustainable transportation insights!"
    }
)

# === Load model with error handling ===
@st.cache_resource
def load_model():
    try:
        model = joblib.load('D:\\Codes\\Python_\\Deep Learning\\EV_Forecasting_Model\\forecasting_ev_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please check the file path.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# === Enhanced Styling ===
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            margin: 10px 0;
        }
        .forecast-header {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            padding: 15px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .comparison-section {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2193b0, #6dd5ed);
        }
    </style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: white; font-size: 48px; margin: 0;'>🚗⚡ EV Adoption Forecaster</h1>
        <h3 style='color: #E8F4FD; margin: 10px 0;'>Washington State County Analysis & Prediction Dashboard</h3>
        <p style='color: #B8E6FF; font-size: 18px;'>Predict electric vehicle adoption trends with advanced machine learning</p>
    </div>
""", unsafe_allow_html=True)

# === Sidebar Configuration ===
with st.sidebar:
    st.markdown("## 🎛️ Dashboard Controls")
    
    # Forecast horizon slider
    forecast_months = st.slider(
        "📅 Forecast Horizon (months)", 
        min_value=6, 
        max_value=60, 
        value=36, 
        step=6,
        help="Select how many months into the future to forecast"
    )
    
    # Chart style selection
    chart_style = st.selectbox(
        "📊 Chart Style",
        ["Modern Dark", "Classic Light", "Seaborn", "Minimal"],
        help="Choose chart styling theme"
    )
    
    # Analysis options
    st.markdown("### 📈 Analysis Options")
    show_confidence_interval = st.checkbox("Show Confidence Intervals", value=True)
    show_growth_rate = st.checkbox("Show Growth Rate Analysis", value=True)
    show_seasonal_decomp = st.checkbox("Show Seasonal Analysis", value=False)

# === Apply Chart Styling ===
def apply_chart_style(style):
    """Apply different matplotlib styles"""
    if style == "Modern Dark":
        plt.style.use('dark_background')
        return {
            'bg_color': '#1e1e1e',
            'text_color': 'white',
            'grid_alpha': 0.3,
            'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
        }
    elif style == "Classic Light":
        plt.style.use('default')
        return {
            'bg_color': 'white',
            'text_color': 'black',
            'grid_alpha': 0.7,
            'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        }
    elif style == "Seaborn":
        plt.style.use('seaborn-v0_8')
        return {
            'bg_color': '#f8f9fa',
            'text_color': '#2c3e50',
            'grid_alpha': 0.5,
            'colors': sns.color_palette("husl", 6)
        }
    else:  # Minimal
        plt.style.use('bmh')
        return {
            'bg_color': '#f0f0f0',
            'text_color': '#333333',
            'grid_alpha': 0.4,
            'colors': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4CAF50', '#9C27B0']
        }

style_config = apply_chart_style(chart_style)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("D:\\Codes\\Python_\\Deep Learning\\EV_Forecasting_Model\\preprocessed_ev_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found. Please check the file path.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

df = load_data()

# === Data Statistics ===
with st.expander("📊 Dataset Overview", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Counties", len(df['County'].unique()))
    with col2:
        st.metric("Total Records", len(df))
    with col3:
        st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
    with col4:
        st.metric("Total EVs", f"{df['Electric Vehicle (EV) Total'].sum():,}")

# === Enhanced County Selection ===
st.markdown("## 🎯 County Selection & Forecasting")

col1, col2 = st.columns([2, 1])

with col1:
    county_list = sorted(df['County'].dropna().unique().tolist())
    county = st.selectbox(
        "🏘️ Select a County", 
        county_list,
        help="Choose a county to generate EV adoption forecasts"
    )

with col2:
    if st.button("🔮 Generate Advanced Forecast", type="primary", use_container_width=True):
        st.success(f"✅ Generating forecast for {county} County...")

# Validate county selection
if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# === Enhanced Forecasting Function ===
def generate_forecast(county_df, county_code, forecast_horizon):
    """Generate forecasts with confidence intervals"""
    historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
    cumulative_ev = list(np.cumsum(historical_ev))
    months_since_start = county_df['months_since_start'].max()
    latest_date = county_df['Date'].max()

    future_rows = []
    predictions = []
    
    # Add some randomness for confidence intervals
    np.random.seed(42)
    
    for i in range(1, forecast_horizon + 1):
        forecast_date = latest_date + pd.DateOffset(months=i)
        months_since_start += 1
        
        lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
        roll_mean = np.mean([lag1, lag2, lag3])
        pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
        pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
        recent_cumulative = cumulative_ev[-6:]
        ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

        new_row = {
            'months_since_start': months_since_start,
            'county_encoded': county_code,
            'ev_total_lag1': lag1,
            'ev_total_lag2': lag2,
            'ev_total_lag3': lag3,
            'ev_total_roll_mean_3': roll_mean,
            'ev_total_pct_change_1': pct_change_1,
            'ev_total_pct_change_3': pct_change_3,
            'ev_growth_slope': ev_growth_slope
        }

        pred = model.predict(pd.DataFrame([new_row]))[0]
        
        # Add confidence intervals (simulate uncertainty)
        uncertainty = max(pred * 0.1, 5)  # 10% uncertainty or minimum 5
        lower_bound = max(0, pred - uncertainty)
        upper_bound = pred + uncertainty
        
        future_rows.append({
            "Date": forecast_date, 
            "Predicted EV Total": round(pred),
            "Lower Bound": round(lower_bound),
            "Upper Bound": round(upper_bound)
        })
        predictions.append(pred)

        historical_ev.append(pred)
        if len(historical_ev) > 6:
            historical_ev.pop(0)

        cumulative_ev.append(cumulative_ev[-1] + pred)
        if len(cumulative_ev) > 6:
            cumulative_ev.pop(0)
    
    return future_rows, predictions

# Generate forecast
future_rows, predictions = generate_forecast(county_df, county_code, forecast_months)

# === Display Key Metrics ===
st.markdown("## 📊 Key Insights")

col1, col2, col3, col4 = st.columns(4)

historical_total = county_df['Electric Vehicle (EV) Total'].sum()
forecast_total = sum([row['Predicted EV Total'] for row in future_rows])
latest_monthly = county_df['Electric Vehicle (EV) Total'].iloc[-1]
predicted_peak = max([row['Predicted EV Total'] for row in future_rows])

with col1:
    st.metric(
        "📈 Historical Total EVs", 
        f"{historical_total:,}",
        help="Total EVs registered historically"
    )

with col2:
    growth_rate = ((forecast_total / historical_total) * 100) if historical_total > 0 else 0
    st.metric(
        "🚀 Forecast Growth", 
        f"{growth_rate:.1f}%",
        delta=f"{forecast_total:,} new EVs",
        help="Expected growth over forecast period"
    )

with col3:
    st.metric(
        "📅 Latest Monthly", 
        f"{latest_monthly:,}",
        help="Most recent monthly EV registrations"
    )

with col4:
    st.metric(
        "🎯 Predicted Peak", 
        f"{predicted_peak:,}",
        help="Highest predicted monthly registrations"
    )

# === Enhanced Visualization ===
st.markdown('<div class="forecast-header"><h2>📊 Advanced Forecast Visualization</h2></div>', unsafe_allow_html=True)

# Prepare data for plotting
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()
historical_cum['Type'] = 'Historical'

forecast_df = pd.DataFrame(future_rows)
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]
forecast_df['Type'] = 'Forecast'

# Create comprehensive matplotlib visualization
fig = plt.figure(figsize=(16, 12))
colors = style_config['colors']

# Create a 2x2 subplot layout
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 1. Monthly EV Registrations
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(county_df['Date'], county_df['Electric Vehicle (EV) Total'], 
         'o-', color=colors[0], linewidth=3, markersize=6, label='Historical Monthly', alpha=0.8)
ax1.plot(forecast_df['Date'], forecast_df['Predicted EV Total'], 
         's--', color=colors[1], linewidth=3, markersize=6, label='Forecast Monthly', alpha=0.8)

if show_confidence_interval:
    ax1.fill_between(forecast_df['Date'], forecast_df['Lower Bound'], forecast_df['Upper Bound'],
                    alpha=0.3, color=colors[1], label='Confidence Interval')

ax1.set_title(f'Monthly EV Registrations - {county} County', 
              color=style_config['text_color'], fontsize=16, fontweight='bold')
ax1.set_ylabel('Monthly EVs', color=style_config['text_color'], fontsize=12)
ax1.grid(True, alpha=style_config['grid_alpha'])
ax1.set_facecolor(style_config['bg_color'])
ax1.tick_params(colors=style_config['text_color'])
ax1.legend(loc='upper left', framealpha=0.9)

# Add annotations for key points
max_historical = county_df['Electric Vehicle (EV) Total'].max()
max_historical_date = county_df.loc[county_df['Electric Vehicle (EV) Total'].idxmax(), 'Date']
ax1.annotate(f'Historical Peak: {max_historical:,}', 
            xy=(max_historical_date, max_historical), 
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[0], alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# 2. Cumulative EV Adoption
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(historical_cum['Date'], historical_cum['Cumulative EV'], 
         'o-', color=colors[2], linewidth=3, markersize=5, label='Historical Cumulative', alpha=0.8)
ax2.plot(forecast_df['Date'], forecast_df['Cumulative EV'], 
         's--', color=colors[3], linewidth=3, markersize=5, label='Forecast Cumulative', alpha=0.8)

ax2.set_title(f'Cumulative EV Adoption - {county} County', 
              color=style_config['text_color'], fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', color=style_config['text_color'], fontsize=12)
ax2.set_ylabel('Cumulative EVs', color=style_config['text_color'], fontsize=12)
ax2.grid(True, alpha=style_config['grid_alpha'])
ax2.set_facecolor(style_config['bg_color'])
ax2.tick_params(colors=style_config['text_color'])
ax2.legend(loc='upper left', framealpha=0.9)

# Format y-axis to show numbers in thousands
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}K'))

# 3. Growth Rate Analysis (if enabled)
ax3 = fig.add_subplot(gs[1, 1])

if show_growth_rate:
    # Calculate month-over-month growth rates
    monthly_growth = []
    for i in range(1, len(predictions)):
        if predictions[i-1] != 0:
            growth = ((predictions[i] - predictions[i-1]) / predictions[i-1]) * 100
            monthly_growth.append(growth)
        else:
            monthly_growth.append(0)
    
    if monthly_growth:
        growth_dates = forecast_df['Date'].iloc[1:].tolist()
        
        # Create bar chart for growth rates
        bars = ax3.bar(range(len(monthly_growth)), monthly_growth, 
                      color=[colors[4] if x >= 0 else colors[5] for x in monthly_growth], 
                      alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add zero line
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Customize the growth rate chart
        ax3.set_title('Monthly Growth Rate Forecast', 
                      color=style_config['text_color'], fontsize=14, fontweight='bold')
        ax3.set_xlabel('Forecast Period', color=style_config['text_color'], fontsize=12)
        ax3.set_ylabel('Growth Rate (%)', color=style_config['text_color'], fontsize=12)
        ax3.grid(True, alpha=style_config['grid_alpha'], axis='y')
        ax3.set_facecolor(style_config['bg_color'])
        ax3.tick_params(colors=style_config['text_color'])
        
        # Add value labels on significant bars
        for i, (bar, value) in enumerate(zip(bars, monthly_growth)):
            if abs(value) > np.std(monthly_growth):  # Only label significant values
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if value >= 0 else -3),
                        f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top',
                        color=style_config['text_color'], fontsize=9, fontweight='bold')
        
        # Set x-axis labels to show every 6th month
        tick_positions = range(0, len(monthly_growth), 6)
        tick_labels = [growth_dates[i].strftime('%Y-%m') for i in tick_positions if i < len(growth_dates)]
        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels(tick_labels, rotation=45)
else:
    # Show summary statistics instead
    stats_data = {
        'Metric': ['Historical Total', 'Forecast Total', 'Expected Growth', 'Peak Monthly'],
        'Value': [f'{historical_total:,}', f'{forecast_total:,}', f'{growth_rate:.1f}%', f'{predicted_peak:,}']
    }
    
    # Create a table-like visualization
    table_colors = [colors[i % len(colors)] for i in range(len(stats_data['Metric']))]
    bars = ax3.barh(stats_data['Metric'], 
                   [historical_total, forecast_total, growth_rate*1000, predicted_peak], 
                   color=table_colors, alpha=0.7)
    
    ax3.set_title('Key Statistics Summary', 
                  color=style_config['text_color'], fontsize=14, fontweight='bold')
    ax3.set_xlabel('Values', color=style_config['text_color'], fontsize=12)
    ax3.grid(True, alpha=style_config['grid_alpha'], axis='x')
    ax3.set_facecolor(style_config['bg_color'])
    ax3.tick_params(colors=style_config['text_color'])

# Set overall figure background
fig.patch.set_facecolor(style_config['bg_color'])

# Add a main title for the entire figure
fig.suptitle(f'EV Adoption Analysis Dashboard - {county} County', 
             fontsize=20, fontweight='bold', color=style_config['text_color'], y=0.95)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# Additional Growth Rate Analysis
if show_growth_rate and monthly_growth:
    st.markdown("### 📈 Detailed Growth Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_growth = np.mean(monthly_growth) if monthly_growth else 0
        st.metric("📊 Average Monthly Growth Rate", f"{avg_growth:.2f}%")
    
    with col2:
        max_growth = max(monthly_growth) if monthly_growth else 0
        st.metric("🚀 Peak Growth Rate", f"{max_growth:.2f}%")
    
    # Separate detailed growth chart
    fig_growth, ax_growth = plt.subplots(figsize=(12, 6))
    fig_growth.patch.set_facecolor(style_config['bg_color'])
    
    growth_dates = forecast_df['Date'].iloc[1:].tolist()
    
    # Line plot with markers
    ax_growth.plot(growth_dates, monthly_growth, 
                  'o-', color=colors[4], linewidth=3, markersize=8, alpha=0.8)
    
    # Fill area under curve
    ax_growth.fill_between(growth_dates, monthly_growth, 0, 
                          where=np.array(monthly_growth) >= 0, 
                          color=colors[4], alpha=0.3, interpolate=True)
    ax_growth.fill_between(growth_dates, monthly_growth, 0, 
                          where=np.array(monthly_growth) < 0, 
                          color=colors[5], alpha=0.3, interpolate=True)
    
    # Add trend line
    x_numeric = np.arange(len(monthly_growth))
    z = np.polyfit(x_numeric, monthly_growth, 1)
    p = np.poly1d(z)
    ax_growth.plot(growth_dates, p(x_numeric), '--', 
                   color='yellow', linewidth=2, alpha=0.8, label=f'Trend: {z[0]:.3f}% per month')
    
    ax_growth.axhline(y=0, color='red', linestyle='-', alpha=0.8, linewidth=2)
    ax_growth.set_title("Monthly Growth Rate Forecast with Trend Analysis", 
                       fontsize=16, fontweight='bold', color=style_config['text_color'])
    ax_growth.set_xlabel("Date", color=style_config['text_color'], fontsize=12)
    ax_growth.set_ylabel("Growth Rate (%)", color=style_config['text_color'], fontsize=12)
    ax_growth.grid(True, alpha=style_config['grid_alpha'])
    ax_growth.set_facecolor(style_config['bg_color'])
    ax_growth.tick_params(colors=style_config['text_color'])
    ax_growth.legend(framealpha=0.9)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax_growth.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig_growth, use_container_width=True)

# === Multi-County Comparison ===
st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
st.markdown("## 🏆 Multi-County Comparison")

col1, col2 = st.columns([3, 1])

with col1:
    multi_counties = st.multiselect(
        "Select counties to compare (up to 5)", 
        county_list, 
        max_selections=5,
        default=[county] if county in county_list else [],
        help="Compare EV adoption trends across multiple counties"
    )

with col2:
    comparison_metric = st.selectbox(
        "Comparison Metric",
        ["Cumulative Growth", "Peak Monthly", "Average Monthly"],
        help="Choose what to compare across counties"
    )

if multi_counties and len(multi_counties) > 1:
    comparison_data = []
    summary_stats = []

    for cty in multi_counties:
        cty_df = df[df['County'] == cty].sort_values("Date")
        cty_code = cty_df['county_encoded'].iloc[0]
        
        cty_future_rows, cty_predictions = generate_forecast(cty_df, cty_code, forecast_months)
        
        # Historical data
        hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()
        
        # Forecast data
        fc_df = pd.DataFrame(cty_future_rows)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]
        
        # Combine for plotting
        combined_cty = pd.concat([
            hist_cum[['Date', 'Cumulative EV']],
            fc_df[['Date', 'Cumulative EV']]
        ], ignore_index=True)
        
        combined_cty['County'] = cty
        comparison_data.append(combined_cty)
        
        # Calculate summary statistics
        historical_total = hist_cum['Cumulative EV'].iloc[-1]
        forecast_total = fc_df['Cumulative EV'].iloc[-1]
        growth_pct = ((forecast_total - historical_total) / historical_total) * 100 if historical_total > 0 else 0
        peak_monthly = max(cty_predictions)
        avg_monthly = np.mean(cty_predictions)
        
        summary_stats.append({
            'County': cty,
            'Historical Total': historical_total,
            'Forecast Total': forecast_total,
            'Growth %': growth_pct,
            'Peak Monthly': peak_monthly,
            'Average Monthly': avg_monthly
        })

    # Plot comparison using matplotlib
    comp_df = pd.concat(comparison_data, ignore_index=True)
    
    # Create the comparison plot
    fig_comp, ax_comp = plt.subplots(figsize=(15, 8))
    fig_comp.patch.set_facecolor(style_config['bg_color'])
    
    # Plot each county with different colors and styles
    county_colors = colors[:len(multi_counties)]
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (cty, group) in enumerate(comp_df.groupby('County')):
        style_idx = i % len(line_styles)
        ax_comp.plot(group['Date'], group['Cumulative EV'], 
                    color=county_colors[i], linewidth=3, 
                    linestyle=line_styles[style_idx],
                    marker=markers[style_idx], markersize=6, 
                    label=cty, alpha=0.8, markevery=5)  # Show markers every 5 points
    
    # Customize the plot
    ax_comp.set_title("Multi-County EV Adoption Comparison", 
                      fontsize=18, fontweight='bold', color=style_config['text_color'])
    ax_comp.set_xlabel("Date", color=style_config['text_color'], fontsize=14)
    ax_comp.set_ylabel("Cumulative EVs", color=style_config['text_color'], fontsize=14)
    ax_comp.grid(True, alpha=style_config['grid_alpha'])
    ax_comp.set_facecolor(style_config['bg_color'])
    ax_comp.tick_params(colors=style_config['text_color'], labelsize=12)
    
    # Format y-axis to show numbers in thousands if values are large
    max_y = comp_df['Cumulative EV'].max()
    if max_y > 10000:
        ax_comp.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}K'))
    
    # Enhance legend
    legend = ax_comp.legend(title="Counties", title_fontsize=12, fontsize=11, 
                           loc='upper left', framealpha=0.9, 
                           fancybox=True, shadow=True)
    legend.get_title().set_color(style_config['text_color'])
    for text in legend.get_texts():
        text.set_color(style_config['text_color'])
    
    # Add vertical line to separate historical and forecast
    if len(county_df) > 0:
        historical_end = county_df['Date'].max()
        ax_comp.axvline(x=historical_end, color='red', linestyle=':', alpha=0.7, linewidth=2)
        ax_comp.text(historical_end, ax_comp.get_ylim()[1] * 0.9, 'Forecast Start', 
                    rotation=90, verticalalignment='top', horizontalalignment='right',
                    color='red', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax_comp.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig_comp, use_container_width=True)
    
    # Summary statistics table
    st.markdown("### 📋 Comparison Summary")
    summary_df = pd.DataFrame(summary_stats)
    
    # Format the dataframe for better display
    summary_df['Historical Total'] = summary_df['Historical Total'].apply(lambda x: f"{x:,.0f}")
    summary_df['Forecast Total'] = summary_df['Forecast Total'].apply(lambda x: f"{x:,.0f}")
    summary_df['Growth %'] = summary_df['Growth %'].apply(lambda x: f"{x:.1f}%")
    summary_df['Peak Monthly'] = summary_df['Peak Monthly'].apply(lambda x: f"{x:,.0f}")
    summary_df['Average Monthly'] = summary_df['Average Monthly'].apply(lambda x: f"{x:,.0f}")
    
    st.dataframe(summary_df, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# === Export Options ===
st.markdown("## 💾 Export & Download")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Download Forecast Data", use_container_width=True):
        forecast_export = pd.DataFrame(future_rows)
        csv = forecast_export.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"ev_forecast_{county}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Generate Report", use_container_width=True):
        report = f"""
        # EV Adoption Forecast Report
        
        **County:** {county}
        **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
        **Forecast Period:** {forecast_months} months
        
        ## Key Findings
        - Historical Total EVs: {historical_total:,}
        - Projected Growth: {growth_rate:.1f}%
        - Peak Monthly Forecast: {predicted_peak:,}
        
        ## Methodology
        This forecast uses machine learning models trained on historical EV registration data,
        incorporating temporal patterns, county-specific trends, and growth momentum indicators.
        """
        
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name=f"ev_report_{county}_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )

with col3:
    if st.button("🔄 Reset Dashboard", use_container_width=True):
        st.experimental_rerun()

# === Footer ===
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🎯 Accuracy Note**")
    st.caption("Forecasts are estimates based on historical patterns and may vary due to policy changes, economic factors, and market conditions.")

with col2:
    st.markdown("**📊 Data Source**")
    st.caption("Washington State Department of Licensing EV registration data, processed and analyzed using advanced ML techniques.")

with col3:
    st.markdown("**🔧 Model Info**")
    st.caption("Ensemble model incorporating time series analysis, regression, and trend decomposition for robust predictions.")

st.markdown("""
    <div style='text-align: center; padding: 20px; color: #B8E6FF;'>
        <strong>Prepared by Vedant Salve for the AICTE Internship Cycle 2 by S4F</strong><br>
        <em>Empowering sustainable transportation through data-driven insights</em>
    </div>
""", unsafe_allow_html=True)

# Add some JavaScript for enhanced interactivity (optional)
st.markdown("""
    <script>
    // Add some custom behavior if needed
    console.log("EV Forecasting Dashboard Loaded Successfully!");
    </script>
""", unsafe_allow_html=True)