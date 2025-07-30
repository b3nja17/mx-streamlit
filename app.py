# streamlit_earnings_analyzer.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import io
from datetime import timedelta

# Set Plotly color template
custom_colors = [
    "#1f77b4", "#419ede", "#708090", "#9467bd", "#7f7f7f",
    "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#2ca02c",
    "#6a5acd", "#708090"
]
pio.templates["custom"] = pio.templates["plotly_white"].update(
    layout={"colorway": custom_colors}
)
pio.templates.default = "custom"

# ---- Normalization Function ----
def normalize_df(df, ticker, earnings_date):
    df = df.copy()
    
    is_alt_format = 'Root' in df.columns

    if is_alt_format:
        rename_map = {
            'Time': 'Time', 'Root': 'Ticker', 'Expiry': 'Expiration Date',
            'Type': 'Option Type', 'Strike': 'Strike Price', 'Qty': 'Qty',
            'Price': 'Price', 'Notional': 'Notional', 'Bid': 'Bid',
            'Ask': 'Ask', 'Side': 'Lean', 'Volatility': 'IV',
            'Delta': 'Delta', 'Hedge Price': 'Underlying', 'Open Interest': 'Open Interest'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df = df[[col for col in rename_map.values() if col in df.columns]]
        df['Spread'] = 'Single'
    else:
        df['Spread'] = df['Spread'].replace('Spread', 'Multi').fillna('Single').replace('', 'Single')
        df = df.dropna()
        df = df[~df.apply(lambda row: row.astype(str).str.startswith("Loading")).any(axis=1)]

        option_parts = df['Option'].str.split(' ', expand=True)
        df['Expiration Date'] = pd.to_datetime(option_parts[0] + ' ' + option_parts[1] + ' ' + option_parts[2], format='%d %b %y')
        df['Strike Price'] = option_parts[3].astype(float)
        df['Option Type'] = option_parts[4]

        bid_ask = df['Market'].str.replace(r'[^\d.x]', '', regex=True).str.split('x', expand=True)
        df['Bid'] = bid_ask[0].astype(float)
        df['Ask'] = bid_ask[1].astype(float)

    df['Earnings Date'] = earnings_date
    df['Ticker'] = ticker
    df['IV'] = pd.to_numeric(df['IV'], errors='coerce')
    df['Notional'] = df['Qty'] * df['Price'] * 100
    df['Delta Notional'] = df['Delta'] * df['Qty']

    if 'Bid' in df.columns and 'Ask' in df.columns:
        df['Midpoint'] = (df['Bid'] + df['Ask']) / 2
        df['Lean'] = df.apply(lambda row: 'Ask' if row['Price'] >= row['Midpoint'] else 'Bid', axis=1)
        df['MidLean'] = df.apply(lambda row: 'Ask' if row['Price'] > row['Midpoint'] else 'Bid' , axis=1)


    def get_sentiment(row):
        if (row['Option Type'] == 'C' and row['Lean'] == 'Ask') or (row['Option Type'] == 'P' and row['Lean'] == 'Bid'):
            return 'Bullish'
        elif (row['Option Type'] == 'P' and row['Lean'] == 'Ask') or (row['Option Type'] == 'C' and row['Lean'] == 'Bid'):
            return 'Bearish'
        else:
            return None

    df['Sentiment'] = df.apply(get_sentiment, axis=1)

    def categorize_expiration(exp):
        if earnings_date:
            if exp < earnings_date + timedelta(days=7): return 'Week'
            elif exp < earnings_date + timedelta(days=31): return 'Month'
            else: return 'LEAP'
        return 'Unknown'

    df['Expiration Bucket'] = df['Expiration Date'].apply(categorize_expiration)

    def compute_moneyness(row):
        if row['Option Type'] == 'C': return row['Strike Price'] / row['Underlying'] - 1
        elif row['Option Type'] == 'P': return 1 - row['Strike Price'] / row['Underlying']

    df['Moneyness'] = df.apply(compute_moneyness, axis=1)

    def classify_position(row):
        if row['Option Type'] == 'C': return 'Call OTM' if row['Strike Price'] > row['Underlying'] else 'Call ITM'
        elif row['Option Type'] == 'P': return 'Put ITM' if row['Strike Price'] > row['Underlying'] else 'Put OTM'

    df['Position Type'] = df.apply(classify_position, axis=1)

    return df

# ---- Streamlit Interface ----
st.set_page_config(page_title="Earnings Analyzer", layout="wide")
st.title("📊 Earnings Options Analyzer")

uploaded_file = st.file_uploader("Upload Excel file with multiple sheets", type="xlsx")

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    dfs = []
    for sheet in xls.sheet_names:
        try:
            raw_df = xls.parse(sheet)
            parts = sheet.split()
            ticker = parts[0] if len(parts) > 0 else "UNKNOWN"
            date_str = parts[1] if len(parts) > 1 else None
            earnings_date = pd.to_datetime(date_str, format="%m-%d-%y") if date_str else None

            normalized_df = normalize_df(raw_df, ticker, earnings_date)
            dfs.append(normalized_df)
        except Exception as e:
            st.warning(f"Error processing sheet '{sheet}': {e}")

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)

        tickers = df_all['Ticker'].unique().tolist()
        selected = st.sidebar.selectbox("Select Ticker", ['All'] + tickers)

        if selected != 'All':
            df_all = df_all[df_all['Ticker'] == selected]

        st.subheader(f"Showing data for: {selected}")

        # --- Chart 1: Notional by Position Type ---
        pos_df = df_all.groupby('Position Type', as_index=False, observed=True)['Notional'].sum()
        pos_df['Percent'] = pos_df['Notional'] / pos_df['Notional'].sum() * 100
        pos_order = ['Call ITM', 'Call OTM', 'Put ITM', 'Put OTM']
        pos_df['Position Type'] = pd.Categorical(pos_df['Position Type'], categories=pos_order, ordered=True)

        fig1 = px.bar(
            pos_df, x='Position Type', y='Notional', text_auto='.2s', custom_data=['Percent'],
            title='Total Notional by Option Position Type'
        )
        fig1.update_traces(hovertemplate='%{x}<br>%{y:,.0f}<br><b>%{customdata[0]:.1f}%</b><extra></extra>')
        fig1.update_layout(xaxis_title='Option Type', yaxis_title='Total Notional', bargap=0.3)
        fig1.add_vrect(x0=-0.5, x1=1.5, fillcolor='lightgreen', opacity=0.15, layer='below', line_width=0)
        fig1.add_vrect(x0=1.5, x1=3.5, fillcolor='lightcoral', opacity=0.15, layer='below', line_width=0)
        st.plotly_chart(fig1, use_container_width=True)

        # Add other charts here in sequence
        # TODO: Spread vs Position Type
        if 'Spread' in df_all.columns:
            grouped_df = df_all.groupby(['Position Type', 'Spread'], observed=True, as_index=False)['Notional'].sum()
            grouped_df['Total Notional'] = grouped_df.groupby('Position Type')['Notional'].transform('sum')
            grouped_df['Percent'] = grouped_df['Notional'] / grouped_df['Total Notional'] * 100

            grouped_df['Position Type'] = pd.Categorical(grouped_df['Position Type'], categories=pos_order, ordered=True)
            grouped_df['Spread'] = pd.Categorical(grouped_df['Spread'], categories=['Multi', 'Single'], ordered=True)

            fig = px.bar(
                grouped_df, x='Position Type', y='Notional', color='Spread', barmode='stack',
                title='Total Notional by Option Type and Spread Type',
                text_auto='.2s', custom_data=['Percent'],
                color_discrete_map={'Multi': '#1f77b4', 'Single': '#ff7f0e'}
            )
            fig.update_traces(hovertemplate='%{customdata[0]:.1f}%<extra></extra>')
            fig.update_layout(xaxis_title='Option Type', yaxis_title='Total Notional', legend_title='Spread Type')
            fig.add_vrect(x0=-0.5, x1=1.5, fillcolor='lightgreen', opacity=0.15, layer='below', line_width=0)
            fig.add_vrect(x0=1.5, x1=3.5, fillcolor='lightcoral', opacity=0.15, layer='below', line_width=0)

            st.plotly_chart(fig, use_container_width=True)

        # TODO: Expiration Bucket breakdown
        grouped_df = df_all.groupby(['Position Type', 'Expiration Bucket'], observed=True, as_index=False)['Notional'].sum()
        grouped_df['Total Notional'] = grouped_df.groupby('Position Type')['Notional'].transform('sum')
        grouped_df['Percent'] = grouped_df['Notional'] / grouped_df['Total Notional'] * 100

        grouped_df['Expiration Bucket'] = pd.Categorical(grouped_df['Expiration Bucket'], categories=['Week', 'Month', 'LEAP'], ordered=True)
        grouped_df['Position Type'] = pd.Categorical(grouped_df['Position Type'], categories=pos_order, ordered=True)

        fig = px.bar(
            grouped_df, x='Position Type', y='Notional', color='Expiration Bucket',
            barmode='stack', title='Notional by Position Type & Expiration',
            text_auto='.2s', custom_data=['Percent'],
            color_discrete_map={'Week': '#1f77b4', 'Month': '#419ede', 'LEAP': '#ff7f0e'}
        )
        fig.update_traces(hovertemplate='%{customdata[0]:.1f}%<extra></extra>')
        fig.update_layout(xaxis_title='Option Type', yaxis_title='Total Notional', legend_title='Expiration Bucket')
        fig.add_vrect(x0=-0.5, x1=1.5, fillcolor='lightgreen', opacity=0.15, layer='below', line_width=0)
        fig.add_vrect(x0=1.5, x1=3.5, fillcolor='lightcoral', opacity=0.15, layer='below', line_width=0)

        st.plotly_chart(fig, use_container_width=True)

        # TODO: Lean (Bid/Ask)
        lean_summary = df_all.groupby(['Position Type', 'Lean'], observed=True, as_index=False)['Notional'].sum()
        lean_summary['Total Notional'] = lean_summary.groupby('Position Type')['Notional'].transform('sum')
        lean_summary['Percent'] = lean_summary['Notional'] / lean_summary['Total Notional'] * 100

        lean_summary['Position Type'] = pd.Categorical(lean_summary['Position Type'], categories=pos_order, ordered=True)
        lean_summary['Lean'] = pd.Categorical(lean_summary['Lean'], categories=['Ask', 'Bid'], ordered=True)

        fig = px.bar(
            lean_summary, x='Position Type', y='Notional', color='Lean', barmode='stack',
            title='Total Notional by Option Type and Lean',
            text_auto='.2s', custom_data=['Percent'],
            color_discrete_map={'Ask': '#1f77b4', 'Bid': '#ff7f0e'}
        )
        fig.update_traces(hovertemplate='%{customdata[0]:.1f}%<extra></extra>')
        fig.update_layout(xaxis_title='Option Type', yaxis_title='Total Notional', legend_title='Lean')
        fig.add_vrect(x0=-0.5, x1=1.5, fillcolor='lightgreen', opacity=0.15, layer='below', line_width=0)
        fig.add_vrect(x0=1.5, x1=3.5, fillcolor='lightcoral', opacity=0.15, layer='below', line_width=0)

        st.plotly_chart(fig, use_container_width=True)

        # TODO: Moneyness distribution (histogram, violin, box)

        df_itm = df_all[df_all['Moneyness'] < 0]
        df_otm = df_all[df_all['Moneyness'] > 0]

        summary_stats = df_all.groupby('Position Type', observed=True)['Moneyness'].agg(
            Mean='mean',
            Median='median',
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            Max='max',
            Min='min'
        )

        summary_stats['IQR'] = summary_stats['Q3'] - summary_stats['Q1']
        summary_stats = summary_stats[['Mean', 'Median', 'Q1', 'Q3', 'Max', 'Min', 'IQR']]
        summary_display = (summary_stats * 100).round(2).astype(str) + '%'

        st.subheader("Summary Statistics by Position Type")
        st.dataframe(summary_display)

        fig = px.histogram(
            df_all,
            x='Moneyness',
            y='Notional',
            color='Option Type',
            nbins=40,
            barmode='overlay',
            opacity=0.6,
            title='Moneyness Distribution: Calls vs Puts',
            color_discrete_map={
                'C': "#2ca02c",  # Green
                'P': "#d62728",  # Red
            }
        )

        fig.update_layout(
            xaxis_title='Moneyness (± % from Underlying)',
            yaxis_title='Count of Trades',
            legend_title='Option Type',
            bargap=0.05
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="ATM",
            annotation_position="top left"
        )

        st.subheader("Moneyness Histogram")
        st.plotly_chart(fig, use_container_width=True)


        df_all['Delta Side'] = df_all['Delta'].apply(lambda x: 'Positive Delta' if x > 0 else 'Negative Delta')

        fig = px.histogram(
            df_all, x='Delta', y='Delta Notional', color='Delta Side',
            nbins=40, barmode='overlay', opacity=0.7,
            title='Delta Distribution: Positive vs Negative',
            color_discrete_map={'Positive Delta': '#2ca02c', 'Negative Delta': '#d62728'}
        )
        fig.update_layout(xaxis_title='Delta', yaxis_title='Delta Notional')
        fig.add_vline(x=0, line_dash='dash', line_color='gray', annotation_text='Zero', annotation_position='top left')

        st.plotly_chart(fig, use_container_width=True)

        # TODO: Delta distribution
        df_itm = df_all[df_all['Moneyness'] < 0]
        df_otm = df_all[df_all['Moneyness'] > 0]

        for subset, label in [(df_itm, 'ITM'), (df_otm, 'OTM')]:
            fig = px.violin(subset, x='Position Type', y='Moneyness', box=True, points='all',
                            title=f'{label} Moneyness Distribution (Moneyness {"< 0" if label == "ITM" else "> 0"})')
            fig.update_layout(yaxis_title='Moneyness', xaxis_title='Position Type')
            fig.add_vrect(x0=-0.5, x1=0.5, fillcolor='lightgreen', opacity=0.15, layer='below', line_width=0)
            fig.add_vrect(x0=0.5, x1=1.5, fillcolor='lightcoral', opacity=0.15, layer='below', line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        for subset, label in [(df_itm, 'ITM'), (df_otm, 'OTM')]:
            fig = px.box(subset, x='Position Type', y='Moneyness', points='all',
                        title=f'Box Plot of {label} Moneyness by Option Position Type')
            fig.update_layout(yaxis_title='Moneyness', xaxis_title='Position Type')
            fig.add_vrect(x0=-0.5, x1=0.5, fillcolor='lightgreen', opacity=0.15, layer='below', line_width=0)
            fig.add_vrect(x0=0.5, x1=1.5, fillcolor='lightcoral', opacity=0.15, layer='below', line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        # Group by Strike Price and Sentiment to sum up Notional
        grouped = df_all.groupby(['Strike Price', 'Sentiment'], as_index=False)['Notional'].sum()

        # Optional: sort Strike Price for cleaner x-axis
        grouped = grouped.sort_values(by='Strike Price')

        # Plot aggregated data
        st.subheader("Aggregated Notional Volume by Strike Price and Sentiment")

        fig = px.bar(
            grouped,
            x='Strike Price',
            y='Notional',
            color='Sentiment',
            title='Notional Volume by Strike Price (Colored by Sentiment)',
            color_discrete_map={
                'Bullish': "#2ca02c",
                'Bearish': "#d62728"
            },
            barmode='stack'
        )

        fig.update_layout(
            xaxis_title='Strike Price',
            yaxis_title='Total Notional',
            bargap=0.1
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Next visualizations coming soon...")

else:
    st.info("Please upload an Excel file with multiple ticker sheets named like 'AAPL 01/24/25'")