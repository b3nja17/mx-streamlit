# streamlit_earnings_analyzer.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import io
import os
from datetime import timedelta
import numpy as np

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

uploaded_file = st.file_uploader("Upload Excel (.xlsx) or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    dfs = []
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".csv":
            raw_df = pd.read_csv(uploaded_file)
            parts = os.path.splitext(filename)[0].split()
            ticker = parts[0] if len(parts) > 0 else "UNKNOWN"
            date_str = parts[1] if len(parts) > 1 else None
            earnings_date = pd.to_datetime(date_str, format="%m-%d-%y") if date_str else None

            normalized_df = normalize_df(raw_df, ticker, earnings_date)
            dfs.append(normalized_df)

        elif ext == ".xlsx":
            xls = pd.ExcelFile(uploaded_file)
            for sheet in xls.sheet_names:
                raw_df = xls.parse(sheet)
                parts = sheet.split()
                ticker = parts[0] if len(parts) > 0 else "UNKNOWN"
                date_str = parts[1] if len(parts) > 1 else None
                earnings_date = pd.to_datetime(date_str, format="%m-%d-%y") if date_str else None

                normalized_df = normalize_df(raw_df, ticker, earnings_date)
                dfs.append(normalized_df)
        else:
            st.warning("Unsupported file format.")

    except Exception as e:
        st.warning(f"Error processing file: {e}")

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)

        tickers = df_all['Ticker'].unique().tolist()
        selected = st.sidebar.selectbox("Select Ticker", ['All'] + tickers)

        if selected != 'All':
            df_all = df_all[df_all['Ticker'] == selected]

        st.subheader(f"Showing data for: {selected}")
        st.markdown("---")
############################################################################
        # Estimate ATM price as the underlying of the latest trade per ticker
        latest_trade = df_all.sort_values('Time').groupby('Ticker').tail(1)
        atm_price = latest_trade['Underlying'].values[0] if not latest_trade.empty else None
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

        ###########
        # Keep existing x-axis (Strike Price) exactly as-is; add a second overlaid axis that shows % levels
        fig.update_layout(
            xaxis_title='Strike Price',
            yaxis_title='Total Notional',
            bargap=0.1
        )

        if atm_price is not None and np.isfinite(atm_price):
            # Reference ATM line
            fig.add_vline(
                x=float(atm_price),
                line_dash='solid',
                line_color='orange',
                line_width=2,
                annotation_text='ATM',
                annotation_position='top left'
            )

            # Build the additional axis ticks at ATM ± given % levels
            pct_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00]
            # Include ATM itself for labeling
            level_points = [atm_price] + [atm_price * (1 + p) for p in pct_levels] + [atm_price * (1 - p) for p in pct_levels]

            # Filter to visible range of the data so labels don't float outside the plot
            min_x = float(grouped['Strike Price'].min())
            max_x = float(grouped['Strike Price'].max())

            # Deduplicate, sort, and keep only points within the plotted domain
            vals = sorted({float(v) for v in level_points if np.isfinite(v) and min_x <= float(v) <= max_x})

            # Map values to labels: ATM for center, ±N% for others based on distance from ATM
            def _fmt_label(v):
                if np.isclose(v, atm_price, rtol=0, atol=max(atm_price * 1e-6, 1e-6)):
                    return "ATM"
                pct = ((v / atm_price) - 1.0) * 100.0
                sign = "+" if pct > 0 else ""
                return f"{sign}{pct:.0f}%"

            ticktext = [_fmt_label(v) for v in vals]

            # Add a second x-axis overlaid on the first.
            # Keep the original axis at the bottom; put the % levels on the top so the existing axis remains untouched.
            fig.update_layout(
                xaxis2=dict(
                    overlaying="x",
                    side="top",
                    tickmode="array",
                    tickvals=vals,
                    ticktext=ticktext,
                    title="Distance from ATM"
                )
        )
        
        st.plotly_chart(fig, use_container_width=True)
############################################################################
        # --- Chart 1: Notional by Position Type ---
        pos_df = df_all.groupby('Position Type', as_index=False, observed=True)['Notional'].sum()
        pos_df['Percent'] = pos_df['Notional'] / pos_df['Notional'].sum() * 100
        pos_order = ['Call ITM', 'Call OTM', 'Put ITM', 'Put OTM']
        pos_df['Position Type'] = pd.Categorical(pos_df['Position Type'], categories=pos_order, ordered=True)
        pos_df['Call/Put'] = pos_df['Position Type'].apply(lambda x: 'Call' if x.startswith('Call') else 'Put')

        fig1 = px.bar(
            pos_df, x='Position Type', y='Notional', text_auto='.2s', custom_data=['Percent'],
            title='Total Notional by Option Position Type'
        )
        fig1.update_traces(hovertemplate='%{x}<br>%{y:,.0f}<br><b>%{customdata[0]:.1f}%</b><extra></extra>')
        fig1.update_layout(xaxis_title='Option Type', yaxis_title='Total Notional', bargap=0.3)
        fig1.add_vrect(x0=-0.5, x1=1.5, fillcolor='lightgreen', opacity=0.15, layer='below', line_width=0)
        fig1.add_vrect(x0=1.5, x1=3.5, fillcolor='lightcoral', opacity=0.15, layer='below', line_width=0)
        st.plotly_chart(fig1, use_container_width=True)

        # --- Data Table Section: Notional by Position Type (expander) ---
        call_otm = pos_df.loc[pos_df['Position Type'] == 'Call OTM', 'Notional'].sum()
        call_itm = pos_df.loc[pos_df['Position Type'] == 'Call ITM', 'Notional'].sum()
        put_otm = pos_df.loc[pos_df['Position Type'] == 'Put OTM', 'Notional'].sum()
        put_itm = pos_df.loc[pos_df['Position Type'] == 'Put ITM', 'Notional'].sum()
        call_total = call_otm + call_itm
        put_total = put_otm + put_itm
        ratio_row = {
            'Group': 'All',
            "Call OTM/ITM": call_otm / call_itm if call_itm else float('nan'),
            "Put OTM/ITM": put_otm / put_itm if put_itm else float('nan'),
            "Call/Put": call_total / put_total if put_total else float('nan'),
            "OTM Call/Put": call_otm / put_otm if put_otm else float('nan'),
            "ITM Call/Put": call_itm / put_itm if put_itm else float('nan')
        }
        ratio_df = pd.DataFrame([ratio_row])
        with st.expander("📄 Underlying Data and Ratios: Notional by Position Type"):
            st.subheader("Underlying Data: Notional by Position Type")
            st.dataframe(pos_df[['Call/Put', 'Position Type', 'Notional', 'Percent']].sort_values(by='Position Type'))
            st.subheader("Analytics Ratios")
            st.dataframe(ratio_df)
        st.markdown("---")

############################################################################
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

            # --- Data Table Section: Spread ---
            spread_ratios = []
            for spread in ['Multi', 'Single']:
                subset = grouped_df[grouped_df['Spread'] == spread]
                call_otm = subset.loc[subset['Position Type'] == 'Call OTM', 'Notional'].sum()
                call_itm = subset.loc[subset['Position Type'] == 'Call ITM', 'Notional'].sum()
                put_otm = subset.loc[subset['Position Type'] == 'Put OTM', 'Notional'].sum()
                put_itm = subset.loc[subset['Position Type'] == 'Put ITM', 'Notional'].sum()
                call_total = call_otm + call_itm
                put_total = put_otm + put_itm

                spread_ratios.append({
                    'Spread Type': spread,
                    'Call OTM/ITM': call_otm / call_itm if call_itm else float('nan'),
                    'Put OTM/ITM': put_otm / put_itm if put_itm else float('nan'),
                    'Call/Put': call_total / put_total if put_total else float('nan'),
                    'OTM Call/Put': call_otm / put_otm if put_otm else float('nan'),
                    'ITM Call/Put': call_itm / put_itm if put_itm else float('nan')
                })

            spread_ratio_df = pd.DataFrame(spread_ratios)
            spread_detail = df_all.groupby(['Position Type', 'Spread'], observed=True)['Notional'].sum().reset_index()
            spread_detail['Percent'] = spread_detail.groupby('Position Type')['Notional'].transform(lambda x: x / x.sum() * 100)
            spread_detail['Percent'] = spread_detail['Percent'].round(2)
            spread_pivot = spread_detail.pivot(index='Position Type', columns='Spread', values='Percent').fillna(0).reset_index()
            with st.expander("📄 Underlying Data and Ratios: Notional by Position Type and Spread"):
                st.subheader("Underlying Data: Notional by Position Type and Spread")
                st.dataframe(grouped_df.sort_values(by=['Position Type', 'Spread']))
                st.subheader("Analytics Ratios by Spread Type")
                st.dataframe(spread_ratio_df)
                st.subheader("Spread Breakdown by Position Type")
                st.dataframe(spread_pivot)
            st.markdown("---")

############################################################################
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

        # --- Data Table Section: Expiration ---
        exp_ratios = []
        for bucket in ['Week', 'Month', 'LEAP']:
            subset = grouped_df[grouped_df['Expiration Bucket'] == bucket]
            call_otm = subset.loc[subset['Position Type'] == 'Call OTM', 'Notional'].sum()
            call_itm = subset.loc[subset['Position Type'] == 'Call ITM', 'Notional'].sum()
            put_otm = subset.loc[subset['Position Type'] == 'Put OTM', 'Notional'].sum()
            put_itm = subset.loc[subset['Position Type'] == 'Put ITM', 'Notional'].sum()
            call_total = call_otm + call_itm
            put_total = put_otm + put_itm

            exp_ratios.append({
                'Expiration Bucket': bucket,
                'Call OTM/ITM': call_otm / call_itm if call_itm else float('nan'),
                'Put OTM/ITM': put_otm / put_itm if put_itm else float('nan'),
                'Call/Put': call_total / put_total if put_total else float('nan'),
                'OTM Call/Put': call_otm / put_otm if put_otm else float('nan'),
                'ITM Call/Put': call_itm / put_itm if put_itm else float('nan')
            })
        exp_ratio_df = pd.DataFrame(exp_ratios)
        expiration_detail = df_all.groupby(['Position Type', 'Expiration Bucket'], observed=True)['Notional'].sum().reset_index()
        expiration_detail['Percent'] = expiration_detail.groupby('Position Type')['Notional'].transform(lambda x: x / x.sum() * 100)
        expiration_detail['Percent'] = expiration_detail['Percent'].round(2)
        expiration_pivot = expiration_detail.pivot(index='Position Type', columns='Expiration Bucket', values='Percent').fillna(0).reset_index()
        with st.expander("📄 Underlying Data and Ratios: Notional by Position Type and Expiration Bucket"):
            st.subheader("Underlying Data: Notional by Position Type and Expiration Bucket")
            st.dataframe(grouped_df.sort_values(by=['Position Type', 'Expiration Bucket']))
            st.subheader("Analytics Ratios by Expiration Bucket")
            st.dataframe(exp_ratio_df.sort_values(by='Call/Put', ascending=False))
            st.subheader("Expiration Breakdown by Position Type")
            st.dataframe(expiration_pivot)
        st.markdown("---")

############################################################################
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

        # --- Data Table Section: Lean ---
        lean_ratios = []
        for lean_type in ['Ask', 'Bid']:
            subset = lean_summary[lean_summary['Lean'] == lean_type]
            call_otm = subset.loc[subset['Position Type'] == 'Call OTM', 'Notional'].sum()
            call_itm = subset.loc[subset['Position Type'] == 'Call ITM', 'Notional'].sum()
            put_otm = subset.loc[subset['Position Type'] == 'Put OTM', 'Notional'].sum()
            put_itm = subset.loc[subset['Position Type'] == 'Put ITM', 'Notional'].sum()
            call_total = call_otm + call_itm
            put_total = put_otm + put_itm

            lean_ratios.append({
                'Lean': lean_type,
                'Call OTM/ITM': call_otm / call_itm if call_itm else float('nan'),
                'Put OTM/ITM': put_otm / put_itm if put_itm else float('nan'),
                'Call/Put': call_total / put_total if put_total else float('nan'),
                'OTM Call/Put': call_otm / put_otm if put_otm else float('nan'),
                'ITM Call/Put': call_itm / put_itm if put_itm else float('nan')
            })
        lean_ratio_df = pd.DataFrame(lean_ratios)
        lean_detail = df_all.groupby(['Position Type', 'Lean'], observed=True)['Notional'].sum().reset_index()
        lean_detail['Percent'] = lean_detail.groupby('Position Type')['Notional'].transform(lambda x: x / x.sum() * 100)
        lean_detail['Percent'] = lean_detail['Percent'].round(2)
        lean_pivot = lean_detail.pivot(index='Position Type', columns='Lean', values='Percent').fillna(0).reset_index()
        with st.expander("📄 Underlying Data and Ratios: Notional by Position Type and Lean"):
            st.subheader("Underlying Data: Notional by Position Type and Lean")
            st.dataframe(lean_summary.sort_values(by=['Position Type', 'Lean']))
            st.subheader("Analytics Ratios by Lean")
            st.dataframe(lean_ratio_df.sort_values(by='Call/Put', ascending=False))
            st.subheader("Lean Breakdown by Position Type")
            st.dataframe(lean_pivot)
        st.markdown("---")

############################################################################
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

        # TODO: Delta distribution
        df_itm = df_all[df_all['Moneyness'] < 0]
        df_otm = df_all[df_all['Moneyness'] > 0]



        for subset, label in [(df_itm, 'ITM'), (df_otm, 'OTM')]:
            fig = px.box(subset, x='Position Type', y='Moneyness', points='all',
                        title=f'Box Plot of {label} Moneyness by Option Position Type')
            fig.update_layout(yaxis_title='Moneyness', xaxis_title='Position Type')
            fig.add_vrect(x0=-0.5, x1=0.5, fillcolor='lightgreen', opacity=0.15, layer='below', line_width=0)
            fig.add_vrect(x0=0.5, x1=1.5, fillcolor='lightcoral', opacity=0.15, layer='below', line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        for subset, label in [(df_itm, 'ITM'), (df_otm, 'OTM')]:
            fig = px.violin(subset, x='Position Type', y='Moneyness', box=True, points='all',
                            title=f'{label} Moneyness Distribution (Moneyness {"< 0" if label == "ITM" else "> 0"})')
            fig.update_layout(yaxis_title='Moneyness', xaxis_title='Position Type')
            fig.add_vrect(x0=-0.5, x1=0.5, fillcolor='lightgreen', opacity=0.15, layer='below', line_width=0)
            fig.add_vrect(x0=0.5, x1=1.5, fillcolor='lightcoral', opacity=0.15, layer='below', line_width=0)
            st.plotly_chart(fig, use_container_width=True)

############################################################################
        bin_size = 0.05
        fig = px.histogram(
            df_all,
            x='Moneyness',
            y='Notional',
            color='Option Type',
            histfunc='sum',
            nbins=None,
            barmode='overlay',
            opacity=0.6,
            title='Moneyness Distribution: Calls vs Puts',
            color_discrete_map={
                'C': "#2ca02c",  # Green
                'P': "#d62728",  # Red
            }
        )
        fig.update_traces(xbins=dict(size=bin_size))

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

        # --- Trade Activity by Hour ---
        st.markdown("---")
        st.subheader("Hourly Trade Activity (Notional Volume)")

        df_all['Hour Label'] = df_all['Time'].apply(lambda t: f"{t.hour:02d}:00")

        hour_df = df_all.groupby('Hour Label', as_index=False)['Notional'].sum()
        hour_df = hour_df.sort_values('Hour Label')

        total_notional = hour_df['Notional'].sum()
        hour_df['Percent'] = hour_df['Notional'] / total_notional * 100

        fig = px.bar(
            hour_df, x='Hour Label', y='Notional',
            title='Total Notional Volume by Hour',
            labels={'Hour Label': 'Hour', 'Notional': 'Total Notional'},
            text=hour_df['Percent'].map(lambda p: f"{p:.1f}%")
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title='Hour of Day (ET)',
            yaxis_title='Total Notional Volume',
            uniformtext_minsize=8,
            uniformtext_mode='hide'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Next visualizations coming soon...")

else:
    st.info("Please upload an Excel (.xlsx) or CSV file. For CSVs, the filename should be like 'AAPL 01-24-25.csv'. For Excel, use multiple sheets named like 'AAPL 01-24-25'.")
