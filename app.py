from __future__ import annotations

import os
import re
from datetime import timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

# =========================
# Plotly Theme & Constants
# =========================
CUSTOM_COLORS = [
    "#1f77b4", "#419ede", "#708090", "#9467bd", "#7f7f7f",
    "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#2ca02c",
    "#6a5acd", "#708090"
]
pio.templates["custom"] = pio.templates["plotly_white"].update(
    layout={"colorway": CUSTOM_COLORS}
)
pio.templates.default = "custom"

POS_ORDER = ["Call ITM", "Call OTM", "Put ITM", "Put OTM"]
SPREAD_ORDER = ["Multi", "Single"]
EXP_BUCKET_ORDER = ["LEAP", "Month", "Week"]
LEAN_ORDER = ["Ask", "Bid"]

# =========================
# Utility Helpers
# =========================

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_notional(df: pd.DataFrame) -> pd.Series:
    # Prefer existing Notional if valid; else derive Qty * Price * 100
    if "Notional" in df.columns:
        n = _to_num(df["Notional"])  # type: ignore[index]
        if n.notna().any():
            return n
    qty = _to_num(df.get("Qty", pd.Series(index=df.index)))
    price = _to_num(df.get("Price", pd.Series(index=df.index)))
    return qty.fillna(0) * price.fillna(0) * 100


def _add_mid_lean(df: pd.DataFrame) -> None:
    if {"Bid", "Ask", "Price"}.issubset(df.columns):
        df["Midpoint"] = (_to_num(df["Bid"]) + _to_num(df["Ask"])) / 2
        df["Lean"] = np.where(_to_num(df["Price"]) >= df["Midpoint"], "Ask", "Bid")
        df["MidLean"] = np.where(_to_num(df["Price"]) > df["Midpoint"], "Ask", "Bid")
    else:
        df["Lean"] = df.get("Lean", pd.Series(index=df.index)).fillna("Unknown")


def _sentiment(row: pd.Series) -> Optional[str]:
    t, lean = row.get("Option Type"), row.get("Lean")
    if t == "C" and lean == "Ask":
        return "Bullish"
    if t == "P" and lean == "Bid":
        return "Bullish"
    if t == "P" and lean == "Ask":
        return "Bearish"
    if t == "C" and lean == "Bid":
        return "Bearish"
    return None


def _bucket_exp(exp: pd.Timestamp, earnings_date: Optional[pd.Timestamp]) -> str:
    exp = pd.to_datetime(exp, errors="coerce")
    if isinstance(earnings_date, str):
        earnings_date = pd.to_datetime(earnings_date, errors="coerce")
    if pd.isna(exp) or earnings_date is None or pd.isna(earnings_date):
        return "Unknown"
    if exp < earnings_date + timedelta(days=7):
        return "Week"
    if exp < earnings_date + timedelta(days=31):
        return "Month"
    return "LEAP"


def _moneyness(row: pd.Series) -> Optional[float]:
    und = _to_num(pd.Series([row.get("Underlying")])).iloc[0]
    strike = _to_num(pd.Series([row.get("Strike Price")])).iloc[0]
    t = row.get("Option Type")
    if pd.isna(und) or und == 0 or pd.isna(strike) or t not in {"C", "P"}:
        return None
    if t == "C":
        return strike / und - 1
    return 1 - strike / und


def _pos_type(row: pd.Series) -> Optional[str]:
    und = _to_num(pd.Series([row.get("Underlying")])).iloc[0]
    strike = _to_num(pd.Series([row.get("Strike Price")])).iloc[0]
    t = row.get("Option Type")
    if pd.isna(und) or pd.isna(strike) or t not in {"C", "P"}:
        return None
    if t == "C":
        return "Call OTM" if strike > und else "Call ITM"
    return "Put ITM" if strike > und else "Put OTM"


# =========================
# Normalization
# =========================

DATE_PATTERNS = [
    (re.compile(r'(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})'),  ("%Y","%m","%d")),      # 20250905
    (re.compile(r'(?P<m>\d{1,2})[-_](?P<d>\d{1,2})[-_](?P<y>\d{4})'), ("%m","%d","%Y")),  # 9-5-2025
    (re.compile(r'(?P<m>\d{1,2})[-_](?P<d>\d{1,2})[-_](?P<y>\d{2})'),  ("%m","%d","%y")),  # 9-5-25
]

def parse_ticker_date(text: str) -> Tuple[str, Optional[pd.Timestamp]]:
    # Guess ticker: last 1–6 uppercase letters chunk
    tickers = re.findall(r"(?<![A-Z])[A-Z]{1,6}(?![A-Z])", text.upper())
    ticker = tickers[-1] if tickers else "UNKNOWN"

    dt: Optional[pd.Timestamp] = None
    for rx, order in DATE_PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        gd = m.groupdict()
        y = gd.get("y")
        mo = gd.get("m")
        d = gd.get("d")
        if not (y and mo and d):
            continue
        # Build date string and format based on pattern
        if order == ("%Y", "%m", "%d"):              # YYYYMMDD
            datestr = f"{y}-{mo}-{d}"
            fmt = "%Y-%m-%d"
        elif order == ("%m", "%d", "%Y"):            # M-D-YYYY
            datestr = f"{mo}-{d}-{y}"
            fmt = "%m-%d-%Y"
        else:                                             # ("%m","%d","%y")  M-D-YY
            datestr = f"{mo}-{d}-{y}"
            fmt = "%m-%d-%y"
        try:
            dt = pd.to_datetime(datestr, format=fmt, errors="coerce")
        except Exception:
            dt = None
        if dt is not None and not pd.isna(dt):
            break
    return ticker, dt


def normalize_alt_format(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Time": "Time", "Root": "Ticker", "Expiry": "Expiration Date",
        "Type": "Option Type", "Strike": "Strike Price", "Qty": "Qty",
        "Price": "Price", "Notional": "Notional", "Bid": "Bid",
        "Ask": "Ask", "Side": "Lean", "Volatility": "IV",
        "Delta": "Delta", "Hedge Price": "Underlying", "Open Interest": "Open Interest",
        "Condition": "Condition"
    }
    cols = {k: v for k, v in rename_map.items() if k in df.columns}
    out = df.rename(columns=cols)
    # Ensure the canonical columns exist even if missing in source
    for v in rename_map.values():
        if v not in out.columns:
            out[v] = np.nan
    out["Spread"] = "Single"
    if "Condition" in out.columns:
        _cond = out["Condition"].astype(str).str.strip().str.lower()
        out["Spread"] = np.where(_cond == "multileg", "Multi", "Single")
    else:
        out["Spread"] = "Single"
    return out[list(rename_map.values()) + ["Spread"]]


def normalize_uw_format(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Clean junk rows
    out = out.dropna(how="all")
    out = out[~out.apply(lambda r: r.astype(str).str.startswith("Loading").any(), axis=1)]

    # Parse columns from text
    if "Option" in out.columns:
        parts = out["Option"].astype(str).str.split(" ", expand=True)
        try:
            out["Expiration Date"] = pd.to_datetime(parts[0] + " " + parts[1] + " " + parts[2], format="%d %b %y")
        except Exception:
            out["Expiration Date"] = pd.NaT
        with pd.option_context('future.no_silent_downcasting', True):
            out["Strike Price"] = pd.to_numeric(parts[3], errors="coerce")
        out["Option Type"] = parts[4].str.upper().str[:1]

    if "Market" in out.columns:
        mk = out["Market"].astype(str).str.replace(r"[^\d.x]", "", regex=True).str.split("x", expand=True)
        out["Bid"] = pd.to_numeric(mk[0], errors="coerce")
        out["Ask"] = pd.to_numeric(mk[1], errors="coerce")

    # Spread normalization
    if "Spread" in out.columns:
        out["Spread"] = (
            out["Spread"].replace({"Spread": "Multi", "": "Single"}).fillna("Single")
        )
    else:
        out["Spread"] = "Single"

    # Ensure presence
    for c in ["Qty", "Price", "IV", "Delta", "Underlying", "Open Interest"]:
        if c not in out.columns:
            out[c] = np.nan

    return out


def normalize_df(df: pd.DataFrame, ticker: str, earnings_date: Optional[pd.Timestamp]) -> pd.DataFrame:
    is_alt = "Root" in df.columns or {"Expiry", "Hedge Price"}.issubset(df.columns)
    base = normalize_alt_format(df) if is_alt else normalize_uw_format(df)

    base = base.copy()
    base["Ticker"] = ticker
    base["Earnings Date"] = earnings_date

    # Types & derived fields
    base["IV"] = _to_num(base["IV"])  # type: ignore[index]
    base["Delta"] = _to_num(base["Delta"])  # type: ignore[index]
    base["Qty"] = _to_num(base["Qty"])  # type: ignore[index]
    base["Price"] = _to_num(base["Price"])  # type: ignore[index]
    base["Strike Price"] = _to_num(base["Strike Price"])  # type: ignore[index]
    base["Underlying"] = _to_num(base["Underlying"])  # type: ignore[index]

    _add_mid_lean(base)
    base["Sentiment"] = base.apply(_sentiment, axis=1)

    base["Expiration Date"] = pd.to_datetime(base["Expiration Date"], errors="coerce")
    base["Expiration Bucket"] = base["Expiration Date"].apply(lambda d: _bucket_exp(d, earnings_date))

    base["Moneyness"] = base.apply(_moneyness, axis=1)
    base["Position Type"] = base.apply(_pos_type, axis=1)

    base["Notional"] = _safe_notional(base)
    base["Delta Notional"] = base["Delta"].fillna(0) * base["Qty"].fillna(0)

    # Time column to pandas datetime if present
    if "Time" in base.columns:
        base["Time"] = pd.to_datetime(base["Time"], errors="coerce")

    return base


# =========================
# I/O & Caching
# =========================

@st.cache_data(show_spinner=False)
def load_input(uploaded_file) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        raw_df = pd.read_csv(uploaded_file)
        ticker, earnings_date = parse_ticker_date(os.path.splitext(filename)[0])
        dfs.append(normalize_df(raw_df, ticker, earnings_date))

    elif ext == ".xlsx":
        xls = pd.ExcelFile(uploaded_file)
        for sheet in xls.sheet_names:
            raw_df = xls.parse(sheet)
            ticker, earnings_date = parse_ticker_date(sheet)
            # Fallback to filename if the sheet name doesn't include ticker/date
            if ticker == "UNKNOWN" or earnings_date is None:
                fticker, fdate = parse_ticker_date(os.path.splitext(filename)[0])
                if ticker == "UNKNOWN":
                    ticker = fticker
                if earnings_date is None:
                    earnings_date = fdate
            dfs.append(normalize_df(raw_df, ticker, earnings_date))
    else:
        raise ValueError("Unsupported file format. Upload .csv or .xlsx")

    df_all = pd.concat(dfs, ignore_index=True)

    # Categorical orderings
    df_all["Position Type"] = pd.Categorical(df_all["Position Type"], categories=POS_ORDER, ordered=True)
    df_all["Spread"] = pd.Categorical(df_all["Spread"], categories=SPREAD_ORDER, ordered=True)
    df_all["Expiration Bucket"] = pd.Categorical(df_all["Expiration Bucket"], categories=EXP_BUCKET_ORDER, ordered=True)
    df_all["Lean"] = pd.Categorical(df_all["Lean"], categories=LEAN_ORDER, ordered=True)

    return df_all


# =========================
# Plotting Utilities
# =========================

def _atm_price(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    latest = df.sort_values("Time").groupby("Ticker").tail(1)
    if latest.empty:
        return None
    val = _to_num(latest["Underlying"]).iloc[0]
    return float(val) if np.isfinite(val) else None


def plot_notional_by_strike_sentiment(df: pd.DataFrame) -> None:
    if df.empty or "Strike Price" not in df.columns:
        st.info("No strike data available to plot.")
        return

    grouped = (
        df.groupby(["Strike Price", "Sentiment"], as_index=False)["Notional"].sum().sort_values("Strike Price")
    )

    fig = px.bar(
        grouped,
        x="Strike Price",
        y="Notional",
        color="Sentiment",
        title="Notional by Strike Price (colored by Sentiment)",
        barmode="stack",
        color_discrete_map={"Bullish": "#2ca02c", "Bearish": "#d62728"},
    )
    fig.update_layout(xaxis_title="Strike Price", yaxis_title="Total Notional", bargap=0.1)

    atm = _atm_price(df)
    if atm is not None and np.isfinite(atm):
        fig.add_vline(x=float(atm), line_dash="solid", line_color="orange", line_width=2, annotation_text="ATM", annotation_position="top left")

        # Top axis showing % distances from ATM
        pct_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00]
        min_x = float(grouped["Strike Price"].min())
        max_x = float(grouped["Strike Price"].max())
        level_points = [atm] + [atm * (1 + p) for p in pct_levels] + [atm * (1 - p) for p in pct_levels]
        vals = sorted({float(v) for v in level_points if np.isfinite(v) and min_x <= float(v) <= max_x})

        def _fmt_label(v: float) -> str:
            if np.isclose(v, atm, rtol=0, atol=max(atm * 1e-6, 1e-6)):
                return "ATM"
            pct = ((v / atm) - 1.0) * 100.0
            sign = "+" if pct > 0 else ""
            return f"{sign}{pct:.0f}%"

        ticktext = [_fmt_label(v) for v in vals]
        fig.update_layout(
            xaxis2=dict(
                overlaying="x",
                side="top",
                tickmode="array",
                tickvals=vals,
                ticktext=ticktext,
                title=dict(text="Distance from ATM", font=dict(size=12, color="#444")),
                tickfont=dict(size=11, color="#444"),
                showgrid=False,
            ),
            margin=dict(t=70)
        )

        # Conditionally cap the right side (> +100% OTM) unless it's material (>5% of notional)
        upper = float(atm * 2.0)  # +100% OTM in strike space
        # Compute share of notional beyond +100% OTM
        df_num = df.copy()
        df_num["Strike Price"] = _to_num(df_num["Strike Price"]) if "Strike Price" in df_num.columns else np.nan
        df_num["Notional"] = _safe_notional(df_num)
        mask_high = df_num["Strike Price"] > upper
        total_notional = float(df_num["Notional"].sum()) if df_num["Notional"].notna().any() else 0.0
        high_share = (float(df_num.loc[mask_high, "Notional"].sum()) / total_notional * 100.0) if total_notional > 0 else 0.0

        # Add white dotted lines at ±10% OTM with dollar annotations
        p10_dn = atm * 0.90
        p10_up = atm * 1.10

        if np.isfinite(p10_dn) and min_x <= float(p10_dn) <= max_x:
            fig.add_vline(
                x=float(p10_dn),
                line_dash="dot",
                line_color="white",
                line_width=2,
                annotation_text=f"-10%<br>${p10_dn:,.2f}",
                annotation_position="top left",
            )
        if np.isfinite(p10_up) and min_x <= float(p10_up) <= max_x:
            fig.add_vline(
                x=float(p10_up),
                line_dash="dot",
                line_color="white",
                line_width=2,
                annotation_text=f"+10%<br>${p10_up:,.2f}",
                annotation_position="top right",
            )

        if high_share <= 5.0:
            # Keep left bound as data min; cap right at +100% OTM
            fig.update_xaxes(range=[min_x, upper])

    st.plotly_chart(fig, use_container_width=True)

    # ---- Top 3 strike distributions above/below ATM (by total notional, using Sentiment) ----
    if atm is not None and np.isfinite(atm):
        work = df.copy()
        work["Strike Price"] = _to_num(work.get("Strike Price", pd.Series(index=work.index)))
        work["Notional"] = _safe_notional(work)
        # Keep only rows with explicit Bullish/Bearish sentiment
        work = work[work["Strike Price"].notna() & work["Notional"].notna() & work["Sentiment"].isin(["Bullish", "Bearish"])]
        if not work.empty:
            total_n = float(work["Notional"].sum()) if work["Notional"].notna().any() else 0.0

            # Aggregate by Strike and Sentiment
            agg = work.groupby(["Strike Price", "Sentiment"], as_index=False)["Notional"].sum()
            piv = agg.pivot(index="Strike Price", columns="Sentiment", values="Notional").fillna(0)
            for c in ("Bullish", "Bearish"):
                if c not in piv.columns:
                    piv[c] = 0.0
            piv["Total Notional"] = piv[["Bullish", "Bearish"]].sum(axis=1)
            piv = piv.reset_index().sort_values("Strike Price")

            # Derived per‑strike metrics
            piv["% OTM vs ATM"] = (piv["Strike Price"] / atm - 1.0) * 100.0
            piv["Sentiment Bias"] = np.where(piv["Bullish"] > piv["Bearish"], "Bullish",
                                       np.where(piv["Bullish"] < piv["Bearish"], "Bearish", "Tie"))
            piv["Bullish/Bearish Ratio"] = np.where(piv["Bearish"] > 0, piv["Bullish"] / piv["Bearish"], np.nan)
            piv["% of Total"] = np.where(total_n > 0, piv["Total Notional"] / total_n * 100.0, 0.0)

            # Split and rank by total notional
            above = piv[piv["Strike Price"] >= atm].nlargest(3, "Total Notional")
            below = piv[piv["Strike Price"] <  atm].nlargest(3, "Total Notional")

            def _fmt_tbl(a: pd.DataFrame) -> pd.DataFrame:
                out = a[["Strike Price", "% OTM vs ATM", "Total Notional", "% of Total", "Sentiment Bias", "Bullish", "Bearish", "Bullish/Bearish Ratio"]].copy()
                out.rename(columns={
                    "Strike Price": "Strike",
                    "% OTM vs ATM": "% OTM",
                }, inplace=True)
                # Pretty formatting
                out["Strike"] = out["Strike"].map(lambda v: f"{float(v):,.2f}" if pd.notna(v) else "")
                out["% OTM"] = out["% OTM"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")
                out["Total Notional"] = out["Total Notional"].map(lambda v: f"{float(v):,.0f}" if pd.notna(v) else "")
                out["% of Total"] = out["% of Total"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")
                out["Bullish"] = out["Bullish"].map(lambda v: f"{float(v):,.0f}" if pd.notna(v) else "")
                out["Bearish"] = out["Bearish"].map(lambda v: f"{float(v):,.0f}" if pd.notna(v) else "")
                out["Bullish/Bearish Ratio"] = out["Bullish/Bearish Ratio"].map(lambda v: f"{float(v):.2f}" if pd.notna(v) and np.isfinite(v) else "∞" if (pd.notna(v) and v == np.inf) else "0.00" if pd.notna(v) and v == 0 else "")
                return out

            with st.expander("Top Strike Distributions (by Sentiment)", expanded=False):
                if not above.empty:
                    st.markdown("**Top 3 Above ATM**")
                    st.dataframe(_fmt_tbl(above), use_container_width=True)
                else:
                    st.caption("No strikes above ATM.")
                if not below.empty:
                    st.markdown("**Top 3 Below ATM**")
                    st.dataframe(_fmt_tbl(below), use_container_width=True)
                else:
                    st.caption("No strikes below ATM.")


def plot_notional_by_expiration_date(df: pd.DataFrame) -> None:
    """Stacked bar of Bullish vs Bearish Notional by exact expiration date."""
    if df.empty or "Expiration Date" not in df.columns or "Sentiment" not in df.columns:
        return

    tmp = df[(df["Expiration Date"].notna()) & (df["Sentiment"].isin(["Bullish", "Bearish"]))].copy()
    if tmp.empty:
        st.info("No bullish/bearish notional to plot for expiration dates.")
        return

    g = (
        tmp.groupby(["Expiration Date", "Sentiment"], as_index=False)["Notional"].sum()
          .sort_values("Expiration Date")
    )

    if g["Notional"].fillna(0).sum() == 0:
        st.info("No notional to plot for expiration dates.")
        return

    fig = px.bar(
        g,
        x="Expiration Date",
        y="Notional",
        color="Sentiment",
        barmode="stack",
        title="Bullish vs Bearish Notional by Expiration Date",
        color_discrete_map={"Bullish": "#2ca02c", "Bearish": "#d62728"},
        text_auto=".2s",
    )
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{legendgroup}: %{y:,.0f}<extra></extra>")
    fig.update_layout(xaxis_title="Expiration Date", yaxis_title="Total Notional", legend_title="Sentiment", bargap=0.1)
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Underlying Data: Bullish/Bearish Notional by Expiration Date", expanded=False):
        st.dataframe(
            g.pivot(index="Expiration Date", columns="Sentiment", values="Notional").fillna(0).reset_index()
        )

def plot_sentiment_vs_atm(df: pd.DataFrame) -> None:
    """
    Notional sum by: Bullish/Bearish × Above/Below ATM.
    - ATM per ticker = latest Underlying in the view (same as _atm_price logic, but per-ticker).
    - 'Above ATM' means Strike Price >= ATM (ties counted as Above).
    """
    if df.empty or "Sentiment" not in df.columns or "Strike Price" not in df.columns:
        return

    # Build per-ticker ATM map using the latest row per ticker
    latest = df.sort_values("Time").groupby("Ticker", as_index=False).tail(1)
    atm_map = latest.set_index("Ticker")["Underlying"].map(_to_num).to_dict()

    work = df.copy()
    work["ATM_Underlying"] = work["Ticker"].map(atm_map)
    work["Strike Price"] = _to_num(work["Strike Price"])
    work = work[
        work["ATM_Underlying"].notna()
        & work["Strike Price"].notna()
        & work["Sentiment"].isin(["Bullish", "Bearish"])
    ].copy()
    if work.empty:
        st.info("No rows with ATM and Sentiment to plot.")
        return

    work["ATM Side"] = np.where(work["Strike Price"] >= work["ATM_Underlying"], "Above ATM", "Below ATM")

    g = (
        work.groupby(["ATM Side", "Sentiment"], as_index=False)["Notional"]
            .sum()
            .sort_values(["ATM Side", "Sentiment"])
    )
    if g["Notional"].fillna(0).sum() == 0:
        st.info("No notional found for Sentiment vs ATM.")
        return

    fig = px.bar(
        g,
        x="ATM Side",
        y="Notional",
        color="Sentiment",
        barmode="group",
        text_auto=".2s",
        title="Bullish vs Bearish Notional Above/Below ATM",
        color_discrete_map={"Bullish": "#2ca02c", "Bearish": "#d62728"},
    )
    fig.update_traces(hovertemplate="%{x}<br>%{legendgroup}: %{y:,.0f}<extra></extra>")
    fig.update_layout(xaxis_title="Relative to ATM", yaxis_title="Total Notional", legend_title="Sentiment")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Underlying Data: Sentiment × ATM side", expanded=False):
        st.dataframe(g.pivot(index="ATM Side", columns="Sentiment", values="Notional").fillna(0).reset_index())


# =========================
# Notional by OTM Bin Plot
# =========================

def plot_notional_by_otm_bins(df: pd.DataFrame) -> None:
    """Stacked bar of Notional by % OTM buckets (10% bands), colored by Sentiment.
    Positive moneyness => OTM, Negative => ITM (per existing definition).
    Also renders a table with per-bin shares and bias metrics.
    """
    if df.empty or "Moneyness" not in df.columns or "Sentiment" not in df.columns:
        return

    work = df.copy()
    work = work[work["Moneyness"].notna() & work["Notional"].notna()]
    if work.empty:
        st.info("No moneyness data to bucket.")
        return

    # Percent moneyness
    work["MoneynessPct"] = work["Moneyness"] * 100.0

    # Create dynamic 10% bins from -100 up to the next 10% above the observed max.
    # Keep lower bound at -100 (theoretical floor), but allow upper to exceed +100.
    observed_max = work["MoneynessPct"].max(skipna=True)
    upper = int(np.ceil((observed_max if np.isfinite(observed_max) else 100) / 10.0) * 10)
    upper = max(upper, 100)
    bins = np.arange(-100, upper + 10, 10)  # e.g., -100.. upper in 10% steps
    # Prevent tiny numeric noise below -100 from escaping the first bin; clamp upper bound so every row is binned
    work["MoneynessPct"] = work["MoneynessPct"].clip(lower=-100, upper=upper)
    cat = pd.cut(work["MoneynessPct"], bins=bins, right=True, include_lowest=True)

    # Map each bin to a numeric label: use the bin's RIGHT edge in percent (…, -90, -80, …, 0, 10, …)
    # We'll still show ticks from -100 through the dynamic upper bound for readability.
    edges_all = list(bins[1:])  # all right edges including 0 and the dynamic upper
    edge_map = {iv: int(iv.right) for iv in cat.cat.categories}
    work["OTM Edge"] = cat.map(edge_map)

    # Aggregate by numeric edge and sentiment
    g = (
        work.groupby(["OTM Edge", "Sentiment"], observed=True)["Notional"]
            .sum()
            .reset_index()
    )

    # Ensure all edges are present, even if zero, and only keep Bullish/Bearish
    g = g.pivot(index="OTM Edge", columns="Sentiment", values="Notional").reindex(edges_all).fillna(0)
    for col in ["Bullish", "Bearish"]:
        if col not in g.columns:
            g[col] = 0
    g = g[["Bullish", "Bearish"]]
    g["Total"] = g.sum(axis=1)

    # Totals and conditional crop logic for >100% OTM share
    total_overall = float(g["Total"].sum())
    high_mask = g.index > 100
    high_share = float(g.loc[high_mask, "Total"].sum()) / total_overall * 100.0 if (total_overall > 0 and high_mask.any()) else 0.0
    crop_high = high_share <= 5.0

    # Plot stacked bar (numeric x)
    plot_df = g.reset_index().melt(id_vars=["OTM Edge", "Total"], value_vars=["Bullish", "Bearish"], var_name="Sentiment", value_name="Notional")
    # Provide left/right edges for hover so bins are explicit: (left, right]
    plot_df["LeftEdge"] = plot_df["OTM Edge"] - 10
    plot_df["Mid"] = plot_df["LeftEdge"] + 5  # midpoint of (left, right]

    # Helper to pretty-print signed ints with true minus sign
    def _fmt_signed(n: int) -> str:
        return f"{n}" if n >= 0 else f"\u2212{abs(n)}"  # use Unicode minus for negatives

    fig = px.bar(
        plot_df,
        x="Mid",
        y="Notional",
        color="Sentiment",
        barmode="stack",
        text_auto=".2s",
        title="Bullish Bearish Notional by % OTM (10% buckets)",
        color_discrete_map={"Bullish": "#2ca02c", "Bearish": "#d62728"},
        custom_data=["LeftEdge", "OTM Edge"],
    )
    # Hover shows explicit interval: (left, right] and bin midpoint
    fig.update_traces(hovertemplate="(%{customdata[0]}%, %{customdata[1]}%] mid=%{x}%<br>%{legendgroup}: %{y:,.0f}<extra></extra>")
    # ATM reference at 0% (matches the strike chart’s orange line)
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="orange",
        line_width=2,
        annotation_text="ATM",
        annotation_position="top left",
    )
    tickvals = list(range(-95, (edges_all[-1] - 5) + 10, 10)) if edges_all else []  # midpoints: -95, -85, ..., -5, 5, 15, ...
    ticktext = [f"{_fmt_signed(v-5)}–{_fmt_signed(v+5)}" for v in tickvals]
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    if crop_high:
        fig.update_xaxes(range=[-95, 95])
    fig.update_layout(xaxis_title="% OTM (negative = ITM)", yaxis_title="Total Notional", legend_title="Sentiment")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bins are right-closed 10% intervals: (edge-10, edge]. The 0% bin is (−10–0], the 10% bin is (0–10].")

    # Build summary table using the same edge order
    if total_overall <= 0:
        st.info("No notional to summarize for % OTM bins.")
        return

    def _round_with_fix(weights: np.ndarray, basis: np.ndarray) -> np.ndarray:
        # weights are proportions that should sum to 100; basis used to choose the bin to absorb residual
        r = np.round(weights, 1)
        resid = np.round(100.0 - r.sum(), 1)
        if abs(resid) >= 0.1 and r.size > 0:
            idx = int(np.argmax(basis))
            r[idx] = np.round(r[idx] + resid, 1)
        return r

    # Raw shares (%) across bins
    w_total = (g["Total"].to_numpy() / total_overall) * 100.0
    w_bull = (g["Bullish"].to_numpy() / total_overall) * 100.0
    w_bear = (g["Bearish"].to_numpy() / total_overall) * 100.0

    # Rounded shares that sum to exactly 100.0 within each row set
    share_total = _round_with_fix(w_total, g["Total"].to_numpy())
    share_bull = _round_with_fix(w_bull, g["Bullish"].to_numpy())
    share_bear = _round_with_fix(w_bear, g["Bearish"].to_numpy())

    # Per-bin lean and bullish percent of bin
    bin_total = g["Total"].replace(0, np.nan)
    bin_bull_pct = (g["Bullish"] / bin_total * 100.0).round(1).fillna(0.0)
    bin_lean = np.where(g["Bullish"] > g["Bearish"], "Bullish", np.where(g["Bullish"] < g["Bearish"], "Bearish", "Tie"))

    table = pd.DataFrame({"Metric": [
        "notional % of total",
        "notional bullish % total",
        "notional bearish % total",
        "column lean bullish/bearish",
        "column bullish %",
    ]})

    for i, edge in enumerate(g.index):  # edges in ascending order
        col_name = f"{int(edge)}%"
        table[col_name] = [
            f"{share_total[i]:.1f}%",
            f"{share_bull[i]:.1f}%",
            f"{share_bear[i]:.1f}%",
            bin_lean[i],
            f"{bin_bull_pct.iloc[i]:.1f}%",
        ]

    with st.expander("Summary table: % OTM buckets", expanded=False):
        # Bullish/Bearish split as whole-number percent of overall notional
        bull_total = float(g["Bullish"].sum())
        bear_total = float(g["Bearish"].sum())
        bull_pct = int(round((bull_total / total_overall) * 100)) if total_overall > 0 else 0
        bear_pct = max(0, 100 - bull_pct)
        st.caption(f"Bullish/Bearish Split: {bull_pct}/{bear_pct}")
        # Min/Max strike prices from the same filtered data used for binning
        sp = _to_num(work.get("Strike Price", pd.Series(dtype=float)))
        sp_min = float(np.nanmin(sp.to_numpy())) if sp.notna().any() else None
        sp_max = float(np.nanmax(sp.to_numpy())) if sp.notna().any() else None
        if sp_min is not None and sp_max is not None and np.isfinite(sp_min) and np.isfinite(sp_max):
            st.caption(f"Strike Price Range: {sp_min:,.2f} to {sp_max:,.2f}")
        else:
            st.caption("Strike Price Range: n/a")
        st.dataframe(table)


def _ratio_block(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    # Compute key ratios
    s = df.groupby("Position Type", observed=True)["Notional"].sum()
    call_otm = s.get("Call OTM", 0.0)
    call_itm = s.get("Call ITM", 0.0)
    put_otm = s.get("Put OTM", 0.0)
    put_itm = s.get("Put ITM", 0.0)
    call_total = call_otm + call_itm
    put_total = put_otm + put_itm
    return pd.DataFrame([
        {
            "Group": group_label,
            "Call OTM/ITM": call_otm / call_itm if call_itm else np.nan,
            "Put OTM/ITM": put_otm / put_itm if put_itm else np.nan,
            "Call/Put": call_total / put_total if put_total else np.nan,
            "OTM Call/Put": call_otm / put_otm if put_otm else np.nan,
            "ITM Call/Put": call_itm / put_itm if put_itm else np.nan,
        }
    ])


def plot_notional_by_position(df: pd.DataFrame) -> None:
    pos_df = df.groupby("Position Type", observed=True, as_index=False)["Notional"].sum()
    pos_df["Percent"] = pos_df["Notional"] / pos_df["Notional"].sum() * 100

    fig1 = px.bar(
        pos_df, x="Position Type", y="Notional", text_auto=".2s", custom_data=["Percent"], title="Total Notional by Option Position Type"
    )
    fig1.update_traces(hovertemplate="%{x}<br>%{y:,.0f}<br><b>%{customdata[0]:.1f}%</b><extra></extra>")
    fig1.update_layout(xaxis_title="Option Type", yaxis_title="Total Notional", bargap=0.3)
    fig1.add_vrect(x0=-0.5, x1=1.5, fillcolor="lightgreen", opacity=0.15, layer="below", line_width=0)
    fig1.add_vrect(x0=1.5, x1=3.5, fillcolor="lightcoral", opacity=0.15, layer="below", line_width=0)
    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("Underlying Data and Ratios: Notional by Position Type"):
        st.dataframe(pos_df[["Position Type", "Notional", "Percent"]].sort_values("Position Type"))
        st.dataframe(_ratio_block(df, "All"))


def plot_spread_breakdown(df: pd.DataFrame) -> None:
    if "Spread" not in df.columns:
        return
    g = df.groupby(["Position Type", "Spread"], observed=True, as_index=False)["Notional"].sum()
    g["Total Notional"] = g.groupby("Position Type")["Notional"].transform("sum")
    g["Percent"] = g["Notional"] / g["Total Notional"] * 100

    fig = px.bar(
        g,
        x="Position Type",
        y="Notional",
        color="Spread",
        barmode="stack",
        title="Total Notional by Option Type and Spread Type",
        text_auto=".2s",
        custom_data=["Percent"],
        color_discrete_map={"Multi": "#1f77b4", "Single": "#ff7f0e"},
    )
    fig.update_traces(hovertemplate="%{customdata[0]:.1f}%<extra></extra>")
    fig.update_layout(xaxis_title="Option Type", yaxis_title="Total Notional", legend_title="Spread Type")
    fig.add_vrect(x0=-0.5, x1=1.5, fillcolor="lightgreen", opacity=0.15, layer="below", line_width=0)
    fig.add_vrect(x0=1.5, x1=3.5, fillcolor="lightcoral", opacity=0.15, layer="below", line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Underlying Data and Ratios: Notional by Position Type and Spread"):
        st.dataframe(g.sort_values(["Position Type", "Spread"]))
        rows = []
        for spread in SPREAD_ORDER:
            subset = g[g["Spread"] == spread]
            rows.append(_ratio_block(subset, spread))
        st.dataframe(pd.concat(rows, ignore_index=True))


def plot_expiration_bucket(df: pd.DataFrame) -> None:
    g = df.groupby(["Position Type", "Expiration Bucket"], observed=True, as_index=False)["Notional"].sum()
    g["Total Notional"] = g.groupby("Position Type")["Notional"].transform("sum")
    g["Percent"] = g["Notional"] / g["Total Notional"] * 100

    fig = px.bar(
        g,
        x="Position Type",
        y="Notional",
        color="Expiration Bucket",
        barmode="stack",
        title="Notional by Position Type & Expiration",
        text_auto=".2s",
        custom_data=["Percent"],
        color_discrete_map={"Week": "#1f77b4", "Month": "#419ede", "LEAP": "#ff7f0e"},
    )
    fig.update_traces(hovertemplate="%{customdata[0]:.1f}%<extra></extra>")
    fig.update_layout(xaxis_title="Option Type", yaxis_title="Total Notional", legend_title="Expiration Bucket")
    fig.add_vrect(x0=-0.5, x1=1.5, fillcolor="lightgreen", opacity=0.15, layer="below", line_width=0)
    fig.add_vrect(x0=1.5, x1=3.5, fillcolor="lightcoral", opacity=0.15, layer="below", line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Underlying Data and Ratios: Notional by Position Type and Expiration Bucket"):
        st.dataframe(g.sort_values(["Position Type", "Expiration Bucket"]))
        rows = []
        for bucket in EXP_BUCKET_ORDER:
            subset = g[g["Expiration Bucket"] == bucket]
            rows.append(_ratio_block(subset, bucket))
        ratio_df = pd.concat(rows, ignore_index=True)
        st.dataframe(ratio_df.sort_values("Call/Put", ascending=False))


def plot_lean(df: pd.DataFrame) -> None:
    g = df.groupby(["Position Type", "Lean"], observed=True, as_index=False)["Notional"].sum()
    g["Total Notional"] = g.groupby("Position Type")["Notional"].transform("sum")
    g["Percent"] = g["Notional"] / g["Total Notional"] * 100

    fig = px.bar(
        g,
        x="Position Type",
        y="Notional",
        color="Lean",
        barmode="stack",
        title="Total Notional by Option Type and Lean",
        text_auto=".2s",
        custom_data=["Percent"],
        color_discrete_map={"Ask": "#1f77b4", "Bid": "#ff7f0e"},
    )
    fig.update_traces(hovertemplate="%{customdata[0]:.1f}%<extra></extra>")
    fig.update_layout(xaxis_title="Option Type", yaxis_title="Total Notional", legend_title="Lean")
    fig.add_vrect(x0=-0.5, x1=1.5, fillcolor="lightgreen", opacity=0.15, layer="below", line_width=0)
    fig.add_vrect(x0=1.5, x1=3.5, fillcolor="lightcoral", opacity=0.15, layer="below", line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Underlying Data and Ratios: Notional by Position Type and Lean"):
        st.dataframe(g.sort_values(["Position Type", "Lean"]))
        rows = []
        for lean_type in LEAN_ORDER:
            subset = g[g["Lean"] == lean_type]
            rows.append(_ratio_block(subset, lean_type))
        st.dataframe(pd.concat(rows, ignore_index=True).sort_values("Call/Put", ascending=False))


def plot_moneyness_box_violin(df: pd.DataFrame) -> None:
    df_itm = df[df["Moneyness"].notna() & (df["Moneyness"] < 0)]
    df_otm = df[df["Moneyness"].notna() & (df["Moneyness"] > 0)]

    # Summary stats table
    summary = df.groupby("Position Type", observed=True)["Moneyness"].agg(
        Mean="mean",
        Median="median",
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75),
        Max="max",
        Min="min",
    )
    summary["IQR"] = summary["Q3"] - summary["Q1"]
    st.subheader("Summary Statistics by Position Type")
    st.dataframe((summary * 100).round(2).astype(str) + "%")

    subsets = {"ITM": df_itm, "OTM": df_otm}

    # First: Box plots for ITM then OTM
    for label in ["ITM", "OTM"]:
        subset = subsets[label]
        if subset.empty:
            continue
        fig = px.box(subset, x="Position Type", y="Moneyness", points="all", title=f"Box Plot of {label} Moneyness by Position Type")
        fig.update_layout(yaxis_title="Moneyness", xaxis_title="Position Type")
        fig.add_vrect(x0=-0.5, x1=0.5, fillcolor="lightgreen", opacity=0.15, layer="below", line_width=0)
        fig.add_vrect(x0=0.5, x1=1.5, fillcolor="lightcoral", opacity=0.15, layer="below", line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # Then: Violin plots for ITM then OTM
    for label in ["ITM", "OTM"]:
        subset = subsets[label]
        if subset.empty:
            continue
        fig = px.violin(subset, x="Position Type", y="Moneyness", box=True, points="all", title=f"{label} Moneyness Distribution")
        fig.update_layout(yaxis_title="Moneyness", xaxis_title="Position Type")
        fig.add_vrect(x0=-0.5, x1=0.5, fillcolor="lightgreen", opacity=0.15, layer="below", line_width=0)
        fig.add_vrect(x0=0.5, x1=1.5, fillcolor="lightcoral", opacity=0.15, layer="below", line_width=0)
        st.plotly_chart(fig, use_container_width=True)


def plot_moneyness_hist(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig = px.histogram(
        df,
        x="Moneyness",
        y="Notional",
        color="Option Type",
        histfunc="sum",
        nbins=60,
        barmode="overlay",
        opacity=0.6,
        title="Moneyness Distribution: Calls vs Puts",
        color_discrete_map={"C": "#2ca02c", "P": "#d62728"},
    )
    fig.update_traces(xbins=dict(size=0.05))
    fig.update_layout(xaxis_title="Moneyness (± % from Underlying)", yaxis_title="Total Notional", legend_title="Option Type", bargap=0.05)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="ATM", annotation_position="top left")
    st.subheader("Moneyness Histogram")
    st.plotly_chart(fig, use_container_width=True)


def plot_delta_hist(df: pd.DataFrame) -> None:
    df = df.copy()
    df["Delta Side"] = np.where(_to_num(df["Delta"]) > 0, "Positive Delta", "Negative Delta")
    fig = px.histogram(
        df, x="Delta", y="Delta Notional", color="Delta Side", nbins=40, barmode="overlay", opacity=0.7,
        title="Delta Distribution: Positive vs Negative", color_discrete_map={"Positive Delta": "#2ca02c", "Negative Delta": "#d62728"}
    )
    fig.update_layout(xaxis_title="Delta", yaxis_title="Delta Notional")
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Zero", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)


def plot_hourly_activity(df: pd.DataFrame) -> None:
    if "Time" not in df.columns or df["Time"].isna().all():
        return
    df = df.copy()
    df["Hour Label"] = df["Time"].dt.strftime("%H:00")
    hour_df = df.groupby("Hour Label", as_index=False)["Notional"].sum().sort_values("Hour Label")
    total = hour_df["Notional"].sum()
    hour_df["Percent"] = np.where(total > 0, hour_df["Notional"] / total * 100, 0)

    fig = px.bar(
        hour_df, x="Hour Label", y="Notional",
        title="Total Notional Volume by Hour", labels={"Hour Label": "Hour", "Notional": "Total Notional"},
        text=hour_df["Percent"].map(lambda p: f"{p:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Hour of Day (ET)", yaxis_title="Total Notional Volume", uniformtext_minsize=8, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)


# =========================
# Streamlit App
# =========================

def main():
    st.set_page_config(page_title="Earnings Analyzer", layout="wide")
    st.title("📊 Earnings Options Analyzer")

    uploaded = st.file_uploader("Upload Excel (.xlsx) or CSV file", type=["xlsx", "csv"])

    if not uploaded:
        st.info("Upload a CSV named like 'AAPL 01-24-25.csv' or an Excel with sheets named like 'AAPL 01-24-25'.")
        return

    try:
        df_all = load_input(uploaded)
    except Exception as e:
        st.warning(f"Error processing file: {e}")
        return

    # Sidebar controls
    tickers = sorted([t for t in df_all["Ticker"].dropna().unique().tolist() if t])
    selected = st.sidebar.selectbox("Select Ticker", ["All"] + tickers)
    view = df_all if selected == "All" else df_all[df_all["Ticker"] == selected]

     # Debug: show parsed tickers and earnings dates before the header
    tickers_in_view = (
        view["Ticker"].dropna().astype(str).unique().tolist()
    )
    ed_series = pd.to_datetime(view["Earnings Date"], errors="coerce")
    ed_list = sorted({d.date().isoformat() for d in ed_series.dropna().tolist()})
    tickers_text = ", ".join(tickers_in_view) if tickers_in_view else "(none)"
    ed_text = ", ".join(ed_list) if ed_list else "(none)"
    st.caption(f"Ticker(s): {tickers_text}  |  Earnings date(s): {ed_text}")


    st.subheader(f"Showing data for: {selected}")
    st.markdown("---")

    with st.expander("Last Minute Analysis Graphs", expanded=True):
        # Notional Sentiment by Strike
        plot_notional_by_strike_sentiment(view)
        st.markdown("---")

        # % OTM buckets (10% bands): stacked Bullish/Bearish
        plot_notional_by_otm_bins(view)
        st.markdown("---")

        # Notional by Expiration Date
        plot_notional_by_expiration_date(view)
        st.markdown("---")

        # Bullish vs Bearish Above/Below ATM
        plot_sentiment_vs_atm(view)
        st.markdown("---")

        # Notional by Bid Ask Lean
        plot_lean(view)
        st.markdown("---")

        # Delta Positive vs Negative
        plot_delta_hist(view)

    with st.expander("Stats, Box/Violin, and Overview Graphs", expanded=False):
        plot_notional_by_position(view)
        st.markdown("---")

        plot_expiration_bucket(view)
        st.markdown("---")

        plot_spread_breakdown(view)
        st.markdown("---")

        plot_moneyness_box_violin(view)
        st.markdown("---")

        plot_moneyness_hist(view)

    with st.expander("Experimental Stats and Graphs", expanded=False):

        # Notional by Hour
        plot_hourly_activity(view)

    # st.subheader("Next visualizations coming soon…")


if __name__ == "__main__":
    main()
