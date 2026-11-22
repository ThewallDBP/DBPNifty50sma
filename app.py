import datetime
import warnings
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION (Edit as needed)
# ==========================================
THRESHOLD_PCT = 2.0            # Flag if Close is within +/- 2.0% of SMA
SMAS_TO_CHECK = [20, 50, 100, 200]  # Add 10 here if you want 10 SMA too
SAVE_CSV = True                # Saves a dated CSV report in the same folder

# Major NSE Indices (Yahoo Finance symbols)
INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY MIDCAP 50": "^NSEMDCP50",
}

# Nifty 50 Stocks (Add any extra F&O stocks with ".NS" suffix)
STOCKS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
    "ITC.NS", "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    # Example extra F&O leaders:
    "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "HAL.NS", "RECLTD.NS", "PFC.NS"
]


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Calculates Wilder's RSI(14) matching TradingView closely."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2) if not rsi.empty else None


def extract_close_df(raw_data: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Safely extracts Close prices across different yfinance versions."""
    if isinstance(raw_data.columns, pd.MultiIndex):
        close_df = raw_data["Close"]
    else:
        close_df = raw_data[["Close"]]
        close_df.columns = tickers[:1]
    return close_df


def scan_timeframe(close_df: pd.DataFrame, name_map: dict, timeframe_label: str) -> pd.DataFrame:
    """Scans all tickers in close_df for proximity to SMAs."""
    hits = []

    for ticker in close_df.columns:
        series = close_df[ticker].dropna()
        if len(series) < 20:
            continue

        last_close = float(series.iloc[-1])
        rsi_val = compute_rsi(series, 14)
        display_name = name_map.get(ticker, ticker.replace(".NS", ""))
        category = "INDEX" if ticker in name_map else "STOCK"

        for period in SMAS_TO_CHECK:
            if len(series) < period:
                continue

            sma_val = float(series.rolling(window=period).mean().iloc[-1])
            dist_pct = ((last_close - sma_val) / sma_val) * 100

            if abs(dist_pct) <= THRESHOLD_PCT:
                position = "Above" if dist_pct >= 0 else "Below"
                hits.append({
                    "Timeframe": timeframe_label,
                    "Type": category,
                    "Symbol": display_name,
                    "Close": round(last_close, 2),
                    "SMA_Period": f"{period} SMA",
                    "SMA_Value": round(sma_val, 2),
                    "Dist_%": round(dist_pct, 2),
                    "Position": position,
                    "RSI(14)": rsi_val,
                })

    result_df = pd.DataFrame(hits)
    if not result_df.empty:
        result_df["Abs_Dist"] = result_df["Dist_%"].abs()
        result_df = (
            result_df.sort_values(by=["Type", "Abs_Dist"])
            .drop(columns=["Abs_Dist"])
            .reset_index(drop=True)
        )
    return result_df


def main():
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    all_tickers = list(INDICES.values()) + STOCKS
    reverse_index_map = {v: k for k, v in INDICES.items()}

    print(f"\nFetching 5-year EOD data for {len(all_tickers)} symbols... Please wait.")

    # Download Daily data once (5y covers >1,200 daily candles & >250 weekly candles)
    raw_daily = yf.download(
        tickers=all_tickers,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    daily_close = extract_close_df(raw_daily, all_tickers)

    # Resample Daily Close to Weekly (Friday close) for exact weekly SMA calculation
    weekly_close = daily_close.resample("W-FRI").last()

    daily_hits = scan_timeframe(daily_close, reverse_index_map, "DAILY")
    weekly_hits = scan_timeframe(weekly_close, reverse_index_map, "WEEKLY")

    print(f"\n{'='*80}")
    print(f" EOD MOVING AVERAGE PROXIMITY SCAN (+/- {THRESHOLD_PCT}%) | Date: {today_str}")
    print(f"{'='*80}")

    print("\n--- DAILY TIMEFRAME HITS ---")
    if daily_hits.empty:
        print("No symbols within threshold on Daily timeframe.")
    else:
        print(daily_hits.to_markdown(index=False))

    print("\n--- WEEKLY TIMEFRAME HITS ---")
    if weekly_hits.empty:
        print("No symbols within threshold on Weekly timeframe.")
    else:
        print(weekly_hits.to_markdown(index=False))

    if SAVE_CSV and (not daily_hits.empty or not weekly_hits.empty):
        combined = pd.concat([daily_hits, weekly_hits], ignore_index=True)
        filename = f"EOD_SMA_Scan_{today_str}.csv"
        combined.to_csv(filename, index=False)
        print(f"\nSaved full report to: {filename}")


if __name__ == "__main__":
    main()
