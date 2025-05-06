import streamlit as st
import json
from datetime import datetime, date

# Cache the data loading to improve performance when re-running the app
@st.cache_data
def load_data(path='final_summaries.jsonl'):
    with open(path) as f:
        # Load each JSON line into a Python dictionary
        return [json.loads(l) for l in f]

# ───────────── Helper Function ─────────────
def tickers(raw):
    """
    Extract a list of ticker symbols from the 'stock' field.
    Handles strings, lists and multiple delimiters.
    """
    if raw is None:
        return ["UNKNOWN"]
    if isinstance(raw, list):
        return [t.strip().upper() for t in raw if t]
    raw = str(raw).upper()
    for sep in [",", ";", "/", "|"]:
        if sep in raw:
            return [t.strip() for t in raw.split(sep) if t.strip()]
    return [raw.strip()]

# ───────────── Main App ─────────────
def main():
    st.title("Financial News Summaries")

    # Load article data from file
    articles = load_data()

    # ───────────── Sidebar Filters ─────────────
    # Create a sorted set of all tickers found in the data
    all_tickers = sorted({t for a in articles for t in tickers(a.get("stock"))})
    # Let user filter by tickers
    sel = st.sidebar.multiselect("Stocks (leave empty = all)", options=all_tickers)

    # Determine the earliest available date in the dataset
    min_date = min(
        datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S").date()
        for a in articles if "timestamp" in a
    )
    # Let user choose start date
    start_date = st.sidebar.date_input("From date", value=min_date)

    # ───────────── Filter Articles ─────────────
    def keep(a):
        ts = datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S").date()
        # Check if article matches selected tickers and date
        ok_stock = (not sel) or any(t in sel for t in tickers(a.get("stock")))
        return ok_stock and ts >= start_date

    # Apply filters and limit to 50 most recent
    shown = sorted(filter(keep, articles),
                   key=lambda a: a["timestamp"], reverse=True)[:50]

    # If no articles match the filters
    if not shown:
        st.warning("No summaries found. Adjust the filters.")
        return

    # ───────────── Display Results ─────────────
    for art in shown:
        header = f"{', '.join(tickers(art.get('stock')))} – {art['timestamp']}"
        with st.expander(header):
            st.subheader("Original Article Text")
            st.write(art["text"])

            st.subheader("Baseline Summary")
            st.write(art["baseline_summary"])

            st.subheader("Fine‑tuned Summary")
            st.write(art["ft_summary"])

            # Link to the original article if available
            if art.get("source_url") and art["source_url"] != "N/A":
                st.markdown(f"[Original Article]({art['source_url']})", unsafe_allow_html=True)
            else:
                st.caption("No URL available")

# Run the Streamlit app
if __name__ == "__main__":
    main()