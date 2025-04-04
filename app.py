import streamlit as st
import similar_stocks
from scipy.stats import rankdata
import similar_stocks

st.title("Stock Analysis Tool")

selected_ticker = st.selectbox("Select a Stock:", similar_stocks.STOCK_UNIVERSE)
selected_sectors = st.multiselect("Select Sectors (Optional):", sorted(list(set(similar_stocks.STOCK_SECTORS.values()))))


if st.button("Get Recommendations"):
    with st.spinner("Fetching and processing data..."):
        feature_vectors = similar_stocks.prepare_data(similar_stocks.STOCK_UNIVERSE)

    if feature_vectors:
        if selected_sectors:
            tickers_to_consider = [ticker for ticker in similar_stocks.STOCK_UNIVERSE 
                                  if similar_stocks.STOCK_SECTORS.get(ticker) in selected_sectors 
                                  or ticker == selected_ticker]
        else:
            tickers_to_consider = similar_stocks.STOCK_UNIVERSE

        recommendations = similar_stocks.knn_recommend(selected_ticker, feature_vectors, tickers_to_consider)

        if recommendations:
            st.subheader(f"Top 5 Recommendations for {selected_ticker}:")

            recommendations.sort(key=lambda item: item[1])
            ranks = rankdata([distance for ticker, distance in recommendations], method='ordinal')

            for i in range(min(5, len(recommendations))):
                  ticker, _ = recommendations[i]
                  rank = int(ranks[i]) 
                  st.write(f"- {ticker} (Rank: {rank}) - Sector: {similar_stocks.STOCK_SECTORS.get(ticker, 'N/A')}")

            if len(recommendations) < 5:
                st.info(f"Only {len(recommendations)} recommendations found.")
                for j in range(len(recommendations), 5):
                    st.write("- N/A")
        else:
            st.warning("No recommendations found.")
    else:
        st.error("Failed to prepare data.  See error messages above.")