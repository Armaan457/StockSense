import streamlit as st
from price_predictor import run_stock_prediction
import similar_stocks
from scipy.stats import rankdata
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np

st.set_page_config(page_title="Stock Dashboard", layout="wide")


st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["📊 Stock Prediction", "🤝 Similar Stocks"])


@st.cache_resource(ttl=86400)
def get_feature_vectors():
    return similar_stocks.prepare_data(similar_stocks.STOCK_UNIVERSE)


if page == "📊 Stock Prediction":
    st.title("📊 Stock Price Predictor")

    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOG):", value="GOOG")

    if st.button("Predict"):
        with st.spinner("Analysing..."):
            try:
                result = run_stock_prediction(stock_symbol.upper())
                st.success(f"Prediction for {result['symbol'].upper()}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Previous Close", f"${result['previous_price']:.2f}")
                with col2:
                    st.metric("📈 Predicted Close", f"${result['predicted_price']:.2f}", delta=f"{result['predicted_price'] - result['previous_price']:.2f}")

                st.write("### Price Comparison")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=["Previous", "Predicted"],
                    y=[result["previous_price"], result["predicted_price"]],
                    mode="lines+markers",
                    name="Price Trend"
                ))
                fig.update_layout(title="Previous vs Predicted Prices", xaxis_title="Type", yaxis_title="Price ($)")
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("🧠 Model Evaluation")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(label="MAPE", value=f"{result['mape']:.4f}")
                with col2:
                    st.metric(label="R² Score", value=f"{result['r2']:.4f}")
                with col3:
                    st.metric(label="NRMSE", value=f"{result['nrmse']:.4f}")
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "🤝 Similar Stocks":
    st.title("🤝 Similar Stock Recommender")

    selected_ticker = st.selectbox("Select a Stock:", similar_stocks.STOCK_UNIVERSE)
    selected_sectors = st.multiselect(
        "Filter by Sectors (Optional):",
        sorted(list(set(similar_stocks.STOCK_SECTORS.values())))
    )

    if st.button("Get Recommendations"):
        with st.spinner("Processing..."):
            feature_vectors = get_feature_vectors()

        if feature_vectors:
            if selected_sectors:
                tickers_to_consider = [
                    ticker for ticker in similar_stocks.STOCK_UNIVERSE
                    if similar_stocks.STOCK_SECTORS.get(ticker) in selected_sectors or ticker == selected_ticker
                ]
            else:
                tickers_to_consider = similar_stocks.STOCK_UNIVERSE

            recommendations = similar_stocks.knn_recommend(selected_ticker, feature_vectors, tickers_to_consider)

            if recommendations:
                st.subheader(f"Top 5 Recommendations for {selected_ticker}:")

                recommendations.sort(key=lambda item: item[1])
                ranks = rankdata([distance for _, distance in recommendations], method='ordinal')

                for i in range(min(5, len(recommendations))):
                    ticker, dist = recommendations[i]
                    rank = int(ranks[i]) 
                    st.write(f"- {ticker} (Rank: {rank}) - Sector: {similar_stocks.STOCK_SECTORS.get(ticker, 'N/A')}")

                if len(recommendations) < 5:
                    st.info(f"Only {len(recommendations)} recommendations found.")
                    for j in range(len(recommendations), 5):
                        st.write("- N/A")

                # Visualize in 3D using PCA
                tickers = [selected_ticker] + [ticker for ticker, _ in recommendations]
                vectors = np.array([feature_vectors[ticker] for ticker in tickers])

                # Reduce to 3D
                pca = PCA(n_components=3)
                reduced_vectors = pca.fit_transform(vectors)

                fig = go.Figure()

                # Selected Stock
                fig.add_trace(go.Scatter3d(
                    x=[reduced_vectors[0][0]],
                    y=[reduced_vectors[0][1]],
                    z=[reduced_vectors[0][2]],
                    mode='markers+text',
                    marker=dict(size=10, color='red'),
                    name=selected_ticker,
                    text=[selected_ticker],
                    textposition='top center'
                ))

                for i in range(1, len(reduced_vectors)):
                    fig.add_trace(go.Scatter3d(
                        x=[reduced_vectors[i][0]],
                        y=[reduced_vectors[i][1]],
                        z=[reduced_vectors[i][2]],
                        mode='markers+text',
                        marker=dict(size=6, color='skyblue'),
                        name=tickers[i],
                        text=[tickers[i]],
                        textposition='top center'
                    ))

                fig.update_layout(
                    title='3D Visualization of Similar Stocks',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'
                    ),
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No recommendations found.")
        else:
            st.error("Failed to prepare data. See logs for more details.")
