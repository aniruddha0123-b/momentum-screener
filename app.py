import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import warnings
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Optional: NSEpy as a fallback
try:
    from nsepy import get_history
    NSEPY_AVAILABLE = True
except ImportError:
    NSEPY_AVAILABLE = False

MODEL_PATH = "momentum_model.pkl"
SCALER_PATH = "momentum_scaler.pkl"

class MomentumStockScreener:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.indian_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'ASIANPAINT.NS'
        ]
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                print("✅ Loaded pre-trained model from disk")
            except Exception as e:
                print(f"⚠️ Could not load saved model: {e}")

    def fetch_stock_data(self, symbol, period='6mo'):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if len(data) >= 50:
                return data
            else:
                raise ValueError("Insufficient data from Yahoo")
        except Exception as e:
            print(f"⚠️ Yahoo Finance failed for {symbol}: {e}")
            if NSEPY_AVAILABLE and symbol.endswith(".NS"):
                try:
                    from datetime import date, timedelta
                    end = date.today()
                    start = end - timedelta(days=180)
                    data = get_history(symbol=symbol.replace(".NS", ""), start=start, end=end)
                    if not data.empty:
                        return data.rename(columns={'Close': 'Close Price', 'Volume': 'Deliverable Volume'})
                except Exception as e:
                    print(f"⚠️ NSEpy fallback failed for {symbol}: {e}")
            cache_file = f"cache_{symbol}.csv"
            if os.path.exists(cache_file):
                print(f"📂 Using cached data for {symbol}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return None

    def calculate_technical_indicators(self, data):
        indicators = {}
        high, low, close, volume = data['High'].values, data['Low'].values, data['Close'].values, data['Volume'].values
        indicators['RSI'] = talib.RSI(close, timeperiod=14)[-1]
        indicators['SMA_20'] = talib.SMA(close, timeperiod=20)[-1]
        indicators['SMA_50'] = talib.SMA(close, timeperiod=50)[-1]
        indicators['Volatility_20d'] = np.std(close[-20:]) / np.mean(close[-20:]) * 100
        indicators['Return_5d'] = (close[-1] / close[-6] - 1) * 100
        indicators['Return_10d'] = (close[-1] / close[-11] - 1) * 100
        indicators['Return_20d'] = (close[-1] / close[-21] - 1) * 100
        indicators['Volume_Ratio'] = volume[-1] / np.mean(volume[-20:])
        indicators['Price_Momentum_Score'] = (
            (indicators['Return_5d'] * 0.4) +
            (indicators['Return_10d'] * 0.35) +
            (indicators['Return_20d'] * 0.25)
        )
        return indicators

    def create_features(self, indicators):
        feature_names = ['RSI', 'SMA_20', 'SMA_50', 'Volatility_20d',
                         'Return_5d', 'Return_10d', 'Return_20d',
                         'Volume_Ratio', 'Price_Momentum_Score']
        features = [indicators.get(name, 0) for name in feature_names]
        return np.array(features), feature_names

    def prepare_training_data(self):
        X, y = [], []
        for symbol in self.indian_stocks:
            data = self.fetch_stock_data(symbol, period='1y')
            if data is None:
                continue
            for i in range(60, len(data) - 5):
                window_data = data.iloc[:i+1]
                future_data = data.iloc[i:i+6]
                indicators = self.calculate_technical_indicators(window_data)
                features, self.feature_names = self.create_features(indicators)
                close_prices = future_data['Close'].values
                current_price, future_price = close_prices[0], close_prices[-1]
                return_pct = (future_price / current_price - 1) * 100
                label = 1 if return_pct > 3 else 0
                X.append(features)
                y.append(label)
        return np.array(X), np.array(y)

    def train_model(self):
        X, y = self.prepare_training_data()
        if len(X) == 0:
            print("No training data available!")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        self.model = VotingClassifier([('rf', rf), ('lr', lr)], voting='soft')
        self.model.fit(X_train_scaled, y_train)
        print(f"Training accuracy: {self.model.score(X_train_scaled, y_train):.3f}")
        print(f"Testing accuracy: {self.model.score(X_test_scaled, y_test):.3f}")
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        print("💾 Model saved to disk")

    def calculate_momentum_score(self, symbol):
        if self.model is None:
            print("Model not trained yet!")
            return None
        data = self.fetch_stock_data(symbol, period='3mo')
        if data is None:
            return None
        indicators = self.calculate_technical_indicators(data)
        features, _ = self.create_features(indicators)
        features_scaled = self.scaler.transform([features])
        prob = self.model.predict_proba(features_scaled)[0][1]
        volatility = indicators.get('Volatility_20d', 1) / 100
        risk_adjusted = prob / (volatility if volatility > 0 else 1)
        return {
            'symbol': symbol,
            'momentum_probability': prob,
            'final_momentum_score': risk_adjusted,
            'current_price': data['Close'][-1],
            'rsi': indicators.get('RSI', 0),
            'price_momentum': indicators.get('Price_Momentum_Score', 0),
            'volatility': indicators.get('Volatility_20d', 0)
        }

# ---------------- Streamlit Wrapper -----------------

screener = MomentumStockScreener()
st.set_page_config(page_title="AI Momentum Stock Screener", layout="wide")
st.title("📈 AI Momentum Stock Screener (India)")
st.sidebar.header("Settings")

selected_symbols = st.sidebar.multiselect("Choose stocks to analyze:", screener.indian_stocks, default=screener.indian_stocks[:5])

if st.sidebar.button("Run Screening"):
    results = []
    for symbol in selected_symbols:
        stock_score = screener.calculate_momentum_score(symbol)
        if stock_score:
            results.append(stock_score)
    if results:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values("final_momentum_score", ascending=False)
        st.subheader("Top Momentum Candidates")
        st.dataframe(df_sorted[["symbol", "current_price", "momentum_probability", "final_momentum_score", "rsi", "volatility"]])
        st.subheader("Momentum Score vs RSI")
        fig, ax = plt.subplots()
        ax.scatter(df_sorted["rsi"], df_sorted["final_momentum_score"])
        for i, row in df_sorted.iterrows():
            ax.text(row["rsi"], row["final_momentum_score"], row["symbol"].replace(".NS", ""))
        ax.set_xlabel("RSI")
        ax.set_ylabel("Risk-Adjusted Momentum Score")
        st.pyplot(fig)
        st.download_button("Download Results as CSV", df_sorted.to_csv(index=False).encode('utf-8'), "momentum_results.csv", "text/csv", key='download-csv')
    else:
        st.warning("No results available. Try different stocks or retrain the model.")
else:
    st.info("Select stocks and click 'Run Screening' to analyze.")