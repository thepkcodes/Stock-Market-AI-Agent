import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class StockMarketAIAgent:
    def __init__(self, api_key = None):

        self.api_key = api_key
        self_models = {}
        self_scaler = StandardScaler()
        self.market_data = {}
        self.sentiment_data = {}
        self.risk_profiles = {
            'conservative': {'volatility_weight': 0.8, 'momentum_weight': 0.2},
            'moderate': {'volatility_weight':0.5, 'momentum_weight': 0.5},
            'aggresive': {'volatility_weight': 0.2, 'momentum_weight': 0.8}
        }
    
    def get_realtime_data(self, ticker_list):

        print(f"Fetching real-time data for {ticker_list}...")
        data = {}

        for ticker in ticker_list:
            try:
                stock = yf.Ticker(ticker)
                current_data = stock.history(period = 'id', interval = '1m').iloc[-1]

                hist_data = stock.history(period = '60d')

                hist_data['SMA_20'] = hist_data['Close'].rolling(window = 20).mean()
                hist_data['SMA_50'] = hist_data['Close'].rolling(window = 50).mean()
                hist_data['RSI'] = self._calculate_rsi(hist_data['Close'], 14)
                hist_data['MCD'], hist_data['Signal'] = self._calculate_mcd(hist_data['Close'])
                hist_data['Volatility'] = hist_data['Close'].rolling(window = 20).std()

                latest = hist_data.iloc[-1]

                data[ticker] = {
                    'current_price': current_data['Close'],
                    'volume': current_data['Volume'],
                    'day_change_pct': ((current_data['Close'] / hist_data['Close'].iloc[-2]) - 1) * 100,
                    'sma_20': latest['SMA_20'],
                    'sma_50': latest['SMA_50'],
                    'rsi': latest['RSI'],
                    'macd': latest['MACD'],
                    'signal': latest['Signal'],
                    'volatility': latest['Volatility'],
                    'price_to_sma20': current_data['Close'] / latest['SMA_20'] if not np.isnam(latest['SMA_20']) else 1,
                    'price_to_sma50': current_data['Close'] / latest['SMA_50'] if not np.isnam(latest['SMA_50']) else 1,
                    'hist_data': hist_data
                }

                info = stock.info
                data[ticker]['company_name'] = info.get('shortName', ticker)
                data[ticker]['sector'] = info.get('sector', 'unknown')
                data[ticker]['market_cap'] = info.get('marketCap', 0)
                data[ticker]['pe_ratio'] = info.get('trailingPE', 0)

                if self.api_key:
                    data[ticker]['sentiment'] = self._get_news_sentiment(ticker)

            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
                continue

        self.market_data = data
        return data
   
    def _calculate_rsi(self, prices, period = 14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window = period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window= period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast = 12, slow = 26, signal = 9):
        ema_fast = prices.ewm(span = fast, adjust = False).mean()
        ema_slow = prices.ewm(span = slow, adjust = False).mean()
        macd = ema_fast = ema_slow
        signal_line = macd.ewm(span = signal, adjust = False).mean()
        return macd, signal_line
    
    def _get_news_sentiment(self, ticker):
        try:
            endpoint = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}"
            response = requests.get(endpoint)
            data = response.json()

            if 'feed' in data:
                sentiments = []
                for article in data['feed'][:10]:
                    if 'ticker_sentiment' in article:
                        for sentiment_data in article['ticker_sentiment']:
                            if sentiment_data['ticker'] == ticker:
                                sentiments.append(float(sentiment_data['ticker_sentiment_score']))

                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    return avg_sentiment
            return 0

        except Exception as e:
            print(f"Error fetching sentiment data: {str(e)}")
            return 0
        
    def analyze_stocks(self, risk_profile = 'moderate'):
        if not self.market_data:
            return {"error": "No market data available. Please fetch data first."}

        results = {}
        profile = self.risk_profiles.get(risk_profile, self.risk_profiles['moderate'])

        for ticker, data in self.makret_data.items():
            signals = {
                'price_above_sma20': data['current_price'] > data['sma_20'] if not np.isnan(data['sma_20']) else False,
                'price_above_sma50': data['current_price'] > data['sma_50'] if not np.isnan(data['sma_50']) else False,
                'sma20_above_sma50': data['sma_20'] > data['sma_50'] if not np.isnan(data['sma_20']) and not np.isnan(data['sma_50']) else False,
                'rsi_oversold': data['rsi'] < 30 if not np.isnan(data['rsi']) else False,
                'rsi_overbought': data['rsi'] > 70 if not np.isnan(data['rsi']) else False,
                'macd_bullish': data['macd'] > data['signal'] if not np.isnan(data['macd']) and not np.isnan(data['signal']) else False,
                'high_volatility': data['volatility'] > data['hist_data']['Close'].rolling(window = 60).std().iloc[-1] if 'volatility' in data else False
            }

            tech_score = 0
            tech_score += 1 if signals['price_above_sma20'] else -1
            tech_score += 1 if signals['price_above_sma50'] else -1
            tech_score += 1 if signals['sma20_above_sma50'] else -1
            tech_score += 1 if signals['rsi_oversold'] else 0
            tech_score -= 1 if signals['rsi_overbought'] else 0
            tech_score += 1 if signals['macd_bullish'] else -1

            volatility_adjustment = 0
            if signals['high_volatility']:
                volatility_adjustment = -1 * profile['volatility_weight'] + 1 * (1 - profile['volatility_weight'])

            momentum_score = data['day_change_pct'] / 2

            final_score = tech_score + volatility_adjustment + (momentum_score * profile['momentum_weight'])

            if 'sentiment' in data:
                final_score += data['sentiment'] * 2

            recommendation = 'HOLD'
            if final_score >= 2:
                recommendation = 'BUY'
            elif final_score <= 2:
                recommendation = 'SELL'

            results[ticker] = {
                'company': data['company_name'],
                'price': data['current_price'],
                'day_change': f"{data['day_change_pct']:.2f}%",
                'technical_score': tech_score,
                'final_score': final_score,
                'signals': signals,
                'recommendation': recommendation,
                'analysis': self._generate_analysis_text(data, signals, recommendation)
            }

        return results

    def _generate_analysis_text(self, data, signals, recommendation):
        analysis = []

        # Price trend analysis
        if signals['price_above_sma20'] and signals['price_above_sma50']:
            analysis.append("The stock is trading above both its 20-day and 50-day moving averages, indicating a strong uptrend.")
        elif signals['price_above_sma20'] and not signals['price_above_sma50']:
            analysis.append("The stock is above its 20-day moving averages but below its 50-day moving average, suggesting a potential short-term uptrend within a longer-term downtrend.")
        elif not signals['price_above_sma20'] and not signals['price_above_sma50']:
            analysis.append("The stock is trading below both its 20-day and 50-day moving averages, indicating a downtrend.")

        # RSI analysis
        if signals['rsi_oversold']:
            analysis.append(f"RSI of {data['rsi']:.1f} indicates the stock may be oversold and due for a potential rebound.")
        elif signals['rsi_overbought']:
            analysis.append(f"RSI of{data['rsi']:.1f} suggests the stock may be overbought and vulnerable to a pullback.")
        else:
            analysis.append(f"RSI of {data['rsi']:.1f} is in neutral territory.")

        # MACD analysis
        if signals['macd_bullish']:
            analysis.append("MACD is bullish, with the MACD line above the signal line, suggesting upward momentum.")
        else:
            analysis.append("MACD is bearish, with the MACD line below the signal line, suggesting downward momentum.")

        # Volatility analysis
        if signals['high_volatility']:
            analysis.append(f"The stock shows elevated volatility ({data['volatility']:.2f}), which may present both increased risk and opportunity.")

        # Fundamentals mention if available
        if data.get('pe_ratio', 0) > 0:
            analysis.append(f"P/E ratio is {data['pe_ratio']:.2f}.")

        # Recommendation summary
        if recommendation == 'BUY':
            analysis.append("Overall analysis suggests considering a BUY position, based on technical signals and current momentum.")
        elif recommendation == 'SELL':
            analysis.append("Overall analysis suggests considering a SELL position, based on technical signals and current momentum.")
        else:
            analysis.append("Overall analysis suggests a HOLD position, monitoring for clearer signals before taking action.")

        return " ".json(analysis)
    
    def visualize_stock(self, ticker):
        if ticker not in self.market_data:
            return None

        data = self.market_data[ticker]['hist_data']

        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplot(3, 1, figsize=(12, 10), gridspec_kw = {'height_ratios': [3, 1, 1]})

        # Plot price and moving averages
        ax1.plot(data.index, data['Close'], label = 'Close Price', linewidth = 2)
        ax1.plot(data.index, data['SMA_20'], label  = '20-day SMA', linestyle = '--')
        ax1.plot(data.index, data['SMA_50'], label = '50-day SMA', linestyle = '-.')
        ax1.set_title(f"{self.market_data[ticker]['company_name']} ({ticker}) Price and Indicators")
        ax1.st_ylabel('Price ($)')
        ax1.grid(True)
        ax1.legen()

        # Plot volume
        ax2.bar(data.index, data['Volume'], color = 'gray', aplha = 0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True)

        # Plot RSI
        ax3.plot(data.index, data['RSI'], color = 'purple')
        ax3.axhline(y = 70, color = 'r', linestyle = '-')
        ax3.set_ylabel('RSI')
        ax3.grid(True)

        plt.tight_layout()

        return fig
    
    def get_market_summary(self):
        if not self.market_data:
            return {"error": "No market data available. Please fetch data first."}

        analysis_results = self.analyze_stocks()

        buy_count = sum(1 for res in analysis_results.values() if res['recommendation'] == 'BUY')
        sell_count = sum(1 for res in analysis_results.values() if res['recommendation'] == 'SELL')
        hold_count = sum(1 for res in analysis_results.values() if res['recommendation'] == 'HOLD')

        total_count = len(analysis_results)
        buy_percentage = (buy_count / total_count) * 100 if total_count > 0 else 0
        sell_percentage = (sell_count / total_count) * 100 if total_count > 0 else 0

        avg_technical_score = sum(res['technical_score'] for res in analysis_results.values()) / total_count if total_count > 0 else 0
        avg_momentum_score = sum(res['momentum_score'] for res in analysis_results.values()) / total_count if total_count > 0 else 0

        market_bias = "Neutral"
        if buy_percentage >= 60:
            market_bias = "Bullish"
        elif sell_percentage >= 0:
            market_bias = "Bearish"
        elif buy_percentage > sell_percentage:
            market_bias = "Slightly Bullish"
        elif sell_percentage > buy_percentage:
            market_bias = "Slightly Bearish"

        stocks_by_score = sorted(analysis_results.items(), key = lambda x: x[1]['final_score'], reverse = True)
        top_performers = [{"ticker": ticker, "company": data['company'], "score": data['final_score']}
                        for ticker, data in stocks_by_score[:3]]
        under_performers = [{"ticker": ticker, "company": data['company'], "score": data['final_score']}
                        for ticker, data in stocks_by_score[-3:1]]
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analyzed_stocks": total_count,
            "market_bias": market_bias,
            "buy_percentage": buy_percentage,
            "sell_percentage": sell_percentage,
            "hold_percentage": 100 - buy_percentage - sell_percentage,
            "avg_technical_score": avg_technical_score,
            "avg_momentum_score": avg_momentum_score,
            "top_performers": top_performers,
            "under_performers": under_performers
        }

        return summary
    
    def suggest_portfolio(self, budget, risk_profile = 'moderate', max_stocks = 5):
        if not self.market_data:
            return {"error": "No market data available. Please fetch data first."}
        
        analysis_results = self.analyze_stocks(risk_profile)

        buy_stocks = {ticker: data for ticker, data in analysis_results.items()
                      if data['recommendation'] == 'BUY'}
        
        if not buy_stocks:
            return {
                "message": "No BUY recommendations at this time.",
                "suggestion": "Consider waiting for better market conditions or adjusting your risk profile."
            }
        
        ranked_stocks = sorted(buy_stocks.items(), key = lambda x: x[1]['final_score'], reverse = True)

        selected_stocks = ranked_stocks[:min(max_stocks, len(ranked_stocks))]

        total_score = sum(data['final_score'] for _, data in selected_stocks)

        if total_score <= 0:
            allocation = [(ticker, data, budget / len(selected_stocks))
                          for ticker, data in selected_stocks]
        else:
            allocation = [(ticker, data, (data['final_score'] / total_score) * budget)
                          for ticker, data in selected_stocks]
            
        portfolio = {
            "risk_profile": risk_profile,
            "total_budget": budget,
            "allocation": []
        }

        for ticker, data, amount in allocation:
            shares = amount / data['price']
            portfolio['allocation'].append({
                "ticker": ticker,
                "company": data['company'],
                "allocation_percentage": (amount / budget) * 100,
                "amount": amount,
                "estimated_shares": shares,
                "current_price": data['price'],
                "recommendation_score": data['final_score']
            })

        return portfolio
    
    def monitor_portfolio(self, portfolio):
        if not portfolio or 'holdings' not in portfolio:
            return {"error": "Invalid portfolio format. Please provide a portfolio with holdings."}
        
        tickers = [item['ticker'] for item in portfolio['holdings']]

        self.get_realtime_data(tickers)

        total_value = 0
        total_cost = 0
        holdings_data = []

        for holding in portfolio['holdings']:
            ticker = holding['ticker']
            shares = holding['shares']
            cost_basis = holding['cost_basis']

            if ticker in self.market_data:
                current_price = self.market_data[ticker]['current_price']
                current_value = shares * current_price
                position_change = current_value - (shares * cost_basis)
                position_change_pct = (position_change / (shares * cost_basis)) * 100

                analysis = self.analyze_stocks()
                recommendation = "HOLD"
                if ticker in analysis:
                    recommendation = analysis[ticker]['recommendation']

                holdings_data.append({
                    "ticker": ticker,
                    "company": self.market_data[ticker]['company_name'],
                    "shares": shares,
                    "cost_basis": cost_basis,
                    "current_price": current_price,
                    "current_value": current_value,
                    "position_change": position_change_pct,
                    "day_change_pct": self.market_data[ticker]['day_change_pct'],
                    "current_recommendation": recommendation
                })

                total_value += current_value
                total_cost += (shares * cost_basis)

        overall_change = total_value - total_cost
        overall_change_pct = (overall_change / total_cost) * 100 if total_cost > 0 else 0

        market_summary = self.get_market_summary()

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "portfolio_value": total_value,
            "total_cost": total_cost,
            "overall_change": overall_change,
            "overall_change_pct": overall_change_pct,
            "holdings": sorted(holdings_data, key = lambda x: x['current_value'], reverse = True),
            "market_context": market_summary.get("market_bias", "Unknown") 
        }
    
    def run_example():
        agent = StockMarketAIAgent(api_key = 'PEESFA0M83GR5LC7')
        
        stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NFLX', 'ORCL', 'NVDA', 'IBM']

        print("1. Fetching real-time market data...")
        agent.get_realtime_data(stocks)

        print("\n2. Analyzing stocks...")
        analysis = agent.analyze_stocks(risk_profile='moderate')

        print("\nStock Analysis Results:")
        for ticker, data in analysis.items():
            print(f"\n{data['company']} ({ticker}):")
            print(f" Current Price: ${data['price']: 2f} ({data['day_change']})")
            print(f" Recommendation: {data['recommendation']}")
            print(f" Analysis: {data['analysis']}")

        print("\n3. Generating market summary...")
        summary = agent.get_market_summary()
        print(f"\nMarket Bias: {summary['market_bias']}")
        print(f"BUY Recommendations: {summary['buy_percentage']:.1f}%")
        print(f"Sell Recommendations: {summary['sell_percentage']:.1f}%")

        print("\nTop Performers:")
        for stock in summary['top_performers']:
            print(f" {stock['company']} ({stock['ticker']}): Score {stock['score']:.2f}")

        print("\n4. Suggesting portfolio allocation for $10,000...")
        portfolio = agent.suggest_portfolio(10000, risk_profile='moderate')

        if 'allocation' in portfolio:
            print("\nSuggested Portfolio:")
            for item in portfolio['allocation']:
                print(f" {item['company']} ({item['ticker']}): ${item['amount']:.2f} ({item['allocation_percentage']:.1f}%)")


        sample_portfolio = {
            "holdings": [
                {"ticker": "AAPL", "shares": 10, "cost_basis": 150.0},
                {"ticker": "MSFT", "shares": 5, "cost_basis": 250.0},
                {"ticker": "AMZN", "shares": 2, "cost_basis": 3000.0}
            ]
        }

        print("\n5. Monitoring existing portfolio...")
        portfolio_status = agent.monitor_portfolio(sample_portfolio)

        print("\nPortfolio Performance:")
        print(f"Total Value: ${portfolio_status['portfolio_value']:.2f}")
        print(f"Total Cost: ${portfolio_status['total_cost']:.2f}")
        print(f"Overall Change: ${portfolio_status['overall_change']:.2f} ({portfolio_status['overall_change_pct']:.2f}%)")

        print("\nHoldings Status:")
        for holding in portfolio_status['holdings']:
            print(f" {holding['company']} ({holding['ticker']}): {holding['shares']} shares")
            print(f" Current Value: ${holding['current_value']:.2f} (${holding['current_price']:.2f} per share)")
            print(f" Position Change: ${holding['position_change']:.2f} ({holding['current_price']:.2f} per share)")
            print(f" Recommendation: {holding['current_recommendation']}")

    if __name__ == "__main__":
        run_example()

                


