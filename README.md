# 🤖 STABLE BOLLINGER BOT

A sophisticated, fully automated cryptocurrency trading bot built for **BitMEX Testnet** that implements advanced Bollinger Band strategies with dynamic order management and robust risk controls.

## 🎯 What This Bot Achieves

### **Core Trading Strategy**
- **Bollinger Band-Based Signals**: Uses 5-period SMA with dynamic standard deviation calculations
- **Multi-Level Entry Orders**: Places up to 5 laddered entry orders at different Bollinger band levels (-3σ, -2.5σ, -2σ, -1.5σ, -1σ for long / +1σ, +1.5σ, +2σ, +2.5σ, +3σ for short)
- **Dynamic Take-Profit Management**: Automatically places TP orders on the opposite side immediately after position creation
- **Intelligent Signal Detection**: Long/Short/Flat signal generation based on price relationship to Bollinger bands

### **Advanced Risk Management**
- **Position Size Limit**: Hard cap at 2,500 USD to prevent overexposure
- **Dynamic Position Sizing**: Each TP order sized at 1/5 of total position
- **Real-time Order Updates**: Continuously adjusts entry and TP orders based on changing Bollinger band levels
- **Minimum Order Size Compliance**: Respects BitMEX's 100-contract minimum order requirement

### **Robust Technical Implementation**
- **Persistent Execution**: Runs continuously using `nohup` - survives SSH disconnections
- **State Management**: Saves and loads trading state across restarts (`state.json`)
- **Error Handling**: Comprehensive exception handling with automatic retries
- **Rate Limit Compliance**: Built-in delays and API call optimization
- **Real-time Logging**: Detailed logging of all trading activities (`bot.log`)

### **Smart Order Management**
- **Dynamic Band Tracking**: Orders automatically updated when Bollinger bands shift
- **Stale Order Cleanup**: Cancels orders "too far from market" and replaces with current levels
- **Immediate TP Placement**: Take-profit orders placed within seconds of position changes
- **Position-Aware Trading**: Adjusts strategy based on current position size and direction

### **Market Condition Adaptability**
- **Volatile Markets**: Executes multiple entry levels for optimal position building
- **Flat Markets**: Continues operation even in zero-volatility conditions
- **Trend Following**: Adapts to both trending and ranging market conditions
- **Signal Confirmation**: Prevents false signals with confirmation logic

### **Production-Ready Features**
- **Virtual Environment**: Isolated Python dependencies for stability
- **Configuration Management**: Externalized settings in `config.json`
- **Time Synchronization**: Automatic server time offset handling
- **Connection Resilience**: Automatic reconnection and retry mechanisms
- **Memory Efficiency**: Optimized for long-running operation

## 📊 Trading Performance Highlights

- **Zero Manual Intervention**: Fully autonomous operation 24/7
- **Risk-Controlled**: Never exceeds predefined position limits
- **Responsive**: 2-second cycle time for rapid market adaptation
- **Comprehensive**: Handles entry, position management, and exit automatically
- **Testnet Proven**: Extensively tested on BitMEX Testnet environment

## 🛠️ Technical Stack

- **Language**: Python 3.x
- **Exchange API**: CCXT (BitMEX Testnet)
- **Data Analysis**: Pandas for Bollinger Band calculations
- **Configuration**: JSON-based settings
- **Logging**: Comprehensive activity logging
- **State Persistence**: JSON-based state management
- **Environment**: Virtual environment with isolated dependencies

## 🚀 Key Achievements

1. **Solved Complex Order Management**: Dynamic updating of multiple order levels based on real-time Bollinger band changes
2. **Implemented Robust Risk Controls**: Position size limits with real-time monitoring
3. **Built Fault-Tolerant System**: Handles API failures, network issues, and exchange quirks
4. **Created Intelligent Signal System**: Advanced Bollinger Band interpretation for market direction
5. **Achieved Production Stability**: Runs continuously without manual intervention
6. **Developed Smart TP Strategy**: Immediate and dynamic take-profit order management
7. **Optimized for Performance**: Efficient API usage and minimal latency
8. **Built Comprehensive Monitoring**: Detailed logging for performance analysis and debugging

## 💡 Innovation Highlights

- **Real-time Band Adaptation**: Unlike static strategies, this bot continuously adapts to changing market volatility
- **Multi-Layer Entry Strategy**: Sophisticated laddered entry system for optimal position building
- **Intelligent Order Lifecycle**: Orders are not just placed but actively managed throughout their lifecycle
- **State-Aware Decision Making**: All decisions consider current position, open orders, and market conditions
- **Testnet Optimization**: Specifically designed and tested for BitMEX Testnet environment

## 📋 Requirements

- Python 3.7+
- BitMEX Testnet API credentials
- Linux/Unix environment (tested on Ubuntu)
- Stable internet connection

## 🔧 Installation & Setup

1. Clone this repository
2. Set up virtual environment: `python3 -m venv venv`
3. Activate environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Configure API credentials in `.env` file
6. Run with nohup: `nohup python3 -u "BOLLINGER BOT.py" > bot.log 2>&1 &`

## 📈 This Bot Demonstrates

- **Advanced Algorithmic Trading**: Implementation of sophisticated trading strategies
- **Production-Grade Software**: Robust, fault-tolerant, and maintainable code
- **Financial Risk Management**: Proper position sizing and risk controls
- **Real-time System Design**: Low-latency, responsive trading system
- **API Integration Expertise**: Efficient and reliable exchange API usage

---

⚠️ **Disclaimer**: This bot is designed for BitMEX Testnet only. Always test thoroughly before considering any live trading implementation.

🎯 **Educational Purpose**: This project demonstrates advanced algorithmic trading concepts and should be used for learning and testing purposes only. 