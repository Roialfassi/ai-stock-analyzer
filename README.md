# AI-Powered Stock Analysis Desktop Application

A sophisticated desktop application for stock market analysis that leverages LLM capabilities to filter stocks, perform fundamental and technical analysis, and provide investment insights through a modern, professional interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This application combines the power of Large Language Models (LLMs) with real-time market data to create an intelligent investment assistant. It features natural language stock screening, comprehensive analysis using chain prompting techniques, and a professional trading terminal interface.

## Features

### 🔍 Natural Language Stock Screening
- Convert queries like "find undervalued tech stocks with strong growth" into market filters
- Support for complex boolean logic and relative terms
- Preset screening templates for common strategies

### 📊 Comprehensive Stock Analysis
- **Fundamental Analysis**: Financial statements, ratios, peer comparison
- **Technical Analysis**: Chart patterns, indicators, trend analysis
- **Sentiment Analysis**: News aggregation and sentiment scoring
- **AI-Powered Insights**: Multi-step LLM analysis with bull/bear cases

### 💼 Portfolio Management
- Track multiple portfolios with real-time P&L
- Risk metrics (Beta, Sharpe ratio, VaR)
- Dividend tracking and forecasting
- Tax implications calculator
- Rebalancing suggestions

### 📈 Professional UI
- Bloomberg Terminal-inspired dark theme
- Interactive candlestick charts with technical indicators
- Real-time market data updates
- Customizable workspace layouts

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows, macOS, or Linux
- No external API keys required for market data (uses yfinance)
- LLM API key (choose one):
  - OpenAI API key for GPT-4
  - Anthropic API key for Claude
  - Google API key for Gemini
  - HuggingFace token
  - OR run locally with LM Studio (no API key needed)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Roialfassi/ai-stock-analyzer.git
cd ai-stock-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
# Normal startup with configuration dialog
python main_window.py

# Or use the CLI for preset-based startup
python cli.py --preset "Quick Start - OpenAI"
```

## Configuration

### First-Time Setup

When you first run the application, you'll see a startup configuration dialog where you can:

1. **Choose your LLM provider**:
   - OpenAI (GPT-4)
   - Anthropic (Claude)
   - Google Gemini
   - HuggingFace
   - LM Studio (local)

2. **Enter your API credentials**
3. **Save as a preset** for quick access later
4. **Test the connection** before starting

### Command Line Usage

The application includes a CLI for automation and quick startup:

```bash
# List all saved presets
python cli.py --list

# Run with a specific preset
python cli.py --preset "My GPT-4 Config"

# Skip market data download for faster startup
python cli.py --preset "Quick Start" --skip-market-data

# Run in demo mode with mock data
python cli.py --preset "Local LLM" --demo

# Create a new preset from command line
python cli.py --create "Production" --provider openai --api-key "sk-..." --model gpt-4
```

### LLM Provider Setup

#### OpenAI
1. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Choose model: `gpt-4`, `gpt-4-turbo-preview`, or `gpt-3.5-turbo`
3. Enter in startup dialog or use CLI

#### Anthropic
1. Get API key from [Anthropic Console](https://console.anthropic.com/)
2. Models: `claude-3-opus` or `claude-3-sonnet`
3. Enter in startup dialog

#### Google Gemini
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Uses Gemini Pro model
3. Fast and cost-effective

#### HuggingFace
1. Get token from [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Choose from models like Mixtral, Llama 2, Falcon
3. Great for open-source models

#### LM Studio (Local)
1. Download [LM Studio](https://lmstudio.ai/)
2. Load any GGUF model
3. Start server (default: http://localhost:1234)
4. No API key required!

### Market Data

**All market data is fetched from Yahoo Finance (yfinance) - no additional API keys required!**

The application fetches:
- Real-time stock prices
- Historical data
- Financial statements
- Technical indicators
- Market indices
- Sector performance

## Features

### 🔍 Enhanced Natural Language Screening

The screener now works efficiently with yfinance data and supports:

- **Sector-based queries**: "Show me all healthcare stocks" 
- **Financial metrics**: "Stocks with P/E under 20 and dividend yield over 3%"
- **Technical patterns**: "Find stocks with bullish momentum"
- **Market cap filters**: "Large cap tech stocks" or "Small cap growth stocks"
- **Advanced filters**: "Oversold stocks near 52-week lows"

The screener intelligently optimizes which stocks to fetch based on your query, making it fast and efficient.

## Usage

### Quick Start

1. **Analyze a Stock**: 
   - Enter a symbol in the toolbar (e.g., "AAPL")
   - Click "Analyze" for comprehensive AI analysis

2. **Natural Language Screening**:
   - Type a query like "tech stocks under $50 with growing earnings"
   - Click "Screen" to find matching stocks

3. **Portfolio Management**:
   - Go to **File → New Portfolio** to create a portfolio
   - Add positions and track performance in real-time

### Example Queries

The natural language screener understands queries like:

- "Show me dividend aristocrats with yields over 3%"
- "Find small-cap growth stocks in healthcare"
- "Which tech stocks have the best profit margins?"
- "Undervalued stocks with low debt and growing revenue"
- "REITs with safe dividends"

### Analysis Types

1. **Comprehensive Analysis**: Full AI-powered analysis including all aspects
2. **Fundamental Only**: Focus on financial metrics and valuation
3. **Technical Only**: Chart patterns and technical indicators
4. **Sentiment Only**: News and market sentiment analysis

## Architecture

The application is built with a modular architecture:

```
├── models.py           # Data models and structures
├── market_data.py      # Market data integration (yfinance)
├── llm_analyzer.py     # LLM analysis engine
├── ui_components.py    # Custom UI widgets
├── screener.py         # Natural language screening
├── portfolio.py        # Portfolio management
└── main_window.py      # Main application window
```

### Key Technologies

- **PyQt6**: Modern Qt binding for Python
- **yfinance**: Yahoo Finance API wrapper
- **OpenAI/Anthropic APIs**: LLM integration
- **SQLite**: Local caching database
- **pandas/numpy**: Data analysis
- **asyncio**: Asynchronous operations

## Advanced Features

### Chain Prompting System

The application uses sophisticated chain prompting for analysis:

1. **Stock Screening Chain**: Parse query → Extract criteria → Apply filters → Rank results
2. **Fundamental Analysis Chain**: Analyze financials → Compare peers → Evaluate growth → Generate thesis
3. **Technical Analysis Chain**: Identify patterns → Analyze indicators → Determine entry/exit points

### Real-time Capabilities

- WebSocket connections for live price updates
- Configurable update intervals
- Market hours awareness
- Automatic data refresh

### Risk Management

- Portfolio risk metrics (VaR, CVaR, Maximum Drawdown)
- Correlation analysis between holdings
- Diversification recommendations
- Alert system for price movements

## Keyboard Shortcuts

- `Ctrl+A`: Analyze stock
- `Ctrl+S`: Screen stocks
- `Ctrl+,`: Open settings
- `Ctrl+Q`: Quit application
- `Ctrl+Tab`: Switch between tabs

## Data Sources

- **Real-time Quotes**: Yahoo Finance (yfinance)
- **Financial Statements**: Yahoo Finance
- **News**: Yahoo Finance News API
- **Analysis**: OpenAI GPT-4 / Anthropic Claude

## Performance

- Caching system reduces API calls
- Async operations prevent UI blocking
- Efficient data structures for large portfolios
- Optimized chart rendering

## Troubleshooting

### Common Issues

1. **"No LLM provider configured"**
   - Go to Settings and add your API keys

2. **"Could not fetch data for symbol"**
   - Check internet connection
   - Verify the symbol is valid
   - Check if markets are open

3. **Slow analysis**
   - LLM analysis can take 5-30 seconds
   - Check your internet speed
   - Consider using cached data

### Debug Mode

Run with debug logging:
```bash
python main_window.py --debug
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational and informational purposes only. It is not financial advice. Always do your own research and consult with qualified financial advisors before making investment decisions.

## Acknowledgments

- yfinance for market data access
- OpenAI and Anthropic for LLM capabilities
- PyQt6 for the excellent GUI framework
- The open-source community

## Support

For issues and feature requests, please use the GitHub issue tracker.

For questions and discussions, join our Discord community: [discord.gg/stockanalysis](https://discord.gg/stockanalysis)

---

**Note**: This application requires valid API keys for full functionality. Free tiers are available for most services, but may have usage limitations.
