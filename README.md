# Market Signals Telegram Bot

![Market Signals Bot Logo](assets/logo.png)

## 🚀 Overview

**Market Signals Telegram Bot** delivers actionable stock signals, strategy insights, and daily market alerts right to your Telegram. Compare strategies against buy & hold, track your favorite tickers, and get your edge—delivered.

Currently testing with a small user base.

---

## ✨ Features

- **Smart Signal Detection:** Mean Reversion, MA Crossover, Volatility Breakout
- **Performance Metrics:** Win Rate, Avg Return, vs Buy & Hold (BH), Max Drawdown
- **Customizable Alerts:** Set notification time, choose strategies, track any stock
- **Interactive Telegram Commands:** Add/remove stocks, view signals, tweak parameters
- **Secure:** API keys and secrets loaded from `.env` (never committed!)

---

## 🛠️ Getting Started

### 1. Clone the Repo

```sh
git clone https://github.com/nijordia/market-signals-telegram-bot.git
cd market-signals-telegram-bot
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Add Your Secrets

Create a `.env` file in the project root:

```
TELEGRAM_API_KEY=your-telegram-bot-api-key
TELEGRAM_CHAT_ID=your-telegram-chat-id
```

> **Note:** `.env` is already in `.gitignore` for your safety.

### 4. Run the Bot

```sh
python main.py
```

Or, for daily notifications:

```sh
python main.py --daily-run
```

---

## 📱 Telegram Commands

- `/start` – Welcome & setup
- `/add SYMBOL` – Track a stock (e.g., `/add AAPL`)
- `/remove SYMBOL` – Untrack a stock
- `/list` – Show tracked stocks
- `/signals` – Get current signals & metrics
- `/settings` – Choose strategies
- `/params` – View/edit signal parameters
- `/metrics` – Explanation of all metrics

---

## 📊 Metrics Explained

- **Win Rate:** % of signals that were profitable
- **Avg Return:** Average % return per signal
- **vs BH:** Outperformance vs Buy & Hold strategy
- **Max Drawdown:** Largest drop from peak
- **Signal Count:** Number of signals generated
- **Period:** Timeframe analyzed

---

## 🔒 Security

- **Never commit your `.env` file or API keys.**
- All secrets are loaded securely from environment variables.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests and issues are welcome! Please open an issue to discuss major changes.

---

## 📬 Contact

Wanna join the development base? DM me on instagram at @aquilare or email at nicolasjordi.aguilar@gmail.com

---

*Happy trading!*