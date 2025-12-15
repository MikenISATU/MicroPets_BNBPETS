# MicroPets Buy Tracker Bot

A Telegram bot that monitors and alerts on MicroPets ($PETS) token buys on BNB Chain (BSC).

## Features

- Real-time monitoring of $PETS token buys on BNB Chain
- Automatic notifications to Telegram with buy details
- Video alerts for different buy sizes
- Manual transaction fetching with `/fetch` command
- Batch processing with `/fetchbatch` command
- Auto-detection of transaction hashes in messages
- Admin-only commands for security

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required environment variables:
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from @BotFather
- `ADMIN_USER_ID`: Your Telegram chat ID (admin user)
- `TELEGRAM_CHAT_ID`: Main Telegram chat/channel ID for alerts
- `CLOUDINARY_CLOUD_NAME`: Cloudinary cloud name for video hosting
- `APP_URL`: Your application URL for webhooks (or use Railway's auto domain)
- `BNB_RPC_URL`: BSC RPC endpoint (Ankr recommended)

### 3. Run the Bot

```bash
python main.py
```

The bot will start on port 8080 by default (configurable via `PORT` env var).

## Commands

### Monitoring Commands
- `/start` - Start the bot
- `/track` - Enable buy tracking and alerts
- `/stop` - Disable buy tracking
- `/status` - Check tracking status

### Manual Fetch Commands
- `/fetch <tx_hash_or_url>` - Process a specific transaction
- `/fetchbatch <tx1> <tx2> ...` - Process multiple transactions
- Auto-detect: Just paste a transaction hash or BscScan link!

### Statistics
- `/stats` - View all buys from last 2 weeks (heavy operation)

### Testing
- `/test` - Test buy alert with video
- `/noV` - Test buy alert without video

### Debug
- `/debug` - Show debug information
- `/help` - Show help message

## How It Works

1. Connects to BNB Chain via Web3
2. Monitors Transfer events from the target LP address
3. Filters transactions that meet minimum criteria ($50+ USD value)
4. Fetches current prices from GeckoTerminal/CoinMarketCap
5. Sends formatted alerts to Telegram with video

## Architecture

- **FastAPI**: Web server for webhooks and health checks
- **python-telegram-bot**: Telegram bot framework
- **Web3.py**: BSC blockchain interaction
- **Tenacity**: Retry logic for API calls
- **Cloudinary**: Video hosting for buy alerts

## Deployment

This bot is designed to run on Railway or similar platforms:

1. Fork this repository
2. Connect to Railway
3. Add environment variables
4. Deploy

Railway will automatically set `RAILWAY_PUBLIC_DOMAIN` for webhooks.

## Notes

- Admin commands are restricted to `ADMIN_USER_ID` only
- BSC has a 5000 block limit per eth_getLogs call
- The bot uses windowed scanning (1000 blocks) for reliability
- Rate limiting is implemented for API calls to avoid throttling
- Transactions are cached to prevent duplicate alerts

## Troubleshooting

If monitoring stops working:
1. Check `/debug` for current status
2. Verify `is_tracking_enabled` is true
3. Check recent errors in debug output
4. Verify BNB RPC connection with `/status`
5. Check logs for Web3 connection issues

## License

MIT
