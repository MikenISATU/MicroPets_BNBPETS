# 🚀 MicroPets BNBPETS Bot - Complete Deployment Guide

## 📋 Current Status

✅ **Bot is fully configured with Moralis API**
✅ **All code committed to branch: `claude/setup-fastapi-project-1180w`**
✅ **Ready to deploy on Railway**

---

## 🔑 Required Railway Environment Variables

Add these to your Railway service:

### **Required Variables:**
```bash
# Telegram Bot
TELEGRAM_BOT_TOKEN=YourTelegramBotToken
ADMIN_USER_ID=YourTelegramUserId
TELEGRAM_CHAT_ID=YourChannelChatId

# Cloudinary (for videos)
CLOUDINARY_CLOUD_NAME=YourCloudinaryName

# API Keys (ADD THIS!)
MORALIS_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6IjU5ZTI3ZDIxLTA0YTgtNDk2MS1hNWNjLTVmNDEyYTBjYjRiOSIsIm9yZ0lkIjoiNDQ5MjgzIiwidXNlcklkIjoiNDYyMjY4IiwidHlwZUlkIjoiZWNiZGFmYjktZjU1MS00OTAwLWE0Y2QtMjhlNzllYzhjNjBmIiwidHlwZSI6IlBST0pFQ1QiLCJpYXQiOjE3NDgyNjk4NjAsImV4cCI6NDkwNDAyOTg2MH0.vUZz_o6E-D7j0yFXwyBfMqVyU2H1NhbiXTHGipzjEqM

# Token Info
CONTRACT_ADDRESS=0x2466858ab5edAd0BB597FE9f008F568B00d25Fe3
TARGET_ADDRESS=0x4BdEcE4E422fA015336234e4fC4D39ae6dD75b01
```

### **Optional Variables:**
```bash
# If you also have BscScan API key (recommended for backup)
ETHERSCAN_API_KEY=YourBscScanKey

# Polling interval (seconds)
POLLING_INTERVAL=60

# Port (Railway sets this automatically)
PORT=8080
```

---

## 📁 File Structure

```
MicroPets_BNBPETS/
├── main.py                    # Complete bot code
├── requirements.txt           # Python dependencies
├── .env                       # Local environment (not in git)
├── posted_transactions.txt    # Transaction log
└── DEPLOYMENT_GUIDE.md        # This file
```

---

## 🔧 How It Works

### **Transaction Fetching:**
```
1. Try BscScan API (if ETHERSCAN_API_KEY is set)
   └─ Fast, reliable, 100k calls/day FREE

2. Fallback to Moralis API (if MORALIS_API_KEY is set)
   └─ 40k compute units/day FREE
   └─ Your key is configured!

3. If both fail → Clear error + retry next interval
```

### **Bot Commands:**
- `/track` - Start monitoring buy transactions
- `/stop` - Stop monitoring
- `/fetch <tx_hash>` - Manually process a transaction
- `/fetchbatch <tx1> <tx2>...` - Process multiple transactions
- `/stats` - Show buys from last 2 weeks
- `/status` - Check if tracking is enabled
- `/debug` - Show debug information
- `/test` - Send test buy notification with video
- `/help` - Show all commands

### **Auto-Detection:**
Just paste a transaction hash or BscScan link and the bot will process it automatically!

---

## 🚀 Deployment Steps

### **1. Push to GitHub** (Already done!)
```bash
git push origin claude/setup-fastapi-project-1180w
```

### **2. Configure Railway**
1. Go to Railway dashboard
2. Select your MicroPets bot service
3. Click "Variables" tab
4. Add `MORALIS_API_KEY` with the value provided above
5. Click "Deploy" or wait for auto-deploy

### **3. Verify Deployment**
Check Railway logs for:
```
============================================================
🚀 Starting MicroPets BNBPETS Tracker Bot
============================================================
✅ Bot initialized successfully
✅ Moralis API key configured: eyJhbGciOiJ...
✅ Webhook mode active - bot ready for commands
============================================================
📱 Bot Commands:
   /track - Start monitoring transactions
   /stop  - Stop monitoring
   /help  - Show all commands
============================================================
✨ Bot is now running and waiting for commands...
💡 Use /track to start transaction monitoring
============================================================
```

### **4. Start Monitoring**
Send `/track` command to your bot in Telegram

---

## 📊 Expected Behavior

### **Startup:**
```
✅ Moralis API key configured
✅ Bot initialized
✅ Webhook/Polling active
⏸️  Waiting for /track command
```

### **After /track:**
```
📡 Trying Moralis API (primary)
✅ Moralis API success!
✅ Fetched 0 buy transactions (if no new buys)
```

### **When Buy Detected:**
```
✅ Found 1 buy transaction
💰 Processing transaction 0x67bb68f...
🚀 Posting to Telegram channel
✅ Successfully posted!
```

---

## 🐛 Troubleshooting

### **Bot not responding:**
- Check Railway logs for errors
- Verify `TELEGRAM_BOT_TOKEN` is correct
- Check webhook/polling status in logs

### **No transactions detected:**
- Verify `MORALIS_API_KEY` is set correctly
- Check if there were actual buys in the time window
- Use `/debug` command to see last block scanned

### **API errors:**
```
❌ Moralis API: Rate limit exceeded
→ Wait a few minutes, Moralis will retry automatically
→ Consider adding ETHERSCAN_API_KEY as backup

❌ Moralis API: Invalid API key
→ Double-check the API key in Railway variables
→ Regenerate key at https://moralis.io if needed
```

---

## 💡 Pro Tips

1. **Add Both API Keys** for maximum reliability:
   - `ETHERSCAN_API_KEY` (BscScan) - Primary
   - `MORALIS_API_KEY` (Moralis) - Backup

2. **Monitor Railway Logs** to see what's happening

3. **Use `/debug`** to check bot status

4. **Transaction Detection:**
   - Scans every 60 seconds (configurable)
   - Only posts buys > $50 USD value
   - Filters transfers FROM LP address only

---

## 📝 Notes

- Bot uses Moralis API for transaction fetching (FREE tier)
- GeckoTerminal API for token prices (FREE)
- Videos hosted on Cloudinary
- Transaction history stored in `posted_transactions.txt`
- Bot remembers last block scanned to avoid duplicates

---

## 🎯 Current Configuration

- ✅ Moralis API integrated
- ✅ BscScan API available as backup (if you add key)
- ✅ Auto-fallback between APIs
- ✅ Polling and Webhook support
- ✅ Transaction auto-detection
- ✅ Video notifications with buy details

---

## 📞 Support

If you encounter issues:
1. Check Railway logs first
2. Verify all environment variables are set
3. Test with `/test` command
4. Check `/debug` output

---

**Ready to deploy!** 🎉

Just add the `MORALIS_API_KEY` to Railway and the bot will start working!
