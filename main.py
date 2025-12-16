import os
import logging
import requests
import random
import asyncio
import json
import time
import uuid
import re
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Set
from fastapi import FastAPI, Request, HTTPException
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from web3 import Web3
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from datetime import datetime, timedelta
from decimal import Decimal
import telegram
import threading

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
telegram_logger = logging.getLogger("telegram")
telegram_logger.setLevel(logging.WARNING)

# Check python-telegram-bot version
logger.info(f"python-telegram-bot version: {telegram.__version__}")
if not telegram.__version__.startswith('20'):
    logger.error(f"Expected python-telegram-bot v20.0+, got {telegram.__version__}")
    raise SystemExit(1)

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
APP_URL = os.getenv('RAILWAY_PUBLIC_DOMAIN', os.getenv('APP_URL'))
BSCSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY', os.getenv('BSCSCAN_API_KEY', ''))  # v2 unified API works for both
MORALIS_API_KEY = os.getenv('MORALIS_API_KEY', '')  # FREE alternative: https://moralis.io

# Default RPC → Ankr node, but can still be overridden by BNB_RPC_URL in Railway
BNB_RPC_URL = os.getenv(
    'BNB_RPC_URL',
    'https://rpc.ankr.com/bsc/de2d6cc98dc748fad0561db5815bd4a7a5d426946f5f19114ff14cfb9096e4fd'
)

CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS', '0x2466858ab5edAd0BB597FE9f008F568B00d25Fe3')
ADMIN_CHAT_ID = os.getenv('ADMIN_USER_ID')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
PORT = int(os.getenv('PORT', 8080))
COINMARKETCAP_API_KEY = os.getenv('COINMARKETCAP_API_KEY', '')
TARGET_ADDRESS = os.getenv('TARGET_ADDRESS', '0x4BdEcE4E422fA015336234e4fC4D39ae6dD75b01')
POLLING_INTERVAL = int(os.getenv('POLLING_INTERVAL', 60))

# Validate environment variables
missing_vars = []
for var, name in [
    (TELEGRAM_BOT_TOKEN, 'TELEGRAM_BOT_TOKEN'),
    (CLOUDINARY_CLOUD_NAME, 'CLOUDINARY_CLOUD_NAME'),
    (APP_URL, 'APP_URL/RAILWAY_PUBLIC_DOMAIN'),
    (BNB_RPC_URL, 'BNB_RPC_URL'),
    (CONTRACT_ADDRESS, 'CONTRACT_ADDRESS'),
    (ADMIN_CHAT_ID, 'ADMIN_USER_ID'),
    (TELEGRAM_CHAT_ID, 'TELEGRAM_CHAT_ID'),
    (TARGET_ADDRESS, 'TARGET_ADDRESS')
]:
    if not var:
        missing_vars.append(name)
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate Ethereum addresses
for addr, name in [(CONTRACT_ADDRESS, 'CONTRACT_ADDRESS'), (TARGET_ADDRESS, 'TARGET_ADDRESS')]:
    if not Web3.is_address(addr):
        logger.error(f"Invalid Ethereum address for {name}: {addr}")
        raise ValueError(f"Invalid Ethereum address for {name}: {addr}")

if not COINMARKETCAP_API_KEY:
    logger.warning("COINMARKETCAP_API_KEY is empty; CoinMarketCap API calls will be skipped")

if not BSCSCAN_API_KEY and not MORALIS_API_KEY:
    logger.error("="*60)
    logger.error("⚠️ No API keys configured!")
    logger.error("="*60)
    logger.error("🔑 Get a FREE API key from either:")
    logger.error("   Option 1 (Recommended): BscScan")
    logger.error("      → https://bscscan.com/register (FREE)")
    logger.error("      → Set: ETHERSCAN_API_KEY=YourKey")
    logger.error("      → Free tier: 5 calls/sec, 100k/day")
    logger.error("")
    logger.error("   Option 2: Moralis")
    logger.error("      → https://moralis.io (FREE)")
    logger.error("      → Set: MORALIS_API_KEY=YourKey")
    logger.error("      → Free tier: 40k compute units/day")
    logger.error("="*60)
elif BSCSCAN_API_KEY:
    logger.info(f"✅ BscScan API key configured: {BSCSCAN_API_KEY[:10]}... (FREE tier: 5 calls/sec)")
    # Test API key validity
    try:
        test_url = "https://api.bscscan.com/api"
        test_params = {
            'module': 'account',
            'action': 'balance',
            'address': CONTRACT_ADDRESS,
            'apikey': BSCSCAN_API_KEY
        }
        test_response = requests.get(test_url, params=test_params, timeout=5)
        test_data = test_response.json()
        if test_data.get('status') == '1':
            logger.info(f"✅ BscScan API key validated successfully")
        else:
            logger.error(f"❌ BscScan API key validation failed: {test_data.get('message', 'Unknown error')}")
            if MORALIS_API_KEY:
                logger.info(f"✅ Will use Moralis API as fallback")
    except Exception as e:
        logger.warning(f"⚠️ Could not validate BscScan API key: {e}")
        if MORALIS_API_KEY:
            logger.info(f"✅ Will use Moralis API as fallback")
elif MORALIS_API_KEY:
    logger.info(f"✅ Moralis API key configured: {MORALIS_API_KEY[:10]}... (FREE tier: 40k compute units/day)")

logger.info(f"Environment loaded successfully. APP_URL={APP_URL}, PORT={PORT}")

# Constants
EMOJI = '💰'
PANCAKESWAP_PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"internalType": "uint112", "name": "_reserve0", "type": "uint112"},
            {"internalType": "uint112", "name": "_reserve1", "type": "uint112"},
            {"internalType": "uint32", "name": "_blockTimestampLast", "type": "uint32"}
        ],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]
cloudinary_videos = {
    'MicroPets Buy': 'SMALLBUY_b3px1p',
    'Medium Bullish Buy': 'MEDIUMBUY_MPEG_e02zdz',
    'Whale Buy': 'micropets_big_msap',
    'Extra Large Buy': 'micropets_big_msapxz'
}
BNB_ADDRESS = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c'
BUY_THRESHOLDS = {
    'small': 100,
    'medium': 500,
    'large': 1000
}
PETS_TOKEN_DECIMALS = 18

# Max number of blocks to scan per window
LOG_WINDOW_SIZE = 1000

TX_HASH_REGEX = re.compile(r"0x[a-fA-F0-9]{64}")

def extract_tx_hash(text: str) -> Optional[str]:
    """
    Extract a transaction hash from plain text or a URL.
    Works with:
    - 0x67bb68f2f91c741c97f6f285...
    - https://bscscan.com/tx/0x67bb68f2f91c741c97f6f285...
    """
    if not text:
        return None
    match = TX_HASH_REGEX.search(text)
    if not match:
        return None
    return match.group(0)

def get_last_transaction_block() -> int:
    """
    Find the block number of the last transaction you referenced.
    This ensures we start from a known good point.
    """
    known_tx_hash = "0x67bb68f2f91c741c97f6f285671286e4de9a0118edc4b49a368a3ca89ba3ffd3"
    try:
        tx = w3.eth.get_transaction(known_tx_hash)
        block_num = tx['blockNumber']
        logger.info(f"Last known transaction was at block {block_num}")
        return block_num
    except Exception as e:
        logger.error(f"Could not fetch last known transaction: {e}")
        # Fallback: use current block minus 1 day
        return w3.eth.block_number - (24 * 60 * 60 // 3)

# Rate limiter
class RateLimiter:
    def __init__(self, calls_per_second: float = 0.2):
        self.min_interval = 1.0 / calls_per_second
        self.last_call: Dict[str, float] = {}
        self.lock = threading.Lock()

    def wait(self, key: str = "default"):
        with self.lock:
            now = time.time()
            if key in self.last_call:
                elapsed = now - self.last_call[key]
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
            self.last_call[key] = now

# Rate limiters for different APIs
api_limiter = RateLimiter(calls_per_second=0.1)  # For GeckoTerminal etc
bscscan_limiter = RateLimiter(calls_per_second=4.0)  # BscScan FREE tier: 5/sec (use 4 to be safe)

# In-memory data
transaction_cache: List[Dict] = []
active_chats: Set[str] = {TELEGRAM_CHAT_ID}
last_transaction_hash: Optional[str] = None
last_block_number: Optional[int] = None
is_tracking_enabled: bool = False
recent_errors: List[Dict] = []
last_transaction_fetch: Optional[float] = None
TRANSACTION_CACHE_THRESHOLD = 10 * 60 * 1000
posted_transactions: Set[str] = set()
transaction_details_cache: Dict[str, float] = {}
monitoring_task = None
polling_task = None
file_lock = threading.Lock()

# Initialize Web3 with Ankr RPC for BSC
# BSC block time: ~3 seconds
# Ankr provides good reliability for BSC but has rate limits on free tier
try:
    w3 = Web3(Web3.HTTPProvider(
        BNB_RPC_URL, 
        request_kwargs={
            'timeout': 30  # Reduced from 60s - BSC is faster than ETH
        }
    ))
    if not w3.is_connected():
        raise Exception("Primary RPC URL (Ankr) connection failed")
    logger.info(f"Successfully initialized Web3 with Ankr BSC RPC: {BNB_RPC_URL[:50]}...")
except Exception as e:
    logger.error(f"Failed to initialize Web3 with Ankr URL: {e}")
    # Fallback to public BSC RPC
    fallback_urls = [
        'https://bsc-dataseed1.binance.org',
        'https://bsc-dataseed2.binance.org',
        'https://bsc-dataseed3.binance.org'
    ]
    
    for fallback_url in fallback_urls:
        try:
            logger.info(f"Trying fallback: {fallback_url}")
            w3 = Web3(Web3.HTTPProvider(fallback_url, request_kwargs={'timeout': 30}))
            if w3.is_connected():
                logger.info(f"Web3 initialized with fallback: {fallback_url}")
                break
        except Exception as fallback_e:
            logger.error(f"Fallback {fallback_url} failed: {fallback_e}")
            continue
    else:
        logger.error("All RPC endpoints failed")
        raise ValueError("Failed to connect to any BSC RPC endpoint (Ankr and all fallbacks)")

# === Transfer event ABI + contract ===
ERC20_TRANSFER_ABI = [{
    "anonymous": False,
    "inputs": [
        {"indexed": True, "internalType": "address", "name": "from", "type": "address"},
        {"indexed": True, "internalType": "address", "name": "to", "type": "address"},
        {"indexed": False, "internalType": "uint256", "name": "value", "type": "uint256"},
    ],
    "name": "Transfer",
    "type": "event"
}]

token_event_contract = w3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS),
    abi=ERC20_TRANSFER_ABI
)

def get_block_timestamp(block_number: int) -> int:
    """Get accurate timestamp for a block using Web3."""
    try:
        block = w3.eth.get_block(block_number)
        return block['timestamp']
    except Exception as e:
        logger.error(f"Failed to get block timestamp for {block_number}: {e}")
        return int(time.time())

@retry(wait=wait_exponential(multiplier=2, min=2, max=10), stop=stop_after_attempt(3))
def fetch_transactions_via_moralis_api(from_block: int, to_block: int) -> List[Dict]:
    """
    Fetch token transfers from Moralis API.

    Moralis FREE tier:
    - 40,000 compute units/day
    - Token transfer endpoints
    - BSC support

    Returns transactions where FROM == TARGET_ADDRESS (LP address).
    """
    if not MORALIS_API_KEY:
        logger.warning("Moralis API key not configured, skipping API call")
        return []

    api_limiter.wait("moralis")

    url = f"https://deep-index.moralis.io/api/v2.2/{TARGET_ADDRESS}/erc20/transfers"
    headers = {
        'Accept': 'application/json',
        'X-API-Key': MORALIS_API_KEY
    }
    params = {
        'chain': 'bsc',
        'from_block': from_block,
        'to_block': to_block,
        'contract_addresses': [CONTRACT_ADDRESS],
        'limit': 100  # Max per page
    }

    try:
        logger.info(f"📡 Moralis API: Fetching transfers from block {from_block} to {to_block}")
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        results = data.get('result', [])
        transactions = []

        for tx in results:
            # Filter transfers FROM the LP address (buys)
            if tx.get('from_address', '').lower() == TARGET_ADDRESS.lower():
                try:
                    transactions.append({
                        'transactionHash': tx['transaction_hash'],
                        'from': Web3.to_checksum_address(tx['from_address']),
                        'to': Web3.to_checksum_address(tx['to_address']),
                        'value': tx['value'],
                        'blockNumber': int(tx['block_number']),
                        'timeStamp': int(datetime.fromisoformat(tx['block_timestamp'].replace('Z', '+00:00')).timestamp())
                    })
                except Exception as parse_error:
                    logger.warning(f"Failed to parse Moralis transaction: {parse_error}")
                    continue

        logger.info(f"✅ Moralis API: Found {len(transactions)} buy transactions (from {len(results)} total transfers)")
        return transactions

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logger.error("❌ Moralis API: Invalid API key")
            logger.error("💡 Get a FREE key at: https://moralis.io")
        elif e.response.status_code == 429:
            logger.error("❌ Moralis API: Rate limit exceeded")
            time.sleep(2)
        else:
            logger.error(f"❌ Moralis API HTTP error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Moralis API error: {e}")
        raise

@retry(wait=wait_exponential(multiplier=2, min=2, max=10), stop=stop_after_attempt(3))
def fetch_transactions_via_bscscan_api(startblock: int, endblock: int) -> List[Dict]:
    """
    Fetch token transfers from BscScan API using tokentx endpoint.

    BscScan FREE tier:
    - 5 calls/second
    - 100,000 calls/day
    - Works with ETHERSCAN_API_KEY (v2 unified API)

    Returns transactions where FROM == TARGET_ADDRESS (LP address).
    """
    if not BSCSCAN_API_KEY:
        logger.warning("BscScan API key not configured, skipping API call")
        return []

    bscscan_limiter.wait("tokentx")

    url = "https://api.bscscan.com/api"
    params = {
        'module': 'account',
        'action': 'tokentx',
        'contractaddress': CONTRACT_ADDRESS,
        'address': TARGET_ADDRESS,  # Filter by LP address
        'startblock': startblock,
        'endblock': endblock,
        'page': 1,
        'offset': 10000,  # Max 10k results per call
        'sort': 'desc',  # Newest first
        'apikey': BSCSCAN_API_KEY
    }

    try:
        logger.info(f"📡 BscScan API: Fetching tokentx from block {startblock} to {endblock}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get('status') != '1':
            error_msg = data.get('message', 'Unknown error')
            result = data.get('result', '')

            # Log detailed error information
            logger.error(f"❌ BscScan API Error:")
            logger.error(f"   Status: {data.get('status')}")
            logger.error(f"   Message: {error_msg}")
            logger.error(f"   Result: {result}")
            logger.error(f"   API Key Present: {'Yes' if BSCSCAN_API_KEY else 'No'}")
            logger.error(f"   API Key (first 10 chars): {BSCSCAN_API_KEY[:10] if BSCSCAN_API_KEY else 'N/A'}")

            # Handle rate limits
            if 'rate limit' in error_msg.lower():
                logger.error(f"🚫 BscScan rate limit hit: {error_msg}")
                time.sleep(1)  # Back off
                raise Exception("Rate limit exceeded")

            # Handle missing/invalid API key
            if 'invalid' in error_msg.lower() or 'missing' in error_msg.lower():
                logger.error(f"🔑 BscScan API key issue: {error_msg}")
                logger.error(f"💡 Get a FREE API key at: https://bscscan.com/myapikey")
                raise Exception(f"Invalid or missing API key: {error_msg}")

            # Handle max block range exceeded
            if 'result window' in error_msg.lower() or 'exceed' in error_msg.lower():
                logger.error(f"📊 Block range too large: {error_msg}")
                raise Exception(f"Block range exceeded: {error_msg}")

            # Generic error
            logger.warning(f"⚠️ BscScan API returned error: {error_msg}")
            raise Exception(f"BscScan API error: {error_msg}")

        results = data.get('result', [])

        # Filter to only transfers FROM the LP address (buys)
        transactions = []
        for tx in results:
            # BscScan returns lowercase addresses
            if tx.get('from', '').lower() == TARGET_ADDRESS.lower():
                transactions.append({
                    'transactionHash': tx['hash'],
                    'from': Web3.to_checksum_address(tx['from']),
                    'to': Web3.to_checksum_address(tx['to']),
                    'value': tx['value'],
                    'blockNumber': int(tx['blockNumber']),
                    'timeStamp': int(tx['timeStamp'])
                })

        logger.info(f"✅ BscScan API: Found {len(transactions)} buy transactions (from {len(results)} total transfers)")
        return transactions

    except requests.exceptions.Timeout:
        logger.error("BscScan API timeout")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"BscScan API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"BscScan API error: {e}")
        raise

def fetch_transactions_window(from_block: int, to_block: int, retries: int = 3) -> List[Dict]:
    """
    Fetch Transfer events in a single small window [from_block, to_block]
    where `from == TARGET_ADDRESS`.

    Uses contract.events.Transfer.get_logs() with argument_filters for BSC.
    Includes retry logic with exponential backoff for BSC rate limits and timeouts.
    
    BSC Limits:
    - Max 5000 block range per eth_getLogs call (BSC hard limit)
    - Ankr RPC recommends smaller ranges for better performance
    - High traffic contracts may need even smaller windows
    """
    if from_block > to_block:
        logger.warning(f"Invalid window from_block={from_block} > to_block={to_block}")
        return []

    # Validate block range doesn't exceed BSC limit
    block_range = to_block - from_block
    if block_range > 5000:
        logger.error(f"Block range {block_range} exceeds BSC limit of 5000")
        return []

    checksummed_target = Web3.to_checksum_address(TARGET_ADDRESS)
    logger.info(f"Fetching Transfer events from block {from_block} to {to_block} (range: {block_range})")
    logger.info(f"Searching for transfers FROM {checksummed_target}")
    
    for attempt in range(retries):
        try:
            # Use the contract event's get_logs() with argument_filters
            # This is the recommended Web3.py approach and handles all encoding automatically
            # Note: Web3.py v6+ uses snake_case (from_block, to_block)
            logs = token_event_contract.events.Transfer.get_logs(
                from_block=from_block,
                to_block=to_block,
                argument_filters={'from': checksummed_target}
            )

            transactions: List[Dict] = []
            for event in logs:
                try:
                    # event.args contains decoded event arguments automatically
                    if event['args']['value'] > 0:
                        transactions.append({
                            "transactionHash": event['transactionHash'].hex(),
                            "to": Web3.to_checksum_address(event['args']['to']),
                            "from": Web3.to_checksum_address(event['args']['from']),
                            "value": str(event['args']['value']),
                            "blockNumber": event['blockNumber'],
                            # timeStamp added later in calling function
                        })
                except Exception as decode_error:
                    logger.error(f"Failed to decode event: {decode_error}")
                    continue

            logger.info(f"Found {len(transactions)} transfers in window {from_block}-{to_block}")
            return transactions

        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a BSC-specific error
            is_range_error = any(x in error_msg for x in ['exceed', 'range', 'limit', '5000', '-32005'])
            is_timeout = any(x in error_msg for x in ['timeout', 'timed out', 'context deadline'])
            is_rate_limit = any(x in error_msg for x in ['rate limit', '429', 'too many requests'])
            
            if attempt < retries - 1:
                # Exponential backoff: 2s, 4s, 8s
                delay = 2 ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{retries} failed for blocks {from_block}-{to_block}: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                
                # If block range is the issue and range is splittable, divide it
                if (is_range_error or is_timeout) and block_range > 100:
                    # Split the range in half and try recursively
                    mid_block = from_block + (block_range // 2)
                    logger.info(f"Splitting range due to error: {from_block}-{mid_block} and {mid_block + 1}-{to_block}")
                    
                    first_half = fetch_transactions_window(from_block, mid_block, retries=2)
                    time.sleep(0.5)  # Small delay between halves
                    second_half = fetch_transactions_window(mid_block + 1, to_block, retries=2)
                    
                    return first_half + second_half
                
                # If rate limited, wait longer
                if is_rate_limit and attempt == 0:
                    logger.warning("Rate limit detected, waiting 10 seconds...")
                    time.sleep(10)
            else:
                logger.error(
                    f"Failed to fetch logs after {retries} attempts for blocks {from_block}-{to_block}: {e}"
                )
                return []

    return []

def fetch_transactions_via_web3(from_block: Optional[int] = None,
                                to_block: Optional[int] = None) -> List[Dict]:
    """
    High-level log fetcher for BSC:
    - If no range: scan last LOG_WINDOW_SIZE blocks.
    - If start/end: scan all windows across the range.
    
    BSC-specific optimizations:
    - Respects BSC's 5000 block limit
    - Uses smaller windows (1000 blocks) for reliability
    - Includes delays between requests to avoid Ankr rate limits
    - Automatically handles large ranges by windowing
    """
    current_block = w3.eth.block_number
    if from_block is None and to_block is None:
        to_block = current_block
        from_block = max(1, to_block - LOG_WINDOW_SIZE)
    
    # Safety check: enforce BSC's 5000 block limit
    if from_block and to_block and (to_block - from_block) > 5000:
        logger.warning(f"Block range too large ({to_block - from_block}), limiting to BSC max of 5000 blocks")
        from_block = to_block - 5000
    elif from_block is not None and to_block is None:
        to_block = min(from_block + LOG_WINDOW_SIZE, current_block)
    elif from_block is None and to_block is not None:
        from_block = max(1, to_block - LOG_WINDOW_SIZE)

    all_txs: List[Dict] = []
    start = from_block
    end = to_block if to_block is not None else current_block
    
    total_blocks = end - start
    windows_needed = (total_blocks // LOG_WINDOW_SIZE) + 1
    logger.info(f"Fetching {total_blocks} blocks in ~{windows_needed} windows of {LOG_WINDOW_SIZE} blocks each")

    window_count = 0
    while start <= end:
        window_to = min(start + LOG_WINDOW_SIZE, end)
        window_count += 1
        
        logger.info(f"Processing window {window_count}/{windows_needed}: blocks {start}-{window_to}")
        window_txs = fetch_transactions_window(start, window_to)
        all_txs.extend(window_txs)
        
        if window_to == end:
            break
        start = window_to + 1
        
        # Small delay between windows to respect Ankr rate limits
        # Ankr free tier is rate limited, so we space out requests
        time.sleep(0.3)

    logger.info(f"Total transfers fetched across range {from_block}-{to_block}: {len(all_txs)}")
    return all_txs
def get_video_url(category: str) -> str:
    public_id = cloudinary_videos.get(category, 'micropets_big_msapxz')
    video_url = f"https://res.cloudinary.com/{CLOUDINARY_CLOUD_NAME}/video/upload/{public_id}.mp4"
    logger.info(f"Generated video URL for {category}: {video_url}")
    return video_url

def categorize_buy(usd_value: float) -> str:
    if usd_value < BUY_THRESHOLDS['small']:
        return 'MicroPets Buy'
    elif usd_value < BUY_THRESHOLDS['medium']:
        return 'Medium Bullish Buy'
    elif usd_value < BUY_THRESHOLDS['large']:
        return 'Whale Buy'
    return 'Extra Large Buy'

def shorten_address(address: str) -> str:
    return f"{address[:6]}...{address[-4:]}" if address and Web3.is_address(address) else ''

def load_posted_transactions() -> Set[str]:
    try:
        with file_lock:
            if not os.path.exists('posted_transactions.txt'):
                return set()
            with open('posted_transactions.txt', 'r') as f:
                return set(line.strip() for line in f if line.strip())
    except Exception as e:
        logger.warning(f"Could not load posted_transactions.txt: {e}")
        return set()

def log_posted_transaction(transaction_hash: str) -> None:
    try:
        with file_lock:
            with open('posted_transactions.txt', 'a') as f:
                f.write(transaction_hash + '\n')
    except Exception as e:
        logger.warning(f"Could not write to posted_transactions.txt: {e}")

@retry(wait=wait_exponential(multiplier=2, min=4, max=20), stop=stop_after_attempt(3))
def get_bnb_to_usd() -> float:
    try:
        api_limiter.wait("geckoterminal_bnb")
        headers = {'Accept': 'application/json;version=20230302'}
        response = requests.get(
            f"https://api.geckoterminal.com/api/v2/simple/networks/bsc/token_price/{BNB_ADDRESS}",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        price_str = data.get('data', {}).get('attributes', {}).get('token_prices', {}).get(BNB_ADDRESS.lower(), '0')
        if not isinstance(price_str, (str, float, int)) or not price_str:
            raise ValueError("Invalid price data from GeckoTerminal")
        price = float(price_str)
        if price <= 0:
            raise ValueError("GeckoTerminal returned non-positive price")
        logger.info(f"BNB price from GeckoTerminal: ${price:.2f}")
        time.sleep(0.5)
        return price
    except Exception as e:
        logger.error(f"GeckoTerminal BNB price fetch failed: {e}, status={getattr(e, 'response', None) and getattr(e.response, 'status_code', 'N/A')}")
        if not COINMARKETCAP_API_KEY:
            logger.warning("Skipping CoinMarketCap due to missing API key")
            return 600
        try:
            api_limiter.wait("coinmarketcap")
            response = requests.get(
                "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                headers={'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY},
                params={'symbol': 'BNB', 'convert': 'USD'},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            price_str = data.get('data', {}).get('BNB', {}).get('quote', {}).get('USD', {}).get('price', '0')
            if not isinstance(price_str, (str, float, int)) or not price_str:
                raise ValueError("Invalid price data from CoinMarketCap")
            price = float(price_str)
            if price <= 0:
                raise ValueError("CoinMarketCap returned non-positive price")
            logger.info(f"BNB price from CoinMarketCap: ${price:.2f}")
            return price
        except Exception as cmc_e:
            logger.error(f"CoinMarketCap BNB price fetch failed: {cmc_e}")
            return 600

@retry(wait=wait_exponential(multiplier=2, min=4, max=20), stop=stop_after_attempt(3))
def get_pets_price_from_pancakeswap() -> float:
    try:
        api_limiter.wait("geckoterminal_pets")
        headers = {'Accept': 'application/json;version=20230302'}
        response = requests.get(
            f"https://api.geckoterminal.com/api/v2/simple/networks/bsc/token_price/{CONTRACT_ADDRESS}",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        price_str = data.get('data', {}).get('attributes', {}).get('token_prices', {}).get(CONTRACT_ADDRESS.lower(), '0')
        if not isinstance(price_str, (str, float, int)) or not price_str:
            raise ValueError("Invalid price data from GeckoTerminal")
        price = float(price_str)
        if price <= 0:
            raise ValueError("GeckoTerminal returned non-positive price")
        logger.info(f"$PETS price from GeckoTerminal: ${price:.10f}")
        time.sleep(0.5)
        return price
    except Exception as e:
        logger.error(f"GeckoTerminal $PETS price fetch failed: {e}, status={getattr(e, 'response', None) and getattr(e.response, 'status_code', 'N/A')}")
        if not COINMARKETCAP_API_KEY:
            logger.warning("Skipping CoinMarketCap due to missing API key")
            try:
                pair_address = Web3.to_checksum_address(TARGET_ADDRESS)
                pair_contract = w3.eth.contract(address=pair_address, abi=PANCAKESWAP_PAIR_ABI)
                reserves = pair_contract.functions.getReserves().call()
                reserve0, reserve1, _ = reserves
                bnb_per_pets = reserve1 / reserve0 / 10**PETS_TOKEN_DECIMALS if reserve0 > 0 else 0
                bnb_to_usd = get_bnb_to_usd()
                price = bnb_per_pets * bnb_to_usd
                if price <= 0:
                    raise ValueError("PancakeSwap returned non-positive price")
                logger.info(f"$PETS price from PancakeSwap: ${price:.10f}")
                return price
            except Exception as pcs_e:
                logger.error(f"PancakeSwap $PETS price fetch failed: {pcs_e}")
                return 0.00003886
        try:
            api_limiter.wait("coinmarketcap")
            response = requests.get(
                "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                headers={'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY},
                params={'symbol': 'PETS', 'convert': 'USD'},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            price_str = data.get('data', {}).get('PETS', {}).get('quote', {}).get('USD', {}).get('price', '0')
            if not isinstance(price_str, (str, float, int)) or not price_str:
                raise ValueError("Invalid price data from CoinMarketCap")
            price = float(price_str)
            if price <= 0:
                raise ValueError("CoinMarketCap returned non-positive price")
            logger.info(f"$PETS price from CoinMarketCap: ${price:.10f}")
            return price
        except Exception as cmc_e:
            logger.error(f"CoinMarketCap $PETS price fetch failed: {cmc_e}")
            try:
                pair_address = Web3.to_checksum_address(TARGET_ADDRESS)
                pair_contract = w3.eth.contract(address=pair_address, abi=PANCAKESWAP_PAIR_ABI)
                reserves = pair_contract.functions.getReserves().call()
                reserve0, reserve1, _ = reserves
                bnb_per_pets = reserve1 / reserve0 / 10**PETS_TOKEN_DECIMALS if reserve0 > 0 else 0
                bnb_to_usd = get_bnb_to_usd()
                price = bnb_per_pets * bnb_to_usd
                if price <= 0:
                    raise ValueError("PancakeSwap returned non-positive price")
                logger.info(f"$PETS price from PancakeSwap: ${price:.10f}")
                return price
            except Exception as pcs_e:
                logger.error(f"PancakeSwap $PETS price fetch failed: {pcs_e}")
                return 0.00003886

@retry(wait=wait_exponential(multiplier=2, min=4, max=20), stop=stop_after_attempt(3))
def get_token_supply() -> float:
    """Get total supply directly from the token contract via Web3."""
    try:
        contract_abi = [{
            "constant": True,
            "inputs": [],
            "name": "totalSupply",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function"
        }]

        contract = w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACT_ADDRESS),
            abi=contract_abi
        )

        total_supply = contract.functions.totalSupply().call()
        supply = total_supply / 10**PETS_TOKEN_DECIMALS
        logger.info(f"Token supply: {supply:,.0f} tokens")
        return supply
    except Exception as e:
        logger.error(f"Failed to get token supply via Web3: {e}")
        return 6_604_885_020

@retry(wait=wait_exponential(multiplier=2, min=4, max=20), stop=stop_after_attempt(3))
def extract_market_cap() -> int:
    try:
        price = get_pets_price_from_pancakeswap()
        token_supply = get_token_supply()
        market_cap = int(token_supply * price)
        logger.info(f"Market cap: ${market_cap:,}")
        return market_cap
    except Exception as e:
        logger.error(f"Failed to calculate market cap: {e}")
        return 256600

@retry(wait=wait_exponential(multiplier=2, min=4, max=20), stop=stop_after_attempt(3))
def get_transaction_details(transaction_hash: str) -> Optional[float]:
    """Get BNB value directly from the transaction via Web3."""
    if transaction_hash in transaction_details_cache:
        logger.info(f"Using cached BNB value for transaction {transaction_hash}")
        return transaction_details_cache[transaction_hash]

    try:
        tx = w3.eth.get_transaction(transaction_hash)
        value_wei = tx['value']
        bnb_value = float(w3.from_wei(value_wei, 'ether'))
        transaction_details_cache[transaction_hash] = bnb_value
        logger.info(f"Transaction {transaction_hash}: BNB value={bnb_value:.6f}")
        return bnb_value
    except Exception as e:
        logger.error(f"Failed to get tx details via Web3 for {transaction_hash}: {e}")
        return None

@retry(wait=wait_exponential(multiplier=2, min=4, max=20), stop=stop_after_attempt(3))
def check_execute_function(transaction_hash: str) -> tuple[bool, Optional[float]]:
    """Check if tx input contains 'execute' using Web3, and get BNB value."""
    try:
        bnb_value = get_transaction_details(transaction_hash)
        if bnb_value is None:
            logger.error(f"No valid BNB value for transaction {transaction_hash}")
            return False, None

        tx = w3.eth.get_transaction(transaction_hash)
        input_data = tx.get('input', '').lower()
        is_execute = 'execute' in input_data

        logger.info(f"Transaction {transaction_hash}: Execute={is_execute}, BNB={bnb_value}")
        return is_execute, bnb_value

    except Exception as e:
        logger.error(f"Failed to check transaction {transaction_hash} via Web3: {e}")
        return False, transaction_details_cache.get(transaction_hash)

def get_balance_before_transaction(wallet_address: str, block_number: int) -> Optional[Decimal]:
    """Get token balance at a specific block using Web3."""
    try:
        contract_abi = [{
            "constant": True,
            "inputs": [{"name": "_owner", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "balance", "type": "uint256"}],
            "type": "function"
        }]

        contract = w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACT_ADDRESS),
            abi=contract_abi
        )

        raw_balance = contract.functions.balanceOf(
            Web3.to_checksum_address(wallet_address)
        ).call(block_identifier=block_number)

        balance = Decimal(raw_balance) / Decimal(10**PETS_TOKEN_DECIMALS)
        logger.info(f"Balance for {shorten_address(wallet_address)} at block {block_number}: {balance:,.0f} tokens")
        return balance
    except Exception as e:
        logger.error(f"Failed to fetch balance via Web3: {e}")
        return None

def fetch_bscscan_transactions(startblock: Optional[int] = None,
                               endblock: Optional[int] = None) -> List[Dict]:
    """
    Fetch buy transactions using BscScan API (preferred) or Web3 RPC (fallback).

    Priority:
    1. BscScan API (reliable, fast, free tier)
    2. Web3 RPC logs (slower, rate limits)
    """
    global transaction_cache, last_transaction_fetch, last_block_number

    try:
        current_block = w3.eth.block_number

        # If this is the first run, start from recent blocks only
        if last_block_number is None and startblock is None:
            startblock = current_block - (LOG_WINDOW_SIZE * 2)  # Only scan last ~400 blocks on first run
            logger.info(f"🚀 First run: starting from block {startblock} (current: {current_block})")
        elif not startblock and last_block_number:
            startblock = last_block_number + 1

        if endblock is None:
            endblock = current_block

        if startblock is None:
            startblock = max(1, endblock - LOG_WINDOW_SIZE)

        # Try multiple FREE APIs in priority order
        txs = []

        # Check if we have any API key
        if not BSCSCAN_API_KEY and not MORALIS_API_KEY:
            logger.error("="*60)
            logger.error("❌ CRITICAL: API key is required!")
            logger.error("="*60)
            logger.error("🔑 Choose ONE of these FREE options:")
            logger.error("")
            logger.error("   Option 1 (Fastest): BscScan")
            logger.error("      → https://bscscan.com/register")
            logger.error("      → Set: ETHERSCAN_API_KEY=YourKey")
            logger.error("      → FREE: 5 calls/sec, 100k/day")
            logger.error("")
            logger.error("   Option 2: Moralis")
            logger.error("      → https://moralis.io")
            logger.error("      → Set: MORALIS_API_KEY=YourKey")
            logger.error("      → FREE: 40k compute units/day")
            logger.error("="*60)
            logger.error("⏸️  Transaction monitoring PAUSED until API key is added")
            logger.error("="*60)
            return []

        # Try BscScan first (faster, more reliable)
        if BSCSCAN_API_KEY:
            try:
                logger.info(f"📡 Trying BscScan API (primary)")
                txs = fetch_transactions_via_bscscan_api(startblock, endblock)
                if txs:
                    logger.info(f"✅ BscScan API success!")
            except Exception as bsc_error:
                logger.warning(f"⚠️ BscScan API failed: {bsc_error}")
                if MORALIS_API_KEY:
                    logger.info(f"🔄 Trying Moralis API as fallback...")
                else:
                    logger.error(f"❌ No fallback API available")
                    return []

        # Try Moralis (fallback or primary if no BscScan key)
        if not txs and MORALIS_API_KEY:
            try:
                if BSCSCAN_API_KEY:
                    logger.info(f"📡 Trying Moralis API (fallback)")
                else:
                    logger.info(f"📡 Trying Moralis API (primary)")
                txs = fetch_transactions_via_moralis_api(startblock, endblock)
                if txs:
                    logger.info(f"✅ Moralis API success!")
            except Exception as moralis_error:
                logger.error(f"❌ Moralis API failed: {moralis_error}")
                logger.error(f"⏸️  Will retry on next polling interval")
                return []

        # Ensure timestamps and update cache
        for tx in txs:
            if 'blockNumber' in tx and 'timeStamp' not in tx:
                tx['timeStamp'] = get_block_timestamp(tx['blockNumber'])

        if txs:
            last_block_number = max(tx['blockNumber'] for tx in txs)
            transaction_cache = (transaction_cache + txs)[-1000:]
            last_transaction_fetch = datetime.now().timestamp() * 1000
            logger.info(f"✅ Fetched {len(txs)} buy transactions, last_block_number={last_block_number}")
        else:
            logger.info(f"No new transactions found")

        return txs

    except Exception as e:
        logger.error(f"❌ Failed to fetch transactions: {e}")
        return transaction_cache or []

async def send_video_with_retry(context, chat_id: str, video_url: str, options: Dict, max_retries: int = 3, delay: int = 2) -> bool:
    for i in range(max_retries):
        try:
            logger.info(f"Attempt {i+1}/{max_retries} to send video to chat {chat_id}: {video_url}")
            response = requests.head(video_url, timeout=5)
            if response.status_code != 200:
                logger.error(f"Video URL inaccessible, status {response.status_code}: {video_url}")
                raise Exception(f"Video URL inaccessible, status {response.status_code}")
            await context.bot.send_video(chat_id=chat_id, video=video_url, **options)
            logger.info(f"Successfully sent video to chat {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send video (attempt {i+1}/{max_retries}): {e}")
            if i == max_retries - 1:
                await context.bot.send_message(
                    chat_id,
                    f"{options['caption']}\n\n⚠️ Video unavailable",
                    parse_mode='Markdown'
                )
                return False
            await asyncio.sleep(delay)
    return False

async def process_transaction(context, transaction: Dict, bnb_to_usd_rate: float, pets_price: float, chat_id: str = TELEGRAM_CHAT_ID) -> bool:
    global posted_transactions
    try:
        if transaction['transactionHash'] in posted_transactions:
            logger.info(f"Skipping already posted transaction: {transaction['transactionHash']}")
            return False
        is_execute, bnb_value = check_execute_function(transaction['transactionHash'])
        if bnb_value is None or bnb_value <= 0:
            logger.info(f"Skipping transaction {transaction['transactionHash']} with invalid BNB value: {bnb_value}")
            return False
        pets_amount = float(transaction['value']) / 10**PETS_TOKEN_DECIMALS
        usd_value = pets_amount * pets_price
        if usd_value < 50:
            logger.info(f"Skipping transaction {transaction['transactionHash']} with USD value < 50: {usd_value}")
            return False
        market_cap = extract_market_cap()
        wallet_address = transaction['to']
        percent_increase = random.uniform(10, 120)
        holding_change_text = f"+{percent_increase:.2f}%"
        emoji_count = min(int(usd_value) // 1, 100)
        emojis = EMOJI * emoji_count
        tx_url = f"https://bscscan.com/tx/{transaction['transactionHash']}"
        category = categorize_buy(usd_value)
        video_url = get_video_url(category)
        message = (
            f"🚀 *MicroPets Buy!* BNB Chain 💰\n\n"
            f"{emojis}\n"
            f"💰 [$PETS](https://pancakeswap.finance/swap?outputCurrency={CONTRACT_ADDRESS}): {pets_amount:,.0f}\n"
            f"💵 BNB Value: {bnb_value:,.4f} (${(bnb_value * bnb_to_usd_rate):,.2f})\n"
            f"🏦 Market Cap: ${market_cap:,.0f}\n"
            f"🔼 Holding Change: {holding_change_text}\n"
            f"🦑 Hodler: {shorten_address(wallet_address)}\n"
            f"[🔍 View on BscScan]({tx_url})\n\n"
            f"💰 [Staking](https://pets.micropets.io/petdex) "
            f"[📈 Chart](https://www.dextools.io/app/en/bnb/pair-explorer/{TARGET_ADDRESS}) "
            f"[🛍 Merch](https://micropets.store/) "
            f"[🤑 Buy $PETS](https://pancakeswap.finance/swap?outputCurrency={CONTRACT_ADDRESS})"
        )
        success = await send_video_with_retry(context, chat_id, video_url, {'caption': message, 'parse_mode': 'Markdown'})
        if success:
            posted_transactions.add(transaction['transactionHash'])
            log_posted_transaction(transaction['transactionHash'])
            logger.info(f"Processed transaction {transaction['transactionHash']} for chat {chat_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error processing transaction {transaction.get('transactionHash', 'unknown')}: {e}")
        return False

async def monitor_transactions(context) -> None:
    global last_transaction_hash, last_block_number, is_tracking_enabled, monitoring_task
    logger.info("Starting transaction monitoring")
    
    # Initialize last_block_number if this is the first run
    if last_block_number is None:
        try:
            # Try to get the last known transaction block
            last_block_number = get_last_transaction_block()
            logger.info(f"Initialized from last known transaction: block {last_block_number}")
        except Exception as e:
            logger.error(f"Failed to initialize from last transaction: {e}")
            # Fallback: start from recent blocks only (not too far back)
            current_block = w3.eth.block_number
            last_block_number = max(1, current_block - (LOG_WINDOW_SIZE * 2))
            logger.info(f"Using fallback starting block: {last_block_number} (current: {current_block})")
    
    while is_tracking_enabled:
        async with asyncio.Lock():
            if not is_tracking_enabled:
                logger.info("Tracking disabled, stopping monitoring")
                break
            try:
                posted_transactions.update(load_posted_transactions())
                
                # Calculate block range for logging
                start_block = (last_block_number + 1) if last_block_number else None
                # BSC produces ~1200 blocks/hour, so 60s polling = ~20 new blocks to scan
                # This keeps us well within the 5000 block BSC limit
                
                current_block = w3.eth.block_number
                logger.info(f"Checking for transactions from block {start_block} to {current_block}")
                
                txs = fetch_bscscan_transactions(startblock=start_block)
                if not txs:
                    logger.info(f"No new transactions found (checked {start_block} to {current_block})")
                    await asyncio.sleep(POLLING_INTERVAL)
                    continue

                logger.info(f"Found {len(txs)} potential transactions to process")
                
                bnb_to_usd_rate = get_bnb_to_usd()
                pets_price = get_pets_price_from_pancakeswap()
                new_last_hash = last_transaction_hash
                for tx in sorted(txs, key=lambda x: x['blockNumber'], reverse=True):
                    if not isinstance(tx, dict):
                        logger.error(f"Invalid transaction format: {tx}")
                        continue
                    if tx['transactionHash'] in posted_transactions:
                        logger.info(f"Skipping already posted transaction: {tx['transactionHash']}")
                        continue
                    if last_transaction_hash and tx['transactionHash'] == last_transaction_hash:
                        continue
                    if last_block_number and tx['blockNumber'] <= last_block_number:
                        logger.info(f"Skipping old transaction {tx['transactionHash']} with block {tx['blockNumber']} <= {last_block_number}")
                        continue
                    if await process_transaction(context, tx, bnb_to_usd_rate, pets_price):
                        new_last_hash = tx['transactionHash']
                        last_block_number = max(last_block_number or 0, tx['blockNumber'])
                last_transaction_hash = new_last_hash
            except Exception as e:
                logger.error(f"Error monitoring transactions: {e}")
                recent_errors.append({'time': datetime.now().isoformat(), 'error': str(e)})
                if len(recent_errors) > 5:
                    recent_errors.pop(0)
            await asyncio.sleep(POLLING_INTERVAL)
    logger.info("Monitoring task stopped")
    monitoring_task = None

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
async def set_webhook_with_retry(bot_app) -> bool:
    webhook_url = f"https://{APP_URL}/webhook"
    logger.info(f"Attempting to set webhook: {webhook_url}")
    try:
        response = requests.get(f"https://{APP_URL}/health", timeout=10)
        if response.status_code != 200:
            logger.error(f"Health check failed, status {response.status_code}, response: {response.text}")
            raise Exception(f"Health check failed, status {response.status_code}")
        logger.info(f"Health check passed, status {response.status_code}")
        await bot_app.bot.delete_webhook(drop_pending_updates=True)
        logger.info("Deleted existing webhook")
        await bot_app.bot.set_webhook(webhook_url, allowed_updates=["message", "channel_post"])
        logger.info(f"Webhook set successfully: {webhook_url}")
        return True
    except Exception as e:
        logger.error(f"Failed to set webhook: {e}")
        raise

async def polling_fallback(bot_app) -> None:
    """Start polling mode if webhook setup fails."""
    global polling_task
    logger.info("🔄 Starting polling fallback mode")

    try:
        # Start polling
        await bot_app.updater.start_polling(
            poll_interval=3,
            timeout=10,
            drop_pending_updates=True,
            error_callback=lambda e: logger.error(f"Polling error: {e}")
        )
        logger.info("✅ Polling started successfully - bot is now active")

        # Keep polling alive until task is cancelled
        while True:
            await asyncio.sleep(60)
            if not bot_app.updater.running:
                logger.warning("⚠️ Updater stopped, restarting...")
                await bot_app.updater.start_polling(
                    poll_interval=3,
                    timeout=10,
                    drop_pending_updates=True,
                    error_callback=lambda e: logger.error(f"Polling error: {e}")
                )
    except asyncio.CancelledError:
        logger.info("🛑 Polling task cancelled")
        raise
    except Exception as e:
        logger.error(f"❌ Fatal polling error: {e}")
        await asyncio.sleep(10)
        raise

async def handle_message(update: Update, context) -> None:
    """
    Auto-detect and process transaction hashes sent as plain messages.
    Supports both plain tx hash and links like:
    https://bscscan.com/tx/0x...
    """
    try:
        # Safety checks
        if not update or not update.message or not update.message.text:
            return
        
        chat_id = update.effective_chat.id
        
        # Only work in admin chat
        if str(chat_id) != ADMIN_CHAT_ID:
            return
        
        message_text = update.message.text.strip()
        tx_hash = extract_tx_hash(message_text)
    except Exception as e:
        logger.error(f"Error in handle_message initial processing: {e}")
        return
    
    if not tx_hash:
        return

    # If we got here, it's a valid transaction hash!
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"🔍 Detected transaction hash, processing...\n`{tx_hash[:10]}...{tx_hash[-8:]}`",
        parse_mode='Markdown'
    )
    
    try:
        # Get transaction details from Web3
        tx = w3.eth.get_transaction(tx_hash)
        if not tx:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Transaction not found: `{tx_hash}`",
                parse_mode='Markdown'
            )
            return
        
        # Get transaction receipt to find logs
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        if not receipt:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Transaction receipt not found: `{tx_hash}`",
                parse_mode='Markdown'
            )
            return
        
        # Check if this is a transfer FROM the target address (a sell from LP = a buy)
        transfer_event = token_event_contract.events.Transfer
        checksummed_target = Web3.to_checksum_address(TARGET_ADDRESS)
        
        # Parse logs to find Transfer events (primary: from LP)
        found_transfer = None
        for log in receipt['logs']:
            try:
                if log['address'].lower() != CONTRACT_ADDRESS.lower():
                    continue
                
                decoded = transfer_event.process_log(log)
                
                if decoded['args']['from'].lower() == checksummed_target.lower():
                    found_transfer = {
                        'transactionHash': tx_hash,
                        'from': decoded['args']['from'],
                        'to': decoded['args']['to'],
                        'value': str(decoded['args']['value']),
                        'blockNumber': receipt['blockNumber'],
                        'timeStamp': get_block_timestamp(receipt['blockNumber'])
                    }
                    break
            except Exception:
                continue

        # Fallback: any PETS transfer in this tx
        if not found_transfer:
            for log in receipt['logs']:
                try:
                    if log['address'].lower() != CONTRACT_ADDRESS.lower():
                        continue
                    decoded = transfer_event.process_log(log)
                    found_transfer = {
                        'transactionHash': tx_hash,
                        'from': decoded['args']['from'],
                        'to': decoded['args']['to'],
                        'value': str(decoded['args']['value']),
                        'blockNumber': receipt['blockNumber'],
                        'timeStamp': get_block_timestamp(receipt['blockNumber'])
                    }
                    break
                except Exception:
                    continue
        
        if not found_transfer:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"❌ No PETS transfer found in `{tx_hash[:10]}...{tx_hash[-8:]}`\n\n"
                    f"This transaction doesn't appear to include a PETS Transfer from or to the LP address:\n"
                    f"`{TARGET_ADDRESS}`"
                ),
                parse_mode='Markdown'
            )
            return
        
        # Get current prices
        bnb_to_usd_rate = get_bnb_to_usd()
        pets_price = get_pets_price_from_pancakeswap()
        
        # Check if already posted
        if tx_hash in posted_transactions:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Transaction `{tx_hash[:10]}...{tx_hash[-8:]}` was already posted. Sending anyway...",
                parse_mode='Markdown'
            )
        
        # Process and send to both chats
        success_main = await process_transaction(
            context, 
            found_transfer, 
            bnb_to_usd_rate, 
            pets_price, 
            chat_id=TELEGRAM_CHAT_ID
        )
        
        success_admin = await process_transaction(
            context, 
            found_transfer, 
            bnb_to_usd_rate, 
            pets_price, 
            chat_id=ADMIN_CHAT_ID
        )
        
        if success_main or success_admin:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"✅ Successfully processed and posted!\n`{tx_hash[:10]}...{tx_hash[-8:]}`",
                parse_mode='Markdown'
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Transaction processed but may not meet posting criteria (USD value < $50)",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"Error in auto-detect handler: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Failed to process transaction: {str(e)}",
            parse_mode='Markdown'
        )

# ⬇️ NEW COMMANDS ⬇️

async def fetch_tx(update: Update, context) -> None:
    """
    Fetch and process a specific transaction by hash or URL.
    Usage: /fetch <transaction_hash_or_link>
    """
    chat_id = update.effective_chat.id
    logger.info(f"/fetch command received from chat {chat_id}")
    
    if not is_admin(update):
        logger.warning(f"Unauthorized /fetch attempt from chat {chat_id}")
        await context.bot.send_message(chat_id=chat_id, text="🚫 Unauthorized")
        return
    
    if not context.args or len(context.args) == 0:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "❌ Please provide a transaction hash or BscScan link\n\n"
                "Usage: `/fetch <tx_hash_or_url>`\n\n"
                "Example:\n"
                "`/fetch 0x67bb68f2f91c741c97f6f285671286e4de9a0118edc4b49a368a3ca89ba3ffd3`\n"
                "or\n"
                "`/fetch https://bscscan.com/tx/0x67bb68f2f91c741c97f6f285671286e4de9a0118edc4b49a368a3ca89ba3ffd3`"
            ),
            parse_mode='Markdown'
        )
        return
    
    raw_arg = " ".join(context.args).strip()
    tx_hash = extract_tx_hash(raw_arg)
    
    if not tx_hash:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ Could not find a valid transaction hash in your input.",
            parse_mode='Markdown'
        )
        return
    
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"⏳ Fetching transaction `{tx_hash[:10]}...{tx_hash[-8:]}`",
        parse_mode='Markdown'
    )
    
    try:
        # Get transaction details from Web3
        tx = w3.eth.get_transaction(tx_hash)
        if not tx:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Transaction not found: `{tx_hash}`",
                parse_mode='Markdown'
            )
            return
        
        # Get transaction receipt to find logs
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        if not receipt:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Transaction receipt not found: `{tx_hash}`",
                parse_mode='Markdown'
            )
            return
        
        # Check if this is a transfer FROM the target address (a sell from LP = a buy)
        transfer_event = token_event_contract.events.Transfer
        checksummed_target = Web3.to_checksum_address(TARGET_ADDRESS)
        
        found_transfer = None
        for log in receipt['logs']:
            try:
                if log['address'].lower() != CONTRACT_ADDRESS.lower():
                    continue
                
                decoded = transfer_event.process_log(log)
                
                if decoded['args']['from'].lower() == checksummed_target.lower():
                    found_transfer = {
                        'transactionHash': tx_hash,
                        'from': decoded['args']['from'],
                        'to': decoded['args']['to'],
                        'value': str(decoded['args']['value']),
                        'blockNumber': receipt['blockNumber'],
                        'timeStamp': get_block_timestamp(receipt['blockNumber'])
                    }
                    break
            except Exception:
                continue

        # Fallback: any PETS transfer
        if not found_transfer:
            for log in receipt['logs']:
                try:
                    if log['address'].lower() != CONTRACT_ADDRESS.lower():
                        continue
                    decoded = transfer_event.process_log(log)
                    found_transfer = {
                        'transactionHash': tx_hash,
                        'from': decoded['args']['from'],
                        'to': decoded['args']['to'],
                        'value': str(decoded['args']['value']),
                        'blockNumber': receipt['blockNumber'],
                        'timeStamp': get_block_timestamp(receipt['blockNumber'])
                    }
                    break
                except Exception:
                    continue
        
        if not found_transfer:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"❌ No PETS transfer found in `{tx_hash[:10]}...{tx_hash[-8:]}`\n\n"
                    f"This transaction doesn't appear to include a PETS Transfer from or to the LP address:\n"
                    f"`{TARGET_ADDRESS}`"
                ),
                parse_mode='Markdown'
            )
            return
        
        # Get current prices
        bnb_to_usd_rate = get_bnb_to_usd()
        pets_price = get_pets_price_from_pancakeswap()
        
        # Check if already posted
        if tx_hash in posted_transactions:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ Transaction `{tx_hash[:10]}...{tx_hash[-8:]}` was already posted. Sending anyway...",
                parse_mode='Markdown'
            )
        
        # Process and send to both chats
        success_main = await process_transaction(
            context, 
            found_transfer, 
            bnb_to_usd_rate, 
            pets_price, 
            chat_id=TELEGRAM_CHAT_ID
        )
        
        success_admin = await process_transaction(
            context, 
            found_transfer, 
            bnb_to_usd_rate, 
            pets_price, 
            chat_id=ADMIN_CHAT_ID
        )
        
        if success_main or success_admin:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"✅ Successfully processed transaction `{tx_hash[:10]}...{tx_hash[-8:]}`",
                parse_mode='Markdown'
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="⚠️ Transaction processed but may not meet posting criteria (USD value < $50)",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"Error in /fetch command: {e}")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Failed to fetch transaction: {str(e)}",
            parse_mode='Markdown'
        )

async def fetch_batch(update: Update, context) -> None:
    """
    Fetch and process multiple transactions by hash or URL.
    Usage: /fetchbatch <tx_hash_or_url1> <tx_hash_or_url2> ...
    """
    chat_id = update.effective_chat.id
    if not is_admin(update):
        await context.bot.send_message(chat_id=chat_id, text="🚫 Unauthorized")
        return
    
    if not context.args or len(context.args) == 0:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "❌ Please provide transaction hashes or BscScan links\n\n"
                "Usage: `/fetchbatch <tx1> <tx2> <tx3> ...`\n\n"
                "Example:\n"
                "`/fetchbatch 0x123... 0x456... 0x789...`"
            ),
            parse_mode='Markdown'
        )
        return
    
    raw_args = context.args
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"⏳ Processing {len(raw_args)} transactions...",
        parse_mode='Markdown'
    )
    
    processed = 0
    failed = 0
    skipped = 0
    
    for arg in raw_args:
        tx_hash = extract_tx_hash(arg.strip())
        if not tx_hash:
            logger.warning(f"Could not extract tx hash from: {arg}")
            failed += 1
            continue
        
        try:
            tx = w3.eth.get_transaction(tx_hash)
            if not tx:
                logger.warning(f"Transaction not found: {tx_hash}")
                failed += 1
                continue
            
            receipt = w3.eth.get_transaction_receipt(tx_hash)
            if not receipt:
                logger.warning(f"Receipt not found: {tx_hash}")
                failed += 1
                continue
            
            transfer_event = token_event_contract.events.Transfer
            checksummed_target = Web3.to_checksum_address(TARGET_ADDRESS)
            
            found_transfer = None
            for log in receipt['logs']:
                try:
                    if log['address'].lower() != CONTRACT_ADDRESS.lower():
                        continue
                    
                    decoded = transfer_event.process_log(log)
                    
                    if decoded['args']['from'].lower() == checksummed_target.lower():
                        found_transfer = {
                            'transactionHash': tx_hash,
                            'from': decoded['args']['from'],
                            'to': decoded['args']['to'],
                            'value': str(decoded['args']['value']),
                            'blockNumber': receipt['blockNumber'],
                            'timeStamp': get_block_timestamp(receipt['blockNumber'])
                        }
                        break
                except Exception:
                    continue
            
            # Fallback: any PETS transfer
            if not found_transfer:
                for log in receipt['logs']:
                    try:
                        if log['address'].lower() != CONTRACT_ADDRESS.lower():
                            continue
                        decoded = transfer_event.process_log(log)
                        found_transfer = {
                            'transactionHash': tx_hash,
                            'from': decoded['args']['from'],
                            'to': decoded['args']['to'],
                            'value': str(decoded['args']['value']),
                            'blockNumber': receipt['blockNumber'],
                            'timeStamp': get_block_timestamp(receipt['blockNumber'])
                        }
                        break
                    except Exception:
                        continue
            
            if not found_transfer:
                logger.warning(f"No PETS transfer found in: {tx_hash}")
                skipped += 1
                continue
            
            bnb_to_usd_rate = get_bnb_to_usd()
            pets_price = get_pets_price_from_pancakeswap()
            
            success = await process_transaction(
                context, 
                found_transfer, 
                bnb_to_usd_rate, 
                pets_price, 
                chat_id=TELEGRAM_CHAT_ID
            )
            
            if success:
                processed += 1
            else:
                skipped += 1
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing {tx_hash}: {e}")
            failed += 1
    
    summary = (
        f"📊 *Batch Processing Complete*\n\n"
        f"✅ Processed: {processed}\n"
        f"⚠️ Skipped: {skipped}\n"
        f"❌ Failed: {failed}\n"
        f"📝 Total: {len(raw_args)}"
    )
    
    await context.bot.send_message(
        chat_id=chat_id,
        text=summary,
        parse_mode='Markdown'
    )

def is_admin(update: Update) -> bool:
    return str(update.effective_chat.id) == ADMIN_CHAT_ID
    
# Command handlers
async def start(update: Update, context) -> None:
    chat_id = update.effective_chat.id
    active_chats.add(str(chat_id))
    await context.bot.send_message(chat_id=chat_id, text="👋 Welcome to PETS Tracker! Use /track to start buy alerts.")

async def track(update: Update, context) -> None:
    global is_tracking_enabled, monitoring_task
    chat_id = update.effective_chat.id
    if not is_admin(update):
        await context.bot.send_message(chat_id=chat_id, text="🚫 Unauthorized")
        return
    if is_tracking_enabled and monitoring_task:
        await context.bot.send_message(chat_id=chat_id, text="🚀 Tracking already enabled")
        return
    is_tracking_enabled = True
    active_chats.add(str(chat_id))
    monitoring_task = asyncio.create_task(monitor_transactions(context))
    await context.bot.send_message(chat_id=chat_id, text="🚖 Tracking started")

async def stop(update: Update, context) -> None:
    global is_tracking_enabled, monitoring_task
    chat_id = update.effective_chat.id
    if not is_admin(update):
        await context.bot.send_message(chat_id=chat_id, text="🚫 Unauthorized")
        return
    is_tracking_enabled = False
    if monitoring_task:
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            logger.info("Monitoring task cancelled")
        monitoring_task = None
    active_chats.discard(str(chat_id))
    await context.bot.send_message(chat_id=chat_id, text="🛑 Stopped")

async def stats(update: Update, context) -> None:
    chat_id = update.effective_chat.id
    if not is_admin(update):
        await context.bot.send_message(chat_id=chat_id, text="🚫 Unauthorized")
        return

    await context.bot.send_message(chat_id=chat_id, text="⏳ Fetching $PETS data for the last 2 weeks")
    try:
        latest_block = w3.eth.block_number
        logger.info(f"Latest BNB block (via Web3): {latest_block}")

        # Approximate blocks per day (~3s per block)
        blocks_per_day = 24 * 60 * 60 // 3
        start_block = latest_block - (14 * blocks_per_day)
        
        # Fetch in smaller chunks to avoid timeout
        logger.info(f"Fetching transactions from block {start_block} to {latest_block} ({latest_block - start_block} blocks)")
        all_txs = []
        chunk_size = 5000  # Process 5000 blocks at a time
        for chunk_start in range(start_block, latest_block, chunk_size):
            chunk_end = min(chunk_start + chunk_size, latest_block)
            logger.info(f"Fetching chunk: {chunk_start} to {chunk_end}")
            chunk_txs = fetch_bscscan_transactions(startblock=chunk_start, endblock=chunk_end)
            all_txs.extend(chunk_txs)
            await asyncio.sleep(1)  # Rate limit between chunks
        
        txs = all_txs
        if not txs:
            logger.info("No transactions found for the last 2 weeks")
            await context.bot.send_message(chat_id=chat_id, text="🚫 No recent buys found")
            return

        two_weeks_ago = int((datetime.now() - timedelta(days=14)).timestamp())
        recent_txs = [tx for tx in txs if isinstance(tx, dict) and tx.get('timeStamp', 0) >= two_weeks_ago]
        if not recent_txs:
            logger.info("No transactions within the last two weeks")
            await context.bot.send_message(chat_id=chat_id, text="🚫 No buys found in the last 2 weeks")
            return

        bnb_to_usd_rate = get_bnb_to_usd()
        pets_price = get_pets_price_from_pancakeswap()
        processed = []
        seen_hashes = set()

        for tx in sorted(recent_txs, key=lambda x: x['timeStamp'], reverse=True):
            if not isinstance(tx, dict):
                logger.error(f"Invalid transaction format in stats: {tx}")
                continue
            if tx['transactionHash'] in seen_hashes or tx['transactionHash'] in posted_transactions:
                logger.info(f"Skipping duplicate transaction: {tx['transactionHash']}")
                continue
            if await process_transaction(context, tx, bnb_to_usd_rate, pets_price, chat_id=TELEGRAM_CHAT_ID):
                processed.append(tx['transactionHash'])
            if await process_transaction(context, tx, bnb_to_usd_rate, pets_price, chat_id=ADMIN_CHAT_ID):
                processed.append(tx['transactionHash'])
            seen_hashes.add(tx['transactionHash'])
            await asyncio.sleep(0.5)

        if processed:
            unique = list(set(processed))
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"✅ Processed {len(unique)} buys from the last 2 weeks:\n" + "\n".join(unique),
                parse_mode='Markdown'
            )
        else:
            logger.info("No transactions met the $50 USD threshold")
            await context.bot.send_message(chat_id=chat_id, text="🚫 No transactions processed (all below $50 USD)")
    except Exception as e:
        logger.error(f"Error in /stats: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"🚫 Failed to fetch data: {str(e)}")

async def help_command(update: Update, context) -> None:
    chat_id = update.effective_chat.id
    if not is_admin(update):
        await context.bot.send_message(chat_id=chat_id, text="🚫 Unauthorized")
        return
    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            "🆘 *Commands:*\n\n"
            "📡 *Monitoring:*\n"
            "/start - Start bot\n"
            "/track - Enable alerts\n"
            "/stop - Disable alerts\n"
            "/status - Tracking status\n\n"
            "🔍 *Manual Fetch:*\n"
            "/fetch `<tx_hash_or_url>` - Process single transaction\n"
            "/fetchbatch `<tx1> <tx2>` - Process multiple transactions\n"
            "📋 Just paste a tx hash or link - Auto-detected! ✨\n"
            "/stats - View buys from last 2 weeks ⚠️ (heavy)\n\n"
            "🧪 *Testing:*\n"
            "/test - Test with video\n"
            "/noV - Test without video\n\n"
            "🛠 *Debug:*\n"
            "/debug - Debug info\n"
            "/help - This message\n\n"
            "💡 *Pro Tip:*\n"
            "Just paste transaction hashes **or BscScan links** directly - no commands needed!\n"
            "Example: `0x67bb68f2f91c741c97f6f285671286e4de9a0118edc4b49a368a3ca89ba3ffd3`"
        ),
        parse_mode='Markdown'
    )

async def status(update: Update, context) -> None:
    chat_id = update.effective_chat.id
    if not is_admin(update):
        await context.bot.send_message(chat_id=chat_id, text="🚫 Unauthorized")
        return
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"🔍 *Status:* {'Enabled' if is_tracking_enabled else 'Disabled'}",
        parse_mode='Markdown'
    )

async def debug(update: Update, context) -> None:
    chat_id = update.effective_chat.id
    if not is_admin(update):
        await context.bot.send_message(chat_id=chat_id, text="🚫 Unauthorized")
        return
    status_obj = {
        'trackingEnabled': is_tracking_enabled,
        'activeChats': list(active_chats),
        'lastTxHash': last_transaction_hash,
        'lastBlockNumber': last_block_number,
        'recentErrors': recent_errors[-5:],
        'apiStatus': {
            'bscWeb3': bool(w3.is_connected()),
            'lastTransactionFetch': datetime.fromtimestamp(last_transaction_fetch / 1000).isoformat() if last_transaction_fetch else None
        },
        'pollingActive': polling_task is not None and not polling_task.done()
    }
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"🔍 Debug:\n```json\n{json.dumps(status_obj, indent=2)}\n```",
        parse_mode='Markdown'
    )

async def test(update: Update, context) -> None:
    chat_id = update.effective_chat.id
    if not is_admin(update):
        await context.bot.send_message(chat_id=chat_id, text="🚫 Unauthorized")
        return
    await context.bot.send_message(chat_id=chat_id, text="⏳ Generating test buy")
    try:
        test_tx_hash = f"0xTest{uuid.uuid4().hex[:16]}"
        test_pets_amount = random.randint(1000000, 5000000)
        pets_price = get_pets_price_from_pancakeswap()
        usd_value = test_pets_amount * pets_price
        bnb_to_usd_rate = get_bnb_to_usd()
        bnb_value = usd_value / bnb_to_usd_rate
        category = categorize_buy(usd_value)
        video_url = get_video_url(category)
        wallet_address = Web3.to_checksum_address(f"0x{os.urandom(20).hex()}")
        emoji_count = min(int(usd_value) // 1, 100)
        emojis = EMOJI * emoji_count
        market_cap = extract_market_cap()
        holding_change_text = f"+{random.uniform(10, 120):.2f}%"
        tx_url = f"https://bscscan.com/tx/{test_tx_hash}"
        message = (
            f"🚖 *MicroPets Buy!* Test\n\n"
            f"{emojis}\n"
            f"💰 [$PETS](https://pancakeswap.finance/swap?outputCurrency={CONTRACT_ADDRESS}): {test_pets_amount:,.0f}\n"
            f"💵 BNB Value: {bnb_value:,.4f} (${(bnb_value * bnb_to_usd_rate):,.2f})\n"
            f"🏦 Market Cap: ${market_cap:,.0f}\n"
            f"🔼 Holding: {holding_change_text}\n"
            f"🦒 Hodler: {shorten_address(wallet_address)}\n"
            f"[🔍 View]({tx_url})\n\n"
            f"💰 [Staking](https://pets.micropets.io/) "
            f"[📈 Chart](https://www.dextools.io/app/en/bnb/pair-explorer/{TARGET_ADDRESS}) "
            f"[🛍 Merch](https://micropets.store/) "
            f"[🤑 Buy](https://pancakeswap.finance/swap?outputCurrency={CONTRACT_ADDRESS})"
        )
        await send_video_with_retry(context, chat_id, video_url, {'caption': message, 'parse_mode': 'Markdown'})
        await context.bot.send_message(chat_id=chat_id, text="🚖 Success")
    except Exception as e:
        logger.error(f"Test error: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"🚫 Failed: {str(e)}")

async def no_video(update: Update, context) -> None:
    chat_id = update.effective_chat.id
    if not is_admin(update):
        await context.bot.send_message(chat_id=chat_id, text="🚫 Unauthorized")
        return
    await context.bot.send_message(chat_id=chat_id, text="⏳ Testing buy (no video)")
    try:
        test_tx_hash = f"0xTestNoV{uuid.uuid4().hex[:16]}"
        test_pets_amount = random.randint(1000000, 5000000)
        pets_price = get_pets_price_from_pancakeswap()
        usd_value = test_pets_amount * pets_price
        bnb_to_usd_rate = get_bnb_to_usd()
        bnb_value = usd_value / bnb_to_usd_rate
        wallet_address = Web3.to_checksum_address(f"0x{os.urandom(20).hex()}")
        emoji_count = min(int(usd_value) // 1, 100)
        emojis = EMOJI * emoji_count
        market_cap = extract_market_cap()
        holding_change_text = f"+{random.uniform(10, 120):.2f}%"
        tx_url = f"https://bscscan.com/tx/{test_tx_hash}"
        message = (
            f"🚖 *MicroPets Buy!* BNB Chain\n\n"
            f"{emojis}\n"
            f"💰 [$PETS](https://pancakeswap.finance/swap?outputCurrency={CONTRACT_ADDRESS}): {test_pets_amount:,.0f}\n"
            f"💵 BNB: {bnb_value:,.4f} (${(bnb_value * bnb_to_usd_rate):,.2f})\n"
            f"🏦 Market Cap: ${market_cap:,.0f}\n"
            f"🔼 Holding: {holding_change_text}\n"
            f"🦀 Hodler: {shorten_address(wallet_address)}\n"
            f"[🔍]({tx_url})\n\n"
            f"[💰 Staking](https://pets.micropets.io/) "
            f"[📈 Chart](https://www.dextools.io/app/en/bnb/pair-explorer/{TARGET_ADDRESS}) "
            f"[🛍 Merch](https://micropets.store/) "
            f"[💖 Buy](https://pancakeswap.finance/swap?outputCurrency={CONTRACT_ADDRESS})"
        )
        await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
        await context.bot.send_message(chat_id=chat_id, text="🚖 OK")
    except Exception as e:
        logger.error(f"/noV error: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"🚖 Error: {str(e)}")

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler - manages bot initialization and keeps app running.
    """
    global monitoring_task, polling_task

    logger.info("="*60)
    logger.info("🚀 Starting MicroPets BNBPETS Tracker Bot")
    logger.info("="*60)

    # Initialization phase
    try:
        await bot_app.initialize()
        await bot_app.start()
        logger.info("✅ Bot initialized successfully")

        # Try webhook first, fallback to polling
        webhook_success = False
        if APP_URL:
            try:
                await set_webhook_with_retry(bot_app)
                webhook_success = True
                logger.info("✅ Webhook mode active - bot ready for commands")
                logger.info(f"📡 Webhook URL: https://{APP_URL}/webhook")
            except Exception as e:
                logger.error(f"❌ Webhook setup failed: {e}")
                webhook_success = False

        if not webhook_success:
            logger.info("🔄 Starting polling mode as fallback")
            polling_task = asyncio.create_task(polling_fallback(bot_app))
            logger.info("✅ Polling mode active - bot ready for commands")

        logger.info("="*60)
        logger.info("📱 Bot Commands:")
        logger.info("   /track - Start monitoring transactions")
        logger.info("   /stop  - Stop monitoring")
        logger.info("   /help  - Show all commands")
        logger.info("="*60)
        logger.info("✨ Bot is now running and waiting for commands...")
        logger.info("💡 Use /track to start transaction monitoring")
        logger.info("="*60)

        # Yield to keep app running
        yield

    except Exception as e:
        logger.error(f"❌ Fatal startup error: {e}")
        raise

    # Shutdown phase
    finally:
        logger.info("="*60)
        logger.info("🛑 Initiating bot shutdown...")
        logger.info("="*60)

        try:
            # Cancel monitoring task
            if monitoring_task and not monitoring_task.done():
                logger.info("Stopping monitoring task...")
                monitoring_task.cancel()
                try:
                    await monitoring_task
                except asyncio.CancelledError:
                    logger.info("✅ Monitoring task stopped")
                monitoring_task = None

            # Cancel polling task
            if polling_task and not polling_task.done():
                logger.info("Stopping polling task...")
                polling_task.cancel()
                try:
                    await polling_task
                except asyncio.CancelledError:
                    logger.info("✅ Polling task stopped")
                polling_task = None

            # Stop bot
            if bot_app.running:
                logger.info("Stopping bot updater...")
                try:
                    if bot_app.updater and bot_app.updater.running:
                        await bot_app.updater.stop()
                    await bot_app.stop()
                    logger.info("✅ Bot stopped")
                except Exception as e:
                    logger.error(f"Error stopping bot: {e}")

            # Clean up webhook
            try:
                await bot_app.bot.delete_webhook(drop_pending_updates=True)
                logger.info("✅ Webhook deleted")
            except Exception as e:
                logger.error(f"Error deleting webhook: {e}")

            # Shutdown bot
            await bot_app.shutdown()
            logger.info("✅ Bot shutdown completed")

            logger.info("="*60)
            logger.info("👋 Bot stopped gracefully")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

# FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    try:
        if not w3.is_connected():
            raise Exception("Web3 is not connected")
        return {"status": "Connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

@app.get("/webhook")
async def webhook_get():
    logger.info("Received GET webhook")
    raise HTTPException(status_code=405, detail="Method Not Allowed")

@app.get("/api/transactions")
async def get_transactions():
    logger.info("Fetching transactions via API")
    return transaction_cache

@app.post("/webhook")
async def webhook(request: Request):
    logger.info("Received POST webhook request")
    try:
        data = await request.json()
        if not isinstance(data, dict):
            logger.error(f"Invalid webhook data: {data}")
            return {"error": "Invalid JSON data"}, 400
        update = Update.de_json(data, bot_app.bot)
        if update:
            await bot_app.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        recent_errors.append({"time": datetime.now().isoformat(), "error": str(e)})
        if len(recent_errors) > 5:
            recent_errors.pop(0)
        return {"error": "Webhook failed"}, 500

# Bot initialization
bot_app = ApplicationBuilder() \
    .token(TELEGRAM_BOT_TOKEN) \
    .build()
bot_app.add_handler(CommandHandler("start", start))
bot_app.add_handler(CommandHandler("track", track))
bot_app.add_handler(CommandHandler("stop", stop))
bot_app.add_handler(CommandHandler("stats", stats))
bot_app.add_handler(CommandHandler("fetch", fetch_tx))
bot_app.add_handler(CommandHandler("fetchbatch", fetch_batch))
bot_app.add_handler(CommandHandler("help", help_command))
bot_app.add_handler(CommandHandler("status", status))
bot_app.add_handler(CommandHandler("debug", debug))
bot_app.add_handler(CommandHandler("test", test))
bot_app.add_handler(CommandHandler("noV", no_video))

# Auto-detect transaction hashes in messages (must be last)
bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)


