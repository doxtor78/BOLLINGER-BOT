import ccxt
import json
import os
import time
import logging
import pandas as pd
from dotenv import load_dotenv
import requests
from datetime import datetime, timezone

# --- Setup ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
STATE_FILE = "state.json"

# --- Time Synchronization ---
def get_server_time():
    """Get BitMEX server time for synchronization"""
    try:
        response = requests.get('https://testnet.bitmex.com/api/v1/announcement')
        server_time = response.headers.get('Date')
        if server_time:
            server_timestamp = pd.to_datetime(server_time).timestamp()
            local_timestamp = time.time()
            time_offset = server_timestamp - local_timestamp
            logger.info(f"Time offset from server: {time_offset:.2f} seconds")
            return time_offset
    except Exception as e:
        logger.warning(f"Could not get server time: {e}")
    return 0

def resync_time(exchange):
    """Periodically resync time with the server."""
    logger.info("Resynchronizing time with server...")
    time_offset = get_server_time()
    # get_server_time returns 0 on failure, so we can't check for None.
    # We will update it anyway. A zero offset is better than a stale one if sync fails.
    exchange.options['timeDifference'] = time_offset * 1000
    logger.info(f"Time offset updated to: {time_offset:.2f} seconds")

# --- Config ---
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("config.json not found.")
        return None

def initialize_exchange():
    api_key = os.getenv('BITMEX_TESTNET_API_KEY')
    api_secret = os.getenv('BITMEX_TESTNET_API_SECRET')
    if not api_key or "YOUR_API_KEY" in api_key or not api_secret or "YOUR_API_SECRET" in api_secret:
        logger.error("API keys not found or are placeholders in .env file.")
        return None
    logger.info("Initializing connection to BitMEX testnet...")
    
    # Get time offset for synchronization
    time_offset = get_server_time()
    
    exchange = ccxt.bitmex({
        'apiKey': api_key, 
        'secret': api_secret,
        'timeout': 30000,  # 30 seconds timeout
        'rateLimit': 1000,  # 1 second between requests
        'enableRateLimit': True,
        'options': {
            'adjustForTimeDifference': True,
            'timeDifference': time_offset * 1000,  # Convert to milliseconds
        }
    })
    exchange.set_sandbox_mode(True)
    
    # Test connection
    try:
        exchange.load_markets()
        logger.info("Successfully connected to BitMEX testnet")
        return exchange
    except Exception as e:
        logger.error(f"Failed to connect to BitMEX: {e}")
        return None

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'last_signal': None,
        'open_order_id': None,
        'open_order_side': None,
        'open_order_price': None,
        'position_size': 0,
        'signal_confirm_count': 0,
        'entry_order_ids': [],
        'tp_placed_for_entry': {},
        'tp_order_ids': {},
        'tp_filled_amount': {},
        'tp_band_prices': {}  # Track last used TP band prices for each entry order
    }

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

# --- Utility Functions ---
def round_to_tick(price, tick_size):
    return round(round(price / tick_size) * tick_size, 8)

# --- Exchange Functions ---
def fetch_position(exchange, symbol):
    try:
        logger.info("Fetching position...")
        # Add retry logic for API calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to fetch position (attempt {attempt + 1}/{max_retries})")
                positions = exchange.private_get_position()
                logger.info("Successfully fetched position data.")
                for pos in positions:
                    if pos.get('symbol') == symbol:
                        position_size = int(pos.get('currentQty', 0))
                        logger.info(f"Position for {symbol}: {position_size}")
                        return position_size
                logger.info(f"No position found for {symbol}.")
                return 0
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying position fetch (attempt {attempt + 1}): {e}")
                    time.sleep(2)
                else:
                    raise e
    except Exception as e:
        logger.error(f"Could not fetch position: {e}")
        return None

def fetch_open_orders(exchange, symbol):
    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return exchange.fetch_open_orders(symbol)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retrying open orders fetch (attempt {attempt + 1}): {e}")
                    time.sleep(2)
                else:
                    raise e
    except Exception as e:
        logger.error(f"Could not fetch open orders: {e}")
        return []

def cancel_all_orders(exchange, symbol):
    try:
        open_orders = fetch_open_orders(exchange, symbol)
        for order in open_orders:
            exchange.cancel_order(order['id'], symbol)
            logger.info(f"Cancelled order {order['id']}")
    except Exception as e:
        logger.error(f"Error cancelling orders: {e}")

# --- Signal Logic ---
def get_signal(df, sma_window):
    price = df['close'].iloc[-1]
    sma = df['close'].rolling(window=sma_window).mean().iloc[-1]
    std = df['close'].rolling(window=sma_window).std().iloc[-1]
    # Calculate Bollinger Bands for 1, 1.5, 2, 2.5, 3 stddev
    bands = {
        '-3': sma - 3 * std,
        '-2.5': sma - 2.5 * std,
        '-2': sma - 2 * std,
        '-1.5': sma - 1.5 * std,
        '-1': sma - 1 * std,
        '+1': sma + 1 * std,
        '+1.5': sma + 1.5 * std,
        '+2': sma + 2 * std,
        '+2.5': sma + 2.5 * std,
        '+3': sma + 3 * std,
    }
    # Long: price between -3σ and -1σ
    if bands['-3'] <= price <= bands['-1']:
        return 'long', price, sma, bands
    # Short: price between +1σ and +3σ
    elif bands['+1'] <= price <= bands['+3']:
        return 'short', price, sma, bands
    else:
        return 'flat', price, sma, bands

# --- Order Management ---
def place_limit_order(exchange, symbol, side, amount, price, tick_size):
    price = round_to_tick(price, tick_size)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if side == 'buy':
                order = exchange.create_limit_buy_order(symbol, amount, price)
            else:
                order = exchange.create_limit_sell_order(symbol, amount, price)
            logger.info(f"Placed {side.upper()} LIMIT order for {amount} at {price}")
            return order['id']
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Retrying order placement (attempt {attempt + 1}): {e}")
                time.sleep(2)
            else:
                logger.error(f"Failed to place {side} order: {e}")
                return None

def amend_order(exchange, symbol, order_id, new_price, tick_size):
    new_price = round_to_tick(new_price, tick_size)
    try:
        order = exchange.fetch_order(order_id, symbol)
        if order['status'] in ['open', 'partially_filled']:
            exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order {order_id} to amend price.")
            side = order['side']
            amount = order['remaining']
            new_order_id = place_limit_order(exchange, symbol, side, amount, new_price, tick_size)
            logger.info(f"Amended order {order_id} to new price {new_price}, new order {new_order_id}")
            return new_order_id
        else:
            logger.info(f"Order {order_id} already filled or closed.")
            return None
    except Exception as e:
        logger.error(f"Failed to amend order: {e}")
        return None

def close_position(exchange, symbol, position_size, tick_size):
    try:
        ticker = exchange.fetch_ticker(symbol)
        action_time = time.time()
        if position_size > 0:
            # Close long
            price = ticker['bid']
            print(f"[ACTION] Closing LONG position with SELL LIMIT order for {abs(position_size)} at {price}")
            logger.info(f"Closing LONG position with SELL order at {price}")
            order_id = place_limit_order(exchange, symbol, 'sell', abs(position_size), price, tick_size)
            elapsed = time.time() - action_time
            print(f"[TIMING] Time to CLOSE LONG after signal: {elapsed:.2f} seconds")
            logger.info(f"Time to CLOSE LONG after signal: {elapsed:.2f} seconds")
            logger.info(f"Closing LONG position with SELL order {order_id}")
            return order_id
        elif position_size < 0:
            # Close short
            price = ticker['ask']
            print(f"[ACTION] Closing SHORT position with BUY LIMIT order for {abs(position_size)} at {price}")
            logger.info(f"Closing SHORT position with BUY order at {price}")
            order_id = place_limit_order(exchange, symbol, 'buy', abs(position_size), price, tick_size)
            elapsed = time.time() - action_time
            print(f"[TIMING] Time to CLOSE SHORT after signal: {elapsed:.2f} seconds")
            logger.info(f"Time to CLOSE SHORT after signal: {elapsed:.2f} seconds")
            logger.info(f"Closing SHORT position with BUY order {order_id}")
            return order_id
        else:
            logger.info("No position to close.")
            return None
    except Exception as e:
        logger.error(f"Failed to close position: {e}")
        return None

def open_position(exchange, symbol, side, amount, tick_size):
    try:
        ticker = exchange.fetch_ticker(symbol)
        action_time = time.time()
        price = ticker['ask'] if side == 'buy' else ticker['bid']
        print(f"[ACTION] Opening {side.upper()} position with LIMIT order for {amount} at {price}")
        logger.info(f"Opening {side.upper()} position with order at {price}")
        order_id = place_limit_order(exchange, symbol, side, amount, price, tick_size)
        elapsed = time.time() - action_time
        print(f"[TIMING] Time to OPEN {side.upper()} after signal: {elapsed:.2f} seconds")
        logger.info(f"Time to OPEN {side.upper()} after signal: {elapsed:.2f} seconds")
        logger.info(f"Opening {side.upper()} position with order {order_id}")
        return order_id
    except Exception as e:
        logger.error(f"Failed to open position: {e}")
        return None

# --- Add this helper function to cancel TP orders for closed positions ---
def cancel_tp_orders(exchange, symbol, tp_order_ids):
    for order_id in tp_order_ids:
        try:
            exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled TP order {order_id}")
        except Exception as e:
            logger.error(f"Failed to cancel TP order {order_id}: {e}")

# --- Update place_mirror_tp_orders to retry on failure ---
def place_mirror_tp_orders(exchange, symbol, side, filled_size, bands, tick_size, tp_orders_state):
    if side == 'buy':
        tp_band_keys = ['+1', '+1.5', '+2', '+2.5', '+3']
        tp_side = 'sell'
    else:
        tp_band_keys = ['-1', '-1.5', '-2', '-2.5', '-3']
        tp_side = 'buy'
    tp_size = filled_size / len(tp_band_keys)
    tp_order_ids = []
    for k in tp_band_keys:
        price = bands[k]
        logger.info(f"Placing TP order at {price:.2f} for {tp_size} contracts (band {k}σ)")
        order_id = None
        for attempt in range(3):
            try:
                order_id = place_limit_order(exchange, symbol, tp_side, tp_size, price, tick_size)
                if order_id:
                    break
            except Exception as e:
                logger.error(f"Failed to place TP order at {price:.2f} (attempt {attempt+1}): {e}")
        if order_id:
            tp_order_ids.append(order_id)
        else:
            logger.error(f"Giving up on TP order at {price:.2f} after 3 attempts.")
    return tp_order_ids

# --- Main Application Logic ---
def main():
    logger.info("--- Starting robust in-and-out trading bot ---")
    config = load_config()
    if not config: return
    
    # Initialize exchange with better error handling
    exchange = initialize_exchange()
    if not exchange: return
    
    state = load_state()
    logger.info(f"Loaded initial state: {state}")

    symbol = config['symbol']
    timeframe = config['timeframe']
    sma_window = config['sma_window']
    order_size_contracts = config.get('order_size_contracts', 100)
    tick_size = config.get('tick_size', 0.5)
    signal_confirm_bars = config.get('signal_confirm_bars', 1)
    order_timeout = config.get('order_timeout', 30)  # seconds

    logger.info(f"Configuration: Trading {symbol} on {timeframe} with SMA window {sma_window}.")
    logger.info(f"Order size: {order_size_contracts} contracts. Tick size: {tick_size}")

    # Connection health check counter
    connection_errors = 0
    max_connection_errors = 5
    last_resync_time = time.time()

    try:
        while True:
            logger.info("--- New Cycle ---")
            
            # Resync time every 30 minutes to avoid clock drift issues
            if time.time() - last_resync_time > 1800:
                resync_time(exchange)
                last_resync_time = time.time()
            
            # 1. Fetch market data with retry logic
            try:
                max_retries = 3
                ohlcv = None
                for attempt in range(max_retries):
                    try:
                        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=sma_window + 5)
                        connection_errors = 0  # Reset error counter on success
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Retrying market data fetch (attempt {attempt + 1}): {e}")
                            time.sleep(5)
                        else:
                            raise e
                
                if ohlcv is None:
                    raise Exception("Failed to fetch market data after retries")
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
            except Exception as e:
                logger.error(f"Could not fetch market data: {e}")
                connection_errors += 1
                if connection_errors >= max_connection_errors:
                    logger.error("Too many connection errors. Reinitializing exchange...")
                    exchange = initialize_exchange()
                    if not exchange:
                        logger.error("Failed to reinitialize exchange. Exiting.")
                        return
                    connection_errors = 0
                time.sleep(60)
                continue

            # 2. Calculate signal
            signal, price, sma, bands = get_signal(df, sma_window)
            logger.info(f"Current Price: {price:.2f}, SMA({sma_window}): {sma:.2f}. Signal: {signal}")

            # 3. Signal confirmation logic
            if signal == state.get('last_signal'):
                state['signal_confirm_count'] += 1
            else:
                state['signal_confirm_count'] = 1
                state['last_signal'] = signal
            save_state(state)
            if state['signal_confirm_count'] < signal_confirm_bars:
                logger.info(f"Waiting for signal confirmation: {state['signal_confirm_count']}/{signal_confirm_bars}")
                time.sleep(60)
                continue

            # 4. Fetch position
            logger.info("Checking current position...")
            position_size = fetch_position(exchange, symbol)
            if position_size is None:
                time.sleep(60)
                continue
            state['position_size'] = position_size
            save_state(state)

            # 5. Order management
            open_orders = fetch_open_orders(exchange, symbol)
            if open_orders:
                logger.info(f"Found {len(open_orders)} open order(s). Managing...")
                for order in open_orders:
                    try:
                        # If order is too far from market, amend
                        ticker = exchange.fetch_ticker(symbol)
                        best_price = ticker['ask'] if order['side'] == 'buy' else ticker['bid']
                        price_diff = abs(order['price'] - best_price)
                        if price_diff > tick_size:
                            logger.info(f"Amending order {order['id']} from {order['price']} to {best_price}")
                            amend_order(exchange, symbol, order['id'], best_price, tick_size)
                        # If order is too old, cancel and replace
                        order_age = (pd.Timestamp.now() - pd.to_datetime(order['timestamp'], unit='ms')).total_seconds()
                        if order_age > order_timeout:
                            logger.info(f"Order {order['id']} is too old ({order_age:.2f}s), cancelling.")
                            exchange.cancel_order(order['id'], symbol)
                    except Exception as e:
                        logger.error(f"Error managing order {order.get('id', 'unknown')}: {e}")
                # Wait for all open orders to be cleared before proceeding
                logger.info("Waiting for all open orders to be cleared before placing new orders...")
                for _ in range(30):  # Wait up to 30 seconds
                    time.sleep(1)
                    open_orders = fetch_open_orders(exchange, symbol)
                    if not open_orders:
                        break
                else:
                    logger.warning("Open orders not cleared after 30 seconds. Skipping this cycle.")
                    continue
                # After open orders are cleared, re-fetch position to ensure it's updated
                for _ in range(10):  # Wait up to 10 seconds for position update
                    new_position = fetch_position(exchange, symbol)
                    if new_position != position_size:
                        logger.info(f"Position updated from {position_size} to {new_position} after clearing open orders.")
                        position_size = new_position
                        state['position_size'] = position_size
                        save_state(state)
                        break
                    time.sleep(1)
                else:
                    logger.warning("Position did not update after clearing open orders. Skipping this cycle.")
                    continue

            # 6. Main strategy logic (in-and-out)
            action_taken = False
            # Before placing a new order, double-check there are no open orders and position is as expected
            open_orders = fetch_open_orders(exchange, symbol)
            if open_orders:
                logger.warning("Open orders detected before placing a new order. Skipping this cycle.")
                continue
            confirmed_position = fetch_position(exchange, symbol)
            if confirmed_position != position_size:
                logger.warning(f"Position changed unexpectedly from {position_size} to {confirmed_position}. Skipping this cycle.")
                position_size = confirmed_position
                state['position_size'] = position_size
                save_state(state)
                continue

            if signal == 'long' and position_size <= 0:
                if position_size < 0:
                    logger.info("Reversing from SHORT to LONG.")
                    close_position(exchange, symbol, position_size, tick_size)
                    # Wait until position is flat
                    for _ in range(60):  # Wait up to 60 seconds
                        time.sleep(1)
                        new_position = fetch_position(exchange, symbol)
                        if new_position == 0:
                            break
                    else:
                        logger.warning("Position did not flatten after closing SHORT. Skipping open.")
                        continue
                # Laddered long orders at -3, -2.5, -2, -1.5, -1 bands
                band_keys = ['-3', '-2.5', '-2', '-1.5', '-1']
                suborder_size = order_size_contracts / len(band_keys)
                entry_order_ids = []
                for k in band_keys:
                    price = bands[k]
                    logger.info(f"Placing laddered LONG order at {price:.2f} for {suborder_size} contracts (band {k}σ)")
                    order_id = place_limit_order(exchange, symbol, 'buy', suborder_size, price, tick_size)
                    if order_id:
                        entry_order_ids.append(order_id)
                state['entry_order_ids'] = entry_order_ids
                state['tp_placed_for_entry'] = {}
                save_state(state)
                action_taken = True
            elif signal == 'short' and position_size >= 0:
                if position_size > 0:
                    logger.info("Reversing from LONG to SHORT.")
                    close_position(exchange, symbol, position_size, tick_size)
                    # Wait until position is flat
                    for _ in range(60):  # Wait up to 60 seconds
                        time.sleep(1)
                        new_position = fetch_position(exchange, symbol)
                        if new_position == 0:
                            break
                    else:
                        logger.warning("Position did not flatten after closing LONG. Skipping open.")
                        continue
                # Laddered short orders at +1, +1.5, +2, +2.5, +3 bands
                band_keys = ['+1', '+1.5', '+2', '+2.5', '+3']
                suborder_size = order_size_contracts / len(band_keys)
                entry_order_ids = []
                for k in band_keys:
                    price = bands[k]
                    logger.info(f"Placing laddered SHORT order at {price:.2f} for {suborder_size} contracts (band {k}σ)")
                    order_id = place_limit_order(exchange, symbol, 'sell', suborder_size, price, tick_size)
                    if order_id:
                        entry_order_ids.append(order_id)
                state['entry_order_ids'] = entry_order_ids
                state['tp_placed_for_entry'] = {}
                save_state(state)
                action_taken = True
            elif signal == 'flat' and position_size != 0:
                logger.info("Signal is FLAT, closing position.")
                close_position(exchange, symbol, position_size, tick_size)
                action_taken = True
            else:
                logger.info("No action needed. Signal and position are aligned.")

            # --- Bollinger Band Stop Loss Logic ---
            # Only run if there is an open position
            if position_size > 0:  # Long position
                if price <= bands.get('-5', float('-inf')):
                    logger.warning("HARD STOP LOSS triggered for LONG: price below -5σ. Closing all with market order.")
                    try:
                        exchange.create_market_sell_order(symbol, abs(position_size))
                        logger.info(f"Market SELL order placed for {abs(position_size)} contracts (HSL)")
                    except Exception as e:
                        logger.error(f"Failed to place market stop loss order: {e}")
                elif price <= bands.get('-4', float('-inf')):
                    logger.warning("SOFT STOP LOSS triggered for LONG: price below -4σ. Closing with limit orders.")
                    close_position(exchange, symbol, position_size, tick_size)
            elif position_size < 0:  # Short position
                if price >= bands.get('+5', float('inf')):
                    logger.warning("HARD STOP LOSS triggered for SHORT: price above +5σ. Closing all with market order.")
                    try:
                        exchange.create_market_buy_order(symbol, abs(position_size))
                        logger.info(f"Market BUY order placed for {abs(position_size)} contracts (HSL)")
                    except Exception as e:
                        logger.error(f"Failed to place market stop loss order: {e}")
                elif price >= bands.get('+4', float('inf')):
                    logger.warning("SOFT STOP LOSS triggered for SHORT: price above +4σ. Closing with limit orders.")
                    close_position(exchange, symbol, position_size, tick_size)

            # Check entry order fills and place TPs
            for entry_order_id in state.get('entry_order_ids', []):
                try:
                    order = exchange.fetch_order(entry_order_id, symbol)
                    filled = float(order.get('filled', 0))
                    already_tp = state.get('tp_filled_amount', {}).get(entry_order_id, 0)
                    new_fill = filled - already_tp
                    if new_fill > 0:
                        logger.info(f"Entry order {entry_order_id} new fill: {new_fill} contracts. Placing mirror TPs.")
                        tp_ids = place_mirror_tp_orders(exchange, symbol, order['side'], new_fill, bands, tick_size, {})
                        if 'tp_order_ids' not in state:
                            state['tp_order_ids'] = {}
                        if entry_order_id not in state['tp_order_ids']:
                            state['tp_order_ids'][entry_order_id] = []
                        state['tp_order_ids'][entry_order_id].extend(tp_ids)
                        if 'tp_filled_amount' not in state:
                            state['tp_filled_amount'] = {}
                        state['tp_filled_amount'][entry_order_id] = already_tp + new_fill
                        save_state(state)
                except Exception as e:
                    logger.error(f"Error checking entry order {entry_order_id}: {e}")

            # Dynamic TP updating for open positions
            for entry_order_id in state.get('entry_order_ids', []):
                tp_ids = state.get('tp_order_ids', {}).get(entry_order_id, [])
                if not tp_ids:
                    continue
                order = exchange.fetch_order(entry_order_id, symbol)
                filled = float(order.get('filled', 0))
                if filled == 0:
                    continue
                # Determine current TP band prices
                if order['side'] == 'buy':
                    tp_band_keys = ['+1', '+1.5', '+2', '+2.5', '+3']
                else:
                    tp_band_keys = ['-1', '-1.5', '-2', '-2.5', '-3']
                current_tp_prices = [bands[k] for k in tp_band_keys]
                last_tp_prices = state.get('tp_band_prices', {}).get(entry_order_id, [])
                # If any TP price has moved by more than tick_size, update TPs
                need_update = False
                if last_tp_prices and len(last_tp_prices) == len(current_tp_prices):
                    for old, new in zip(last_tp_prices, current_tp_prices):
                        if abs(old - new) > tick_size:
                            need_update = True
                            break
                else:
                    need_update = True
                if need_update:
                    logger.info(f"TP bands moved for entry {entry_order_id}. Updating TP orders.")
                    cancel_tp_orders(exchange, symbol, tp_ids)
                    tp_size = (state.get('tp_filled_amount', {}).get(entry_order_id, 0) or filled) / len(tp_band_keys)
                    new_tp_ids = []
                    for k in tp_band_keys:
                        price = bands[k]
                        for attempt in range(3):
                            try:
                                tp_id = place_limit_order(exchange, symbol, 'sell' if order['side']=='buy' else 'buy', tp_size, price, tick_size)
                                if tp_id:
                                    new_tp_ids.append(tp_id)
                                    break
                            except Exception as e:
                                logger.error(f"Failed to place updated TP order at {price:.2f} (attempt {attempt+1}): {e}")
                    state['tp_order_ids'][entry_order_id] = new_tp_ids
                    if 'tp_band_prices' not in state:
                        state['tp_band_prices'] = {}
                    state['tp_band_prices'][entry_order_id] = current_tp_prices
                    save_state(state)
                else:
                    # No update needed, but store the current prices if not already
                    if 'tp_band_prices' not in state:
                        state['tp_band_prices'] = {}
                    state['tp_band_prices'][entry_order_id] = current_tp_prices
                    save_state(state)

            if action_taken:
                time.sleep(5)
            else:
                time.sleep(60)

            # --- In main loop, after closing a position (flat signal or SL), cancel all TP orders ---
            if (signal == 'flat' and position_size != 0) or (position_size == 0):
                # Cancel all TP orders for all entry orders
                for tp_list in state.get('tp_order_ids', {}).values():
                    cancel_tp_orders(exchange, symbol, tp_list)
                state['tp_order_ids'] = {}
                state['tp_filled_amount'] = {}
                state['tp_placed_for_entry'] = {}
                save_state(state)

            # --- Placeholder for dynamic TP update logic ---
            # (You can add logic here to check if bands have moved significantly and reissue TP orders if needed)

    except KeyboardInterrupt:
        logger.info("Bot stopped manually. Cancelling all open orders...")
        cancel_all_orders(exchange, symbol)
        save_state(state)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        cancel_all_orders(exchange, symbol)
        save_state(state)

if __name__ == "__main__":
    main()