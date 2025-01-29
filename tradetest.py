import ccxt
import pandas as pd
import time
import logging
import threading

# 配置日志
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# 配置交易所和参数
exchange_id = 'binance'
symbol = 'SUI/USDT'
timeframe = '1h'
capital = 1000
position_size = 0.1  # 每次交易的资金比例 (10%)
take_profit_ratio = 0.15  # 止盈比例 (15%)
stop_loss_ratio = 0.05    # 止损比例 (5%)

# 交易状态变量
is_position_open = False
monitor_running = False  # 监控线程运行状态

# 初始化 Binance 交易所
exchange = getattr(ccxt, exchange_id)({
    'apiKey': 'apiKey',         # 替换为你的API Key
    'secret': 'secret',         # 替换为你的API Secret
})
exchange.set_sandbox_mode(True)  # enable sandbox mode
exchange.load_markets()

def fetch_balances():
    """查询 BTC、USDT、ETH、SUI 余额，并打印/记录日志"""
    try:
        balances = exchange.fetch_balance()
        assets_to_check = ["BTC", "USDT", "ETH", "SUI"]
        
        balance_info = {asset: float(balances['total'].get(asset, 0.0)) for asset in assets_to_check}

        print("\n=== 当前账户余额 ===")
        logger.info("=== 当前账户余额 ===")
        for asset, amount in balance_info.items():
            print(f"{asset}: {amount:.6f}")
            logger.info(f"{asset}: {amount:.6f}")

        return balance_info
    except Exception as e:
        logger.error(f"Error fetching balances: {e}")
        print(f"Error fetching balances: {e}")
        return None


def fetch_data(symbol, timeframe, limit=50):
    """获取历史K线数据"""
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"Fetched {len(df)} rows of data for {symbol} with timeframe {timeframe}")
        logger.info(f"Fetched {len(df)} rows of data for {symbol} with timeframe {timeframe}")
        return df
    except Exception as e:
        logger.error(f"❌ 获取市场数据失败: {e}")
        return None
    
def fetch_current_price(symbol):
    """查询当前交易对的实时价格"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        print(f"📈 当前 {symbol} 价格: {current_price:.6f}")
        logger.info(f"📈 当前 {symbol} 价格: {current_price:.6f}")
        return current_price
    except Exception as e:
        logger.error(f"❌ 获取 {symbol} 价格失败: {e}")
        return None


def calculate_bollinger_bands(df, window=20, num_std_dev=2):
    """计算布林带上下限"""
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['ma'] + num_std_dev * df['std']
    df['lower_band'] = df['ma'] - num_std_dev * df['std']
    
    if len(df) > 0:
        last_row = df.iloc[-1]
        print(f"Current Bollinger Bands for {symbol}: Upper={last_row['upper_band']:.2f}, Lower={last_row['lower_band']:.2f}")
        logger.info(f"Current Bollinger Bands for {symbol}: Upper={last_row['upper_band']:.2f}, Lower={last_row['lower_band']:.2f}")
    return df


def calculate_rsi(df, window=14):
    """计算RSI指标"""
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    if len(df) > 0:
        last_row = df.iloc[-1]
        print(f"Current RSI for {symbol}: RSI={last_row['rsi']:.2f}")
        logger.info(f"Current RSI for {symbol}: RSI={last_row['rsi']:.2f}")
    return df


def check_signals(df):
    """检查买入或卖出信号"""
    last_row = df.iloc[-1]
    if last_row['close'] < last_row['lower_band'] and last_row['rsi'] < 30:
        print("Buy signal detected")
        logger.info("Buy signal detected")
        return 'buy'
    elif last_row['close'] > last_row['upper_band'] and last_row['rsi'] > 70:
        print("Sell signal detected")
        logger.info("Sell signal detected")
        return 'sell'
    return None


def execute_trade(signal, symbol, capital, position_size):
    """执行买入或卖出交易"""
    global is_position_open, monitor_running
    try:
        if signal == 'buy' and not is_position_open:
            amount = (capital * position_size) / exchange.fetch_ticker(symbol)['last']
            order = exchange.create_market_buy_order(symbol, amount)
            print(f"BUY Order Executed: {order}")
            logger.info(f"BUY Order Executed: {order}")
            is_position_open = True
            monitor_running = True  # 开启监控线程
            return order
        elif signal == 'sell' and is_position_open:
            balance = exchange.fetch_balance()
            amount = balance['total'].get(symbol.split('/')[0], 0)
            order = exchange.create_market_sell_order(symbol, amount)
            print(f"SELL Order Executed: {order}")
            logger.info(f"SELL Order Executed: {order}")
            is_position_open = False
            monitor_running = False  # 关闭监控线程
            return order
    except Exception as e:
        print(f"Error executing trade: {e}")
        logger.error(f"Error executing trade: {e}")
    return None


def monitor_trade(entry_price, stop_loss_ratio, take_profit_ratio, symbol):
    """监控持仓，设置止盈止损"""
    global is_position_open, monitor_running

    while is_position_open and monitor_running:
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            logger.info(f'📉 当前 {symbol} 价格: {current_price}')

            if current_price >= entry_price * (1 + take_profit_ratio):
                print(f'🎯 触发止盈: {current_price}')
                logger.info(f"🎯 触发止盈: {current_price}")
                execute_trade('sell', symbol, capital, position_size)
                break
            elif current_price <= entry_price * (1 - stop_loss_ratio):
                print(f'⛔ 触发止损: {current_price}')
                logger.info(f"⛔ 触发止损: {current_price}")
                execute_trade('sell', symbol, capital, position_size)
                break
        except Exception as e:
            print(f"⚠️ 监控交易错误: {e}")
            logger.error(f"⚠️ 监控交易错误: {e}")

        time.sleep(10)

    monitor_running = False
    print("✅ 监控线程已关闭")
    logger.info("✅ 监控线程已关闭")


if __name__ == "__main__":
    print("🚀 启动交易机器人...")
    logger.info("🚀 启动交易机器人...")


    while True:
        try:
            logger.info("📊 交易循环仍在运行...")

            fetch_balances()

            data = fetch_data(symbol, timeframe)
            if data is None:
                time.sleep(60)
                continue

            data = calculate_bollinger_bands(data)
            data = calculate_rsi(data)
            signal = check_signals(data)

            if signal == 'buy' and not is_position_open:
                entry_order = execute_trade(signal, symbol, capital, position_size)
                if entry_order:
                    threading.Thread(target=monitor_trade, args=(entry_order['price'], stop_loss_ratio, take_profit_ratio, symbol)).start()

            elif signal == 'sell' and is_position_open:
                execute_trade(signal, symbol, capital, position_size)

            for _ in range(60):
                fetch_current_price(symbol)
                time.sleep(60)
                print("⏳ 交易机器人仍在运行...")
                logger.info("⏳ 交易机器人仍在运行...")
        except Exception as e:
            print(f"❌ 交易循环错误: {e}")
            logger.error(f"❌ 交易循环错误: {e}")
            time.sleep(60)
