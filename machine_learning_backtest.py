import ccxt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import math


# ========== 参数设置 (参考Pine Script) ==========
TIMEFRAME = '1h'                   # K线周期：1小时
ATR_LEN = 10                       # ATR计算周期，对应Pine Script的ATR Length
SUPERTREND_FACTOR = 3.0            # SuperTrend因子，对应Pine Script的SuperTrend Factor
TRAINING_DATA_PERIOD = 100         # 聚类训练数据长度，对应Training Data Length
CLUSTERS = 3                       # 聚类类别数

# K-Means初始猜测参数（百分位数猜测，Pine Script中用的是0.75、0.5、0.25）
HIGH_VOL_GUESS = 0.75
MID_VOL_GUESS = 0.5
LOW_VOL_GUESS = 0.25

# 风控参数（与原代码一致，可根据实际情况调整）
RISK_PARAMS = {
    'stop_loss_pct': 0.03,    # 止损3%
    'take_profit_pct': 0.10,   # 固定止盈10%（备用）
    'max_drawdown': 0.15,     # 最大回撤15%
    'position_risk_pct': 0.02 # 每次仓位风险2%
}

# 额外参数：追踪止损和回调买入
TRAILING_STOP_PCT = 0.035    # 2%的追踪止损
PULLBACK_THRESHOLD = 0.01   # 1%的回调幅度作为买入信号

# ========== 核心策略基类 ==========
class AdaptiveSuperTrend:
    def __init__(self):
        # 初始化KMeans对象（初始时会重新训练）
        self.kmeans = KMeans(n_clusters=CLUSTERS)
    
    def calculate_atr(self, df, period=14):
        """计算ATR（平均真实波幅）"""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def train_clusters(self, atr_series):
        """
        利用最近TRAINING_DATA_PERIOD根K线的ATR数据训练K-Means模型。
        使用Pine Script中定义的初始猜测方法：根据训练期间ATR的最高值与最低值计算
        初始高、中、低波动率猜测值。
        """
        vol_upper = atr_series.max()
        vol_lower = atr_series.min()
        high_volatility = vol_lower + (vol_upper - vol_lower) * HIGH_VOL_GUESS
        medium_volatility = vol_lower + (vol_upper - vol_lower) * MID_VOL_GUESS
        low_volatility = vol_lower + (vol_upper - vol_lower) * LOW_VOL_GUESS

        # 注意：此处初始中心顺序设为 [low, medium, high]，与Pine Script计算结果一致
        initial_centers = np.array([[low_volatility],
                                    [medium_volatility],
                                    [high_volatility]])
        self.kmeans = KMeans(n_clusters=CLUSTERS, init=initial_centers, n_init=1)
        self.kmeans.fit(atr_series.values.reshape(-1, 1))
    
    def get_cluster_centers(self):
        """返回排序后的聚类中心（升序排列）"""
        return sorted(self.kmeans.cluster_centers_.flatten())
    
    def supertrend(self, df, atr_value, factor):
        """
        根据传入数据计算SuperTrend指标，返回最后一根K线的SuperTrend值和方向（1：多头，-1：空头）。
        逻辑与Pine Script代码类似。
        """
        hl2 = (df['high'] + df['low']) / 2
        upper = hl2 + factor * atr_value
        lower = hl2 - factor * atr_value

        # 初始化数组
        n = len(df)
        upper_bands = [np.nan] * n
        lower_bands = [np.nan] * n
        directions = [1] * n
        supertrend = [np.nan] * n

        # 设置初始值：以第一个K线的计算结果为初始值
        upper_bands[0] = upper.iloc[0]
        lower_bands[0] = lower.iloc[0]
        # 初始方向可根据需求设定，此处假设初始为多头，故SuperTrend取下轨
        supertrend[0] = lower_bands[0]
        
        for i in range(1, n):
            prev_upper = upper_bands[i-1]
            prev_lower = lower_bands[i-1]
            prev_close = df['close'].iloc[i-1]
            
            # 下轨逻辑：如果当前计算的下轨上移或前一根收盘低于前一根下轨，则更新下轨
            if lower.iloc[i] > prev_lower or prev_close < prev_lower:
                current_lower = lower.iloc[i]
            else:
                current_lower = prev_lower
                    
            # 上轨逻辑：如果当前计算的上轨下移或前一根收盘高于前一根上轨，则更新上轨
            if upper.iloc[i] < prev_upper or prev_close > prev_upper:
                current_upper = upper.iloc[i]
            else:
                current_upper = prev_upper
                    
            upper_bands[i] = current_upper
            lower_bands[i] = current_lower
            print(f"Current Upper={current_upper:.2f}, Lower={current_lower:.2f}, Close={df['close'].iloc[i]}")
                
            # 判断方向
            if np.isnan(supertrend[i-1]):
                directions[i] = -1
            elif supertrend[i-1] == prev_upper:
                directions[i] = 1 if df['close'].iloc[i] > current_upper else -1
            else:
                directions[i] = -1 if df['close'].iloc[i] < current_lower else 1
                    
            # 更新SuperTrend值
            supertrend[i] = current_lower if directions[i] == 1 else current_upper
            
        return supertrend[-1], directions[-1]


# ========== 回测专用策略类 ==========
class BacktestEnhancedSuperTrend(AdaptiveSuperTrend):
    def __init__(self, historical_data, initial_balance=10000):
        """
        historical_data: 包含历史K线数据的 DataFrame，必须包含['open','high','low','close','volume']，索引为时间戳
        initial_balance: 初始账户余额（USDT）
        """
        super().__init__()
        self.historical_data = historical_data.copy()
        self.balance = initial_balance  # 账户余额
        self.coin = 0                   # 持仓数量
        self.position = 0               # 0：空仓；1：持仓
        self.current_trend = None       # 当前趋势方向（1：多头；-1：空头）
        self.entry_price = None
        self.max_price_since_entry = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.trade_log = []             # 记录每笔交易
        self.equity_curve = []          # 记录账户权益变化（时间, 权益）
    
    def calculate_position_size(self, atr):
        """根据当前余额及ATR计算买入数量"""
        current_price = self.current_price
        risk_amount = self.balance * RISK_PARAMS['position_risk_pct']
        position_size = risk_amount / (atr * RISK_PARAMS['stop_loss_pct'])
        max_affordable = self.balance / current_price
        return min(position_size, max_affordable)
    
    def execute_trade(self, side, reason=None):
        """模拟交易执行，不调用真实交易所接口"""
        if reason:
            print(f"{self.current_time} - Trade Reason: {reason}")
        if side == 'buy':
            atr = self.current_atr
            size = self.calculate_position_size(atr)
            cost = size * self.current_price
            if cost > self.balance:
                print(f"{self.current_time} - Insufficient balance to buy")
                return
            print(f"{self.current_time} - Buy: Price {self.current_price:.2f}, Size {size:.4f}")
            self.balance -= cost
            self.coin += size
            self.position = 1
            self.entry_price = self.current_price
            self.max_price_since_entry = self.current_price
            self.stop_loss_price = self.current_price * (1 - RISK_PARAMS['stop_loss_pct'])
            self.take_profit_price = self.current_price * (1 + RISK_PARAMS['take_profit_pct'])
            self.record_trade('buy', self.current_price, size)
        elif side == 'sell':
            if self.coin <= 0:
                return
            proceeds = self.coin * self.current_price
            print(f"{self.current_time} - Sell: Price {self.current_price:.2f}, Size {self.coin:.4f}")
            self.balance += proceeds
            self.record_trade('sell', self.current_price, self.coin)
            self.coin = 0
            self.position = 0
            self.entry_price = None
            self.max_price_since_entry = None
    
    def record_trade(self, trade_type, price, quantity):
        """记录交易日志"""
        self.trade_log.append({
            'timestamp': self.current_time,
            'type': trade_type,
            'price': price,
            'quantity': quantity,
            'balance': self.get_total_equity()
        })
    
    def get_total_equity(self):
        """计算当前总权益（余额 + 持仓市值）"""
        return self.balance + self.coin * self.current_price
    
    def run_backtest(self):
        """逐根K线进行回测模拟"""
        df = self.historical_data.copy()
        # ATR使用ATR_LEN参数
        df['ATR'] = self.calculate_atr(df, ATR_LEN)
        df = df.dropna().copy()
        
        # 从足够的数据处开始模拟（保证有TRAINING_DATA_PERIOD + ATR_LEN根K线）
        start_index = TRAINING_DATA_PERIOD + ATR_LEN
        for i in range(start_index, len(df)):
            # 更新当前时间、价格和ATR
            self.current_time = df.index[i]
            self.current_price = df['close'].iloc[i]
            self.current_atr = df['ATR'].iloc[i]
            print(f"{self.current_time} - Current Price: {self.current_price:.2f}, ATR: {self.current_atr:.2f}")
            
            # 每4小时重新训练聚类模型（Pine Script中按bar_index进行训练，但此处用小时判断）
            if self.current_time.hour % 4 == 0:
                recent_atr = df['ATR'].iloc[max(0, i-TRAINING_DATA_PERIOD):i]
                if len(recent_atr) > 0:
                    self.train_clusters(recent_atr)
                centers = self.get_cluster_centers() if hasattr(self.kmeans, 'cluster_centers_') else [self.current_atr]*3
                cluster = self.kmeans.predict([[self.current_atr]])[0] if hasattr(self.kmeans, 'cluster_centers_') else 1
                selected_atr = centers[cluster]
            else:
                centers = self.get_cluster_centers() if hasattr(self.kmeans, 'cluster_centers_') else [self.current_atr]*3
                selected_atr = self.current_atr
            
            # 使用最近ATR_LEN根K线计算SuperTrend信号
            subset = df.iloc[max(0, i-ATR_LEN):i]
            st_value, direction = self.supertrend(subset, selected_atr, SUPERTREND_FACTOR)
            print(f"{self.current_time} - SuperTrend: {st_value:.2f}, Direction: {direction}")
            
            # 更新趋势：如果趋势发生变化则更新当前趋势标识
            if self.current_trend != direction:
                print(f"{self.current_time} - Trend changed to: {'Bullish' if direction==1 else 'Bearish'}")
                self.current_trend = direction
            
            # 【回调买入逻辑】：在多头趋势且空仓时，若当前价格比最近3根K线的最高价回调超过PULLBACK_THRESHOLD，则买入
            if self.current_trend == 1 and self.position == 0:
                recent_closes = df['close'].iloc[max(0, i-3):i]
                if len(recent_closes) > 0:
                    recent_peak = recent_closes.max()
                    if self.current_price < recent_peak * (1 - PULLBACK_THRESHOLD):
                        self.execute_trade('buy', reason="Pullback in Bullish trend")
            
            # 【趋势转空逻辑】：在空头趋势且持仓时，立即卖出
            if self.current_trend == -1 and self.position == 1:
                self.execute_trade('sell', reason="Bearish trend detected")
            
            # 【持仓时更新追踪止损】：若价格创出新高则更新追踪止损价；若当前价格低于止损价则平仓
            if self.position == 1:
                if self.current_price > self.max_price_since_entry:
                    self.max_price_since_entry = self.current_price
                    self.stop_loss_price = self.max_price_since_entry * (1 - TRAILING_STOP_PCT)
                    print(f"{self.current_time} - Update trailing stop loss to: {self.stop_loss_price:.2f}")
                if self.current_price <= self.stop_loss_price:
                    self.execute_trade('sell', reason="Trailing stop triggered")
            
            # 记录每根K线时的账户权益
            self.equity_curve.append((self.current_time, self.get_total_equity()))
        
        # 回测结束后输出结果
        print("\n=== Backtest Completed ===")
        print("Final Equity:", self.get_total_equity())
        print("Trade Log:")
        for trade in self.trade_log:
            print(trade)

# ========== 利用 CCXT 获取历史数据 ==========
def fetch_historical_data(symbol, timeframe, since, limit):
    """
    利用 CCXT 从 Binance 获取历史K线数据，
    返回 DataFrame，索引为时间戳，字段包括 open, high, low, close, volume
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df

# ========== 回测主函数 ==========
def main():
    symbol = 'BTC/USDT'
    timeframe = TIMEFRAME
    exchange = ccxt.binance({'enableRateLimit': True})
    limit = 1000  # 获取的K线数量
    # 计算起始时间：从当前时间向前推limit个K线周期（1h）
    since = exchange.milliseconds() - limit * 60 * 60 * 1000
    historical_data = fetch_historical_data(symbol, timeframe, since, limit)
    print("Historical data fetched:", historical_data.shape)
    
    backtester = BacktestEnhancedSuperTrend(historical_data, initial_balance=10000)
    backtester.run_backtest()

if __name__ == "__main__":
    main()
