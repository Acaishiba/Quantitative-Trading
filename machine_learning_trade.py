import ccxt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from time import sleep
from datetime import datetime
import math

# ========== 交易所配置 ==========
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})
#exchange.set_sandbox_mode(True)  # enable sandbox mode
exchange.load_markets()

# ========== 策略参数 ==========
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'

# 核心参数
ATR_PERIOD = 10
SUPERTREND_FACTOR = 3.0
TRAINING_WINDOW = 100
CLUSTERS = 3  # 高/中/低波动

# 风控参数
RISK_PARAMS = {
    'stop_loss_pct': 0.03,    # 止损3%
    'take_profit_pct': 0.10,   # 固定止盈10%（备用）
    'max_drawdown': 0.15,     # 最大回撤15%
    'position_risk_pct': 0.02 # 每次仓位风险2%
}

# 新增参数
TRAILING_STOP_PCT = 0.035    # 2% 的追踪止损比例
PULLBACK_THRESHOLD = 0.01    # 1% 的回调幅度作为入场信号

# ========== 核心策略基类 ==========
class AdaptiveSuperTrend:
    def __init__(self):
        self.data = pd.DataFrame()
        self.kmeans = KMeans(n_clusters=CLUSTERS)
        self.current_trend = None
        self.position = 0  # 0: 无持仓, 1: 多头

    def fetch_data(self, limit=500):
        """从交易所获取K线数据"""
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')

    def calculate_atr(self, df, period=14):
        """计算平均真实波幅"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def train_clusters(self, atr_series):
        """改进的聚类初始化方法"""
        # 使用分位数初始化
        high_vol = np.quantile(atr_series, 0.75)
        mid_vol = np.quantile(atr_series, 0.5)
        low_vol = np.quantile(atr_series, 0.25)
        
        # 使用K-Means++初始化
        self.kmeans = KMeans(n_clusters=3, init=np.array([
            [low_vol], 
            [mid_vol], 
            [high_vol]
        ]), n_init=1)
        
        self.kmeans.fit(atr_series.values.reshape(-1, 1))
        
    def get_cluster_centers(self):
        """获取排序后的聚类中心"""
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

# ========== 增强策略子类 ==========
class EnhancedSuperTrend(AdaptiveSuperTrend):
    def __init__(self):
        super().__init__()
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.max_price_since_entry = None  # 新增：记录持仓后的最高价格
        self.equity_curve = []
        self.max_equity = 0
        
        # 初始化交易日志
        self.trade_log = pd.DataFrame(columns=[
            'timestamp', 'type', 'price', 'quantity',
            'stop_loss', 'take_profit', 'balance'
        ])

    def calculate_position_size(self, atr):
        # 获取账户和交易对信息
        balance = exchange.fetch_balance()['USDT']['free']
        market = exchange.market(SYMBOL)
        min_amount = market['limits']['amount']['min']
        precision = market['precision']['amount']
        decimals = abs(int(round(math.log10(precision))))
        
        # 计算理论仓位
        risk_amount = balance * RISK_PARAMS['position_risk_pct']
        position_size = risk_amount / (atr * RISK_PARAMS['stop_loss_pct'])
        
        # 获取当前价格
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        
        # 实际可买数量限制
        max_affordable = (balance * 0.99) / current_price  # 保留1%作为手续费
        position_size = min(position_size, max_affordable)
        
        # 遵守交易所规则
        position_size = max(min_amount, position_size)  # 不低于最小交易量
        position_size = round(position_size, decimals) # 精度处理
        
        return position_size

    def update_drawdown(self):
        """实时回撤监控"""
        current_equity = self.get_total_equity()
        self.equity_curve.append(current_equity)
        self.max_equity = max(self.max_equity, current_equity)
        
        drawdown = (self.max_equity - current_equity) / self.max_equity
        if drawdown > RISK_PARAMS['max_drawdown']:
            print(f"⚠️ 最大回撤触发：{drawdown*100:.1f}% > {RISK_PARAMS['max_drawdown']*100}%")
            if self.position == 1:
                self.execute_trade('sell', reason="Max Drawdown")
            return False
        return True

    def get_total_equity(self):
        """计算总资产价值"""
        balance = exchange.fetch_balance()
        ticker = exchange.fetch_ticker(SYMBOL)
        usdt_balance = balance['USDT']['free']
        coin_balance = balance[SYMBOL.split('/')[0]]['free'] * ticker['last']
        return usdt_balance + coin_balance

    def execute_trade(self, side, reason=None):
        """执行交易操作"""
        try:
            current_price = exchange.fetch_ticker(SYMBOL)['last']
            if reason:
                print(f"交易原因: {reason}")
            if side == 'buy':
                atr = self.data['ATR'].iloc[-1]
                position_size = self.calculate_position_size(atr)
                print(f"📈 买入信号 | 价格: {current_price:.2f}")
                exchange.create_order(SYMBOL, 'market', 'buy', position_size)
                self.position = 1
                self.entry_price = current_price
                self.max_price_since_entry = current_price  # 初始化最高价格为入场价
                self.stop_loss_price = current_price * (1 - RISK_PARAMS['stop_loss_pct'])
                self.take_profit_price = current_price * (1 + RISK_PARAMS['take_profit_pct'])
                self.record_trade('buy', current_price, position_size)
                
            elif side == 'sell':
                coin_balance = exchange.fetch_balance()[SYMBOL.split('/')[0]]['free']
                if coin_balance <= 0:
                    print("无可卖出的仓位")
                    return
                print(f"📉 卖出信号 | 价格: {current_price:.2f}")
                exchange.create_order(SYMBOL, 'market', 'sell', coin_balance)
                self.position = 0
                self.entry_price = None
                self.max_price_since_entry = None
                self.record_trade('sell', current_price, coin_balance)
                
        except Exception as e:
            print(f"❌ 交易失败: {str(e)}")

    def check_risk_limits(self):
        """风险规则检查：包括止损、止盈和追踪止损"""
        if self.position == 1:
            current_price = exchange.fetch_ticker(SYMBOL)['last']
            # 追踪止损：若当前价格下破追踪止损价则平仓
            if current_price <= self.stop_loss_price:
                self.execute_trade('sell', reason="Trailing Stop Triggered")
                print("🔴 追踪止损触发")
            # 固定止盈检查（如有需要可同时保留）
            elif current_price >= self.take_profit_price:
                self.execute_trade('sell', reason="Take Profit Triggered")
                print("🟢 止盈触发")

    def record_trade(self, trade_type, price, quantity):
        """记录交易到CSV"""
        new_trade = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': trade_type,
            'price': price,
            'quantity': quantity,
            'stop_loss': self.stop_loss_price if trade_type == 'buy' else None,
            'take_profit': self.take_profit_price if trade_type == 'buy' else None,
            'balance': self.get_total_equity()
        }])
        new_trade.to_csv('trading_log.csv', mode='a', header=False, index=False)

    def analyze_performance(self):
        """增强版绩效分析"""
        try:
            import os
            if not os.path.exists('trading_log.csv'):
                print("⚠️ 无交易日志文件")
                return

            # 强制类型转换 + 处理空值
            trades = pd.read_csv('trading_log.csv', 
                dtype={
                    'type': 'category',
                    'price': float,
                    'quantity': float,
                    'balance': float
                },
                parse_dates=['timestamp']
            ).dropna(subset=['type', 'balance'])

            if len(trades) < 2:
                print("⏳ 交易数据不足")
                return

            # 计算胜率（仅比较完整交易对）
            buy_mask = trades['type'] == 'buy'
            sell_mask = trades['type'] == 'sell'
            
            # 确保买卖交替出现
            valid_pairs = []
            buy_index = -1
            for i, row in trades.iterrows():
                if row['type'] == 'buy':
                    buy_index = i
                elif row['type'] == 'sell' and buy_index != -1:
                    valid_pairs.append( (buy_index, i) )
                    buy_index = -1

            # 统计盈利交易
            winning = 0
            for buy, sell in valid_pairs:
                if trades.iloc[sell]['balance'] > trades.iloc[buy]['balance']:
                    winning += 1
            win_rate = winning / len(valid_pairs) if len(valid_pairs) > 0 else 0

            # 夏普比率计算
            returns = trades['balance'].pct_change().dropna()
            if len(returns) < 2:
                print("⚠️ 收益数据不足")
                return
                
            sharpe = returns.mean() / returns.std() * np.sqrt(365*24)

            # 最大回撤计算
            equity = trades['balance'].values
            max_dd = 0
            peak = equity[0]
            for value in equity:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)

            print("\n=== 策略表现报告 ===")
            print(f"有效交易对: {len(valid_pairs)}")
            print(f"胜率: {win_rate:.2%}")
            print(f"夏普比率: {sharpe:.2f}")
            print(f"最大回撤: {max_dd:.2%}")

        except Exception as e:
            print(f"性能分析异常: {str(e)}")
            import traceback
            traceback.print_exc()

    def run_strategy(self):
        """策略主循环（改进版：每5分钟检查一次）"""
        print("🚀 启动自适应SuperTrend策略")
        last_training = None
        
        while True:
            try:
                # 获取数据（基于1小时K线数据）
                df = self.fetch_data(limit=TRAINING_WINDOW+ATR_PERIOD)
                if len(df) < TRAINING_WINDOW:
                    print("⏳ 等待更多数据...")
                    sleep(300)
                    continue
                
                # 计算ATR
                atr_series = self.calculate_atr(df, ATR_PERIOD)
                df['ATR'] = atr_series
                self.data = df.dropna()
                
                # 每4小时重新训练模型
                current_hour = datetime.now().hour
                if last_training is None or (current_hour % 4 == 0 and last_training != current_hour):
                    self.train_clusters(atr_series[-TRAINING_WINDOW:])
                    centers = self.get_cluster_centers()
                    print(f"\n🔁 更新波动率聚类中心：\n"
                          f"低波动: {centers[0]:.2f}\n"
                          f"中波动: {centers[1]:.2f}\n"
                          f"高波动: {centers[2]:.2f}")
                    last_training = current_hour
                
                # 根据当前ATR选择波动率等级
                current_atr = atr_series.iloc[-1]
                cluster = self.kmeans.predict([[current_atr]])[0]
                centers = self.get_cluster_centers()
                selected_atr = centers[cluster]
                
                # 使用最新ATR_PERIOD的K线生成SuperTrend信号
                st_value, direction = self.supertrend(df.iloc[-ATR_PERIOD:], selected_atr, SUPERTREND_FACTOR)
                
                # 当趋势发生变化时，更新当前趋势标识
                if self.current_trend != direction:
                    print(f"趋势变化检测: 新趋势 {'多头' if direction==1 else '空头'}")
                    self.current_trend = direction
                
                # ★【新增】回调买入逻辑：在多头趋势中，若当前没有持仓且发现价格出现短暂回调，则买入
                if self.current_trend == 1 and self.position == 0:
                    ticker = exchange.fetch_ticker(SYMBOL)
                    current_price = ticker['last']
                    print(f"当前价格: {current_price:.2f}")
                    # 取最近3根K线收盘价作为参考
                    recent_closes = df['close'].iloc[-3:]
                    recent_peak = recent_closes.max()
                    if current_price < recent_peak * (1 - PULLBACK_THRESHOLD):
                        print(f"检测到上行趋势中的回调: 当前价格 {current_price:.2f} 距离最近高点 {recent_peak:.2f} 下跌超过 {PULLBACK_THRESHOLD*100:.1f}%")
                        self.execute_trade('buy')
                
                # ★【新增】趋势转空时，若有持仓，则立即卖出
                if self.current_trend == -1 and self.position == 1:
                    print("检测到空头趋势，准备平多仓")
                    self.execute_trade('sell')
                
                # ★【新增】持仓时更新追踪止损：
                if self.position == 1:
                    ticker = exchange.fetch_ticker(SYMBOL)
                    current_price = ticker['last']
                    if self.max_price_since_entry is None or current_price > self.max_price_since_entry:
                        self.max_price_since_entry = current_price
                        # 更新追踪止损价：以最高价下跌TRAILING_STOP_PCT比例作为止损价
                        self.stop_loss_price = self.max_price_since_entry * (1 - TRAILING_STOP_PCT)
                        print(f"更新追踪止损价: {self.stop_loss_price:.2f} (最高价 {self.max_price_since_entry:.2f})")
                
                # 每5分钟检查风险限制（包括追踪止损和固定止盈）
                self.check_risk_limits()
                self.update_drawdown()
                
                # 每小时整点分析一次策略表现
                if datetime.now().minute == 0:
                    self.analyze_performance()
                
                sleep(300)  # 每5分钟运行一次
                
            except Exception as e:
                print(f"策略异常: {str(e)}")
                sleep(60)

# ========== 启动策略 ==========
if __name__ == "__main__":
    strategy = EnhancedSuperTrend()
    try:
        strategy.run_strategy()
    except KeyboardInterrupt:
        print("\n🛑 手动停止策略")
    finally:
        strategy.analyze_performance()
        print("✅ 策略已安全停止")
