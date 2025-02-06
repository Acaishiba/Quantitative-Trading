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
SYMBOL = 'SUI/USDT'
TIMEFRAME = '1h'

# 核心参数
ATR_PERIOD = 10
SUPERTREND_FACTOR = 3.0
TRAINING_WINDOW = 100
CLUSTERS = 3  # 高/中/低波动

# 风控参数
RISK_PARAMS = {
    'stop_loss_pct': 0.05,
    'take_profit_pct': 1.0,
    'max_drawdown': 0.15,
    'position_risk_pct': 0.02
}

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
        """训练波动率聚类模型"""
        self.kmeans.fit(atr_series.values.reshape(-1, 1))
        
    def get_cluster_centers(self):
        """获取排序后的聚类中心"""
        return sorted(self.kmeans.cluster_centers_.flatten())

    def supertrend(self, df, atr_value, factor):
        """生成SuperTrend信号"""
        hl2 = (df['high'] + df['low']) / 2
        upper = hl2 + factor * atr_value
        lower = hl2 - factor * atr_value
        
        st = [np.nan] * len(df)
        direction = [1] * len(df)  # 1=多头，-1=空头
        
        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            
            # 更新趋势方向
            if df['close'].iloc[i] > upper.iloc[i-1]:
                direction[i] = 1
            elif df['close'].iloc[i] < lower.iloc[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            # 更新SuperTrend值
            st[i] = lower.iloc[i] if direction[i] == 1 else upper.iloc[i]
            
        return st[-1], direction[-1]

# ========== 增强策略子类 ==========
class EnhancedSuperTrend(AdaptiveSuperTrend):
    def __init__(self):
        super().__init__()
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
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
                self.sell(reason="Max Drawdown")
            return False
        return True

    def get_total_equity(self):
        """计算总资产价值"""
        balance = exchange.fetch_balance()
        ticker = exchange.fetch_ticker(SYMBOL)
        usdt_balance = balance['USDT']['free']
        coin_balance = balance[SYMBOL.split('/')[0]]['free'] * ticker['last']
        return usdt_balance + coin_balance

    def execute_trade(self, side):
        """执行交易操作"""
        try:
            current_price = exchange.fetch_ticker(SYMBOL)['last']
            print(side)
            
            if side == 'buy':
                atr = self.data['ATR'].iloc[-1]
                position_size = self.calculate_position_size(atr)
                print(f"📈 买入信号 | 价格: {current_price:.2f}")
                exchange.create_order(SYMBOL, 'market', 'buy', position_size)
                self.position = 1
                self.entry_price = current_price
                self.stop_loss_price = current_price * (1 - RISK_PARAMS['stop_loss_pct'])
                self.take_profit_price = current_price * (1 + RISK_PARAMS['take_profit_pct'])
                self.record_trade('buy', current_price, position_size)
                
            elif side == 'sell':
                coin_balance = exchange.fetch_balance()[SYMBOL.split('/')[0]]['free']
                print(f"📉 卖出信号 | 价格: {current_price:.2f}")
                exchange.create_order(SYMBOL, 'market', 'sell', coin_balance)
                self.position = 0
                self.entry_price = None
                self.record_trade('sell', current_price, coin_balance)
                
        except Exception as e:
            print(f"❌ 交易失败: {str(e)}")

    def check_risk_limits(self):
        """风险规则检查"""
        if self.position == 1:
            current_price = exchange.fetch_ticker(SYMBOL)['last']
            if current_price <= self.stop_loss_price:
                self.execute_trade('sell')
                print("🔴 止损触发")
            elif current_price >= self.take_profit_price:
                self.execute_trade('sell')
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
        """策略绩效分析"""
        try:
            trades = pd.read_csv('trading_log.csv')
            if len(trades) < 2:
                return

            # 计算胜率
            winning = trades[trades['type'] == 'sell']['balance'] > trades['balance'].shift(1)
            win_rate = winning.mean()
            
            # 计算夏普比率
            returns = trades['balance'].pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(365*24)
            
            # 最大回撤
            equity = trades['balance'].values
            peak = equity[0]
            max_dd = 0
            for value in equity:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)

            print("\n=== 策略表现报告 ===")
            print(f"累计收益: {equity[-1]/equity[0]-1:.2%}")
            print(f"胜率: {win_rate:.2%}")
            print(f"夏普比率: {sharpe:.2f}")
            print(f"最大回撤: {max_dd:.2%}")
            print(f"总交易次数: {len(trades)//2}")
            
        except Exception as e:
            print(f"性能分析失败: {str(e)}")

    def run_strategy(self):
        """策略主循环"""
        print("🚀 启动自适应SuperTrend策略")
        last_training = None
        
        while True:
            try:
                # 获取数据
                df = self.fetch_data(limit=TRAINING_WINDOW+ATR_PERIOD)
                if len(df) < TRAINING_WINDOW:
                    print("⏳ 等待更多数据...")
                    sleep(60)
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
                
                # 选择当前波动率等级
                current_atr = atr_series.iloc[-1]
                cluster = self.kmeans.predict([[current_atr]])[0]
                selected_atr = centers[cluster]
                
                # 生成SuperTrend信号
                st_value, direction = self.supertrend(df.iloc[-ATR_PERIOD:], selected_atr, SUPERTREND_FACTOR)
                
                # 趋势变化检测
                if self.current_trend != direction:
                    action = '买入' if direction == 1 else '卖出'
                    print(f"\n🔄 趋势变化检测 ({action})")
                    print(f"当前价格: {df['close'].iloc[-1]:.2f}")
                    print(f"ATR值: {current_atr:.2f} ({['低','中','高'][cluster]}波动)")
                    self.execute_trade('buy' if direction == 1 else 'sell')
                    self.current_trend = direction
                
                # 每小时检查风险
                self.check_risk_limits()
                self.update_drawdown()
                
                # 每小时保存日志
                if datetime.now().minute == 0:
                    self.analyze_performance()
                
                sleep(3600)  # 每小时运行
                
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