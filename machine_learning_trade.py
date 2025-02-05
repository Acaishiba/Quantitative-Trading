import ccxt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from time import sleep
from datetime import datetime

# ========== 配置部分 ==========
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# 策略参数
SYMBOL = 'BTC/USDT'
RISK_PARAMS = {  # ✅ 风控参数
    'stop_loss_pct': 0.05,    # 止损5%
    'take_profit_pct': 1.0,   # 止盈100%
    'max_drawdown': 0.15,     # 最大回撤15%
    'position_risk_pct': 0.02 # 每笔交易风险2%
}
ATR_PERIOD = 10
SUPERTREND_FACTOR = 3.0
TRAINING_WINDOW = 100  # 聚类训练数据长度
CLUSTERS = 3  # 高/中/低波动

# ========== 基类定义 ==========
class AdaptiveSuperTrend:
    def __init__(self):
        self.data = pd.DataFrame()
        self.kmeans = KMeans(n_clusters=CLUSTERS)
        self.current_trend = None
        self.position = 0  # 0: 无持仓, 1: 多头

    def fetch_data(self, limit=500):
        """从交易所获取OHLCV数据"""
        ohlcv = exchange.fetch_ohlcv(SYMBOL, '1h', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')

    def calculate_atr(self, df, period=14):
        """计算ATR"""
        high, low, close = df['high'], df['low'], df['close']
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def train_clusters(self, atr_series):
        """训练K-Means聚类模型"""
        self.kmeans.fit(atr_series.values.reshape(-1, 1))
        
    def get_cluster_centers(self):
        """获取聚类中心并排序（低->高）"""
        return sorted(self.kmeans.cluster_centers_.flatten())

    def supertrend(self, df, atr_value, factor):
        """计算SuperTrend信号"""
        hl2 = (df['high'] + df['low']) / 2
        upper = hl2 + factor * atr_value
        lower = hl2 - factor * atr_value
        
        st = [np.nan]*len(df)
        direction = [1]*len(df)  # 1=上升趋势
        
        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            
            # 更新上下轨
            if df['close'].iloc[i] > upper.iloc[i-1]:
                direction[i] = 1
            elif df['close'].iloc[i] < lower.iloc[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            st[i] = lower.iloc[i] if direction[i] == 1 else upper.iloc[i]
            
        return st[-1], direction[-1]

# ========== 核心策略类 ==========
class EnhancedSuperTrend(AdaptiveSuperTrend):
    def __init__(self):
        super().__init__()
        # ✅ 初始化风控模块
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.equity_curve = []
        self.max_equity = 0
        self.trade_log = pd.DataFrame(columns=[
            'timestamp', 'type', 'price', 'quantity', 
            'stop_loss', 'take_profit', 'balance'
        ])
        
    # ✅ 动态仓位计算（凯利公式改进版）
    def calculate_position_size(self, atr):
        balance = exchange.fetch_balance()['USDT']['free']
        risk_amount = balance * RISK_PARAMS['position_risk_pct']
        return risk_amount / (atr * RISK_PARAMS['stop_loss_pct'])

    def update_drawdown(self):
        """更新最大回撤监控"""
        current_equity = self.get_total_equity()
        self.equity_curve.append(current_equity)
        
        # 更新峰值资产
        self.max_equity = max(self.max_equity, current_equity)
        
        # 计算当前回撤
        drawdown = (self.max_equity - current_equity) / self.max_equity
        if drawdown > RISK_PARAMS['max_drawdown']:
            print(f"⚠️ 最大回撤触发：{drawdown*100:.1f}% > {RISK_PARAMS['max_drawdown']*100}%")
            if self.position == 1:
                self.sell( reason="Max Drawdown")
            return False  # 暂停策略
        return True

    def get_total_equity(self):
        """计算总资产（法币价值）"""
        balance = exchange.fetch_balance()
        ticker = exchange.fetch_ticker(SYMBOL)
        return (
            balance['USDT']['free'] 
            + balance[SYMBOL.split('/')[0]]['free'] * ticker['last']
        )

    # ✅ 增强买卖逻辑
    def buy(self):
        try:
            current_price = exchange.fetch_ticker(SYMBOL)['last']
            
            # 动态仓位计算
            atr = self.data['ATR'].iloc[-1]
            position_size = self.calculate_position_size(atr)
            
            # 执行买入
            print(f"📈 买入信号 | 价格: {current_price:.2f}")
            exchange.create_order(SYMBOL, 'market', 'buy', position_size)
            
            # 记录交易参数
            self.entry_price = current_price
            self.stop_loss_price = current_price * (1 - RISK_PARAMS['stop_loss_pct'])
            self.take_profit_price = current_price * (1 + RISK_PARAMS['take_profit_pct'])
            self.position = 1
            
            # ✅ 记录交易日志
            self.record_trade('buy', current_price, position_size)
            
        except Exception as e:
            print(f"❌ 买入失败: {str(e)}")

    def sell(self, reason="Signal"):
        try:
            current_price = exchange.fetch_ticker(SYMBOL)['last']
            coin_balance = exchange.fetch_balance()[SYMBOL.split('/')[0]]['free']
            
            # 执行卖出
            print(f"📉 卖出信号 [{reason}] | 价格: {current_price:.2f}")
            exchange.create_order(SYMBOL, 'market', 'sell', coin_balance)
            
            # 重置状态
            self.position = 0
            self.entry_price = None
            
            # ✅ 记录交易日志
            self.record_trade('sell', current_price, coin_balance)
            
        except Exception as e:
            print(f"❌ 卖出失败: {str(e)}")

    def check_risk_limits(self):
        """实时风险检查"""
        if self.position == 1:
            current_price = exchange.fetch_ticker(SYMBOL)['last']
            
            # ✅ 止损/止盈检查
            if current_price <= self.stop_loss_price:
                self.sell(reason="Stop Loss")
            elif current_price >= self.take_profit_price:
                self.sell(reason="Take Profit")

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
        
        # 追加到CSV文件
        new_trade.to_csv('trading_log.csv', mode='a', header=False, index=False)
        
    def calculate_performance(self):
        """计算性能指标"""
        trades = pd.read_csv('trading_log.csv')
        
        if len(trades) < 2:
            print("⚠️ 尚无足够交易数据")
            return
        
        # ✅ 计算胜率
        winning_trades = trades[trades['type'] == 'sell']['balance'] > trades['balance'].shift(1)
        win_rate = winning_trades.mean()
        
        # ✅ 计算夏普比率
        returns = trades['balance'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365*24)  # 按小时计算
        
        # ✅ 最大回撤
        equity = trades['balance'].values
        max_drawdown = 0
        peak = equity[0]
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_drawdown:
                max_drawdown = dd
                
        print("\n=== 策略表现分析 ===")
        print(f"总交易次数: {len(trades)//2}")
        print(f"胜率: {win_rate:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"最终余额: ${equity[-1]:.2f}")

    def execute_strategy(self):
        """增强版策略循环"""
        print("启动增强版策略...")
        while True:
            try:
                # 获取数据
                new_df = self.fetch_data(limit=TRAINING_WINDOW+ATR_PERIOD)
                
                # ✅ 实时风险检查
                if not self.update_drawdown():
                    print("⛔ 策略暂停")
                    break
                
                self.check_risk_limits()
                
                
                if len(new_df) < TRAINING_WINDOW:
                    print("数据不足，等待更多数据...")
                    sleep(60)
                    continue
                
                # 计算ATR
                atr_series = self.calculate_atr(new_df, ATR_PERIOD).dropna()
                
                # 每4小时重新训练聚类模型
                if len(self.data) == 0 or len(new_df) > len(self.data):
                    self.train_clusters(atr_series[-TRAINING_WINDOW:])
                    centers = self.get_cluster_centers()
                    print(f"更新聚类中心：低波动={centers[0]:.2f}, 中波动={centers[1]:.2f}, 高波动={centers[2]:.2f}")
                
                # 动态选择ATR
                current_atr = atr_series.iloc[-1]
                cluster_idx = self.kmeans.predict([[current_atr]])[0]
                selected_atr = centers[cluster_idx]
                
                # 计算SuperTrend
                st_value, new_direction = self.supertrend(new_df.iloc[-ATR_PERIOD:], selected_atr, SUPERTREND_FACTOR)
                
                # 趋势变化检测
                if self.current_trend is None:
                    self.current_trend = new_direction
                else:
                    if new_direction != self.current_trend:
                        print(f"趋势变化：{'看涨' if new_direction ==1 else '看跌'} | 价格：{new_df['close'].iloc[-1]}")
                        
                        # 执行交易
                        if new_direction == 1:
                            self.buy()
                        else:
                            self.sell()
                            
                        self.current_trend = new_direction
                
                # 每小时保存日志
                if datetime.now().minute == 0:  
                    self.calculate_performance()
                
                sleep(3600)
                
            except Exception as e:
                print(f"策略异常: {str(e)}")
                sleep(60)

# ========== 运行策略 ==========
if __name__ == "__main__":
    strategy = EnhancedSuperTrend()
    try:
        strategy.execute_strategy()
    finally:
        # 结束时生成报告
        strategy.calculate_performance()