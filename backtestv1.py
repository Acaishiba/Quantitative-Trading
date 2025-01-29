# backtest.py 文件开头
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta, timezone
import logging

class EnhancedBacktester:
    def __init__(self, symbol, timeframe, initial_capital=1000, 
                 position_size=0.1, take_profit=0.15, stop_loss=0.05, 
                 fee_rate=0.0004, since_days=90):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.fee_rate = fee_rate
        self.since_days = since_days
        
        # 初始化交易所
        # 在初始化时添加数据延迟补偿
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'fetchMissingOHLCV': True,  # 自动补偿缺失数据
                'maxRetries': 5             # 增加重试次数
            }
        })
        
        # 状态变量
        self.data = None
        self.trades = []
        self.equity_curve = []
        self._prepare_logger()

    def _prepare_logger(self):
        """配置日志记录"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('backtest.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def fetch_historical_data(self):
        """分页获取完整历史数据"""
        all_ohlcv = []
        since = self.exchange.parse8601(
            (datetime.now(timezone.utc) - timedelta(days=self.since_days + 3)).isoformat()
        )
        
        self.logger.info(f"开始获取 {self.symbol} {self.timeframe} 历史数据...")
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol, 
                    self.timeframe, 
                    since=since,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv += ohlcv
                since = ohlcv[-1][0] + 1  # 下一批数据的起始时间戳
                
                # Binance API限制处理
                time.sleep(self.exchange.rateLimit / 1000)
                
                if len(ohlcv) < 1000:
                    break
                    
            except Exception as e:
                self.logger.error(f"获取数据失败: {e}")
                break
        
        if all_ohlcv:
            self.data = pd.DataFrame(
                all_ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms')
            self.data.set_index('timestamp', inplace=True)
            
            # 计算技术指标
            self._calculate_indicators()
            
            self.logger.info(f"成功获取 {len(self.data)} 根K线数据")
            return True
        return False

    def _calculate_indicators(self):
        """计算技术指标"""
        # 布林带
        window = 20
        self.data['ma'] = self.data['close'].rolling(window=window).mean()
        self.data['std'] = self.data['close'].rolling(window=window).std()
        self.data['upper_band'] = self.data['ma'] + 2 * self.data['std']
        self.data['lower_band'] = self.data['ma'] - 2 * self.data['std']
        
        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # 清理临时列
        self.data.drop(['ma', 'std'], axis=1, inplace=True)

    def run_backtest(self):
        """执行增强版回测"""
        if self.data is None or len(self.data) < 50:
            self.logger.error("数据不足或未加载")
            return
        
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        in_position = False
        self.equity_curve = [capital]
        self.trades = []
        
        self.logger.info("开始回测...")
        
        for i in range(20, len(self.data)):  # 跳过初始计算期
            current = self.data.iloc[i]
            
            # 持仓监控
            if in_position:
                current_value = position * current['close']
                unrealized_pnl = current_value - (position * entry_price)
                print(f"未实现盈亏: {unrealized_pnl:.4f}")
                
                # 止盈止损检查
                take_profit_price = entry_price * (1 + self.take_profit)
                stop_loss_price = entry_price * (1 - self.stop_loss)
                
                close_reason = None
                if current['close'] >= take_profit_price:
                    close_reason = 'Take Profit'
                elif current['close'] <= stop_loss_price:
                    close_reason = 'Stop Loss'
                    
                if close_reason:
                    # 平仓
                    close_value = position * current['close']
                    capital += close_value * (1 - self.fee_rate)
                    pct_change = (current['close'] - entry_price) / entry_price
                    
                    self.trades.append({
                        'type': 'sell',
                        'timestamp': current.name,
                        'price': current['close'],
                        'size': position,
                        'reason': close_reason,
                        'profit_pct': pct_change,
                        'equity': capital
                    })
                    
                    position = 0.0
                    in_position = False
                    self.logger.debug(f"平仓: {close_reason} @ {current['close']:.4f}")
            
            # 信号生成
            signal = self._generate_signal(current)
            
            # 开仓逻辑
            if signal == 'buy' and not in_position and capital > 10:
                try:
                    position_value = capital * self.position_size
                    fee = position_value * self.fee_rate
                    capital -= (position_value + fee)
                    position = position_value / current['close']
                    entry_price = current['close']
                    in_position = True
                    
                    self.trades.append({
                        'type': 'buy',
                        'timestamp': current.name,
                        'price': entry_price,
                        'size': position,
                        'equity': capital + position_value
                    })
                    self.logger.debug(f"开仓买入 @ {entry_price:.4f}")
                except Exception as e:
                    self.logger.error(f"开仓失败: {e}")
            
            # 更新净值
            current_equity = capital + (position * current['close'] if in_position else 0)
            self.equity_curve.append(current_equity)
        
        # 强制平仓
        if in_position:
            close_price = self.data.iloc[-1]['close']
            close_value = position * close_price
            capital += close_value * (1 - self.fee_rate)
            pct_change = (close_price - entry_price) / entry_price
            
            self.trades.append({
                'type': 'sell',
                'timestamp': self.data.index[-1],
                'price': close_price,
                'size': position,
                'reason': 'Force Close at End',
                'profit_pct': pct_change,
                'equity': capital
            })
            self.logger.info(f"强制平仓 @ {close_price:.4f}")
        
        return self.trades

    def _generate_signal(self, data_point):
        """增强信号生成逻辑"""
        try:
            if data_point['close'] < data_point['lower_band'] and data_point['rsi'] < 30:
                return 'buy'
            elif data_point['close'] > data_point['upper_band'] and data_point['rsi'] > 70:
                return 'sell'
            return None
        except KeyError as e:
            self.logger.error(f"信号生成错误: {e}")
            return None

    def analyze_performance(self):
        """增强绩效分析"""
        if not self.trades:
            print("没有交易记录")
            return
        
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        total_return = (self.equity_curve[-1] / self.initial_capital - 1) * 100
        
        # 计算年化收益率
        days = (self.data.index[-1] - self.data.index[0]).days
        annualized_return = ((self.equity_curve[-1] / self.initial_capital) ** (365/days) - 1) * 100
        
        # 最大回撤
        max_drawdown = (pd.Series(self.equity_curve).cummax() - pd.Series(self.equity_curve)).max()
        
        # 夏普比率
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365*24)  # 小时数据
        
        # 盈亏比
        profit_factor = self._calculate_profit_factor()
        
        # 胜率
        winning_trades = [t for t in self.trades if t.get('profit_pct', 0) > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        print(f"""
        ========== 增强回测报告 ==========
        交易对\t\t{self.symbol}
        时间周期\t\t{self.timeframe}
        测试期间\t\t{self.data.index[0]} - {self.data.index[-1]}
        初始资金\t\t${self.initial_capital:,.2f}
        最终资金\t\t${self.equity_curve[-1]:,.2f}
        总收益率\t\t{total_return:.2f}%
        年化收益率\t{annualized_return:.2f}%
        最大回撤\t\t{max_drawdown/self.initial_capital*100:.2f}%
        夏普比率\t\t{sharpe_ratio:.2f}
        盈亏比\t\t{profit_factor:.2f}
        交易次数\t\t{len(self.trades)}
        胜率\t\t{win_rate*100:.2f}%
        ================================
        """)
        
        self._plot_analysis()
        self._save_trade_logs()

    def _calculate_profit_factor(self):
        """计算增强版盈亏比"""
        gross_profit = sum(t['profit_pct'] for t in self.trades if t.get('profit_pct', 0) > 0)
        gross_loss = abs(sum(t['profit_pct'] for t in self.trades if t.get('profit_pct', 0) < 0))
        return gross_profit / gross_loss if gross_loss != 0 else np.inf

    def _plot_analysis(self):
        """增强可视化分析"""
        plt.figure(figsize=(18, 12))
        
        # 净值曲线
        plt.subplot(3, 2, 1)
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Capital')
        
        # 价格与交易信号
        plt.subplot(3, 2, 2)
        plt.plot(self.data['close'], label='Price', alpha=0.5)
        buy_dates = [t['timestamp'] for t in self.trades if t['type'] == 'buy']
        sell_dates = [t['timestamp'] for t in self.trades if t['type'] == 'sell']
        plt.scatter(buy_dates, self.data.loc[buy_dates, 'close'], 
                   marker='^', color='g', label='Buy')
        plt.scatter(sell_dates, self.data.loc[sell_dates, 'close'],
                   marker='v', color='r', label='Sell')
        plt.title('Price with Trading Signals')
        plt.legend()
        
        # 每日收益率分布
        plt.subplot(3, 2, 3)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        plt.hist(returns, bins=50, alpha=0.7)
        plt.title('Daily Returns Distribution')
        
        # 回撤曲线
        plt.subplot(3, 2, 4)
        cumulative = pd.Series(self.equity_curve)
        max_drawdown = cumulative.cummax() - cumulative
        plt.fill_between(max_drawdown.index, max_drawdown, color='red', alpha=0.3)
        plt.title('Drawdown Curve')
        
        # RSI指标
        plt.subplot(3, 2, 5)
        plt.plot(self.data['rsi'], label='RSI')
        plt.axhline(30, color='gray', linestyle='--')
        plt.axhline(70, color='gray', linestyle='--')
        plt.title('RSI Indicator')
        
        # 持仓标记
        plt.subplot(3, 2, 6)
        in_position = [any(t['timestamp'] <= idx for t in self.trades if t['type'] == 'buy') 
                      and not any(t['timestamp'] <= idx for t in self.trades if t['type'] == 'sell')
                      for idx in self.data.index]
        plt.fill_between(self.data.index, 0, 1, where=in_position, 
                        color='green', alpha=0.3, transform=plt.gca().get_xaxis_transform())
        plt.title('Position Holding Periods')
        
        plt.tight_layout()
        plt.show()

    def _save_trade_logs(self):
        """保存详细交易日志"""
        log_df = pd.DataFrame(self.trades)
        if not log_df.empty:
            log_df.to_csv('trade_details.csv', index=False)
            print("交易详情已保存到 trade_details.csv")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 配置参数
    config = {
        'symbol': 'SUI/USDT',
        'timeframe': '1h',
        'initial_capital': 1000,
        'position_size': 0.1,
        'take_profit': 0.15,
        'stop_loss': 0.05,
        'fee_rate': 0.0004,
        'since_days': 30
    }
    
    # 初始化回测引擎
    backtester = EnhancedBacktester(**config)
    
    if backtester.fetch_historical_data():
        trades = backtester.run_backtest()
        backtester.analyze_performance()
    else:
        print("数据获取失败，请检查网络或参数设置")