# -*- coding: utf-8 -*-
"""JoinQuant (Python3) ETF momentum rotation strategy.

在聚宽平台可直接运行的A股ETF动量轮动策略示例。
策略特点：
1. 以沪深市场流动性较好的ETF为标的，假设总资金5万元。
2. 使用多周期收益率（1/3/6个月）并以波动率进行归一化构建动量因子。
3. 每月调仓一次，选择得分最高的ETF持有，最多持有2只。
4. 每天检查单笔持仓的止盈止损，获利超过15%即落袋为安，亏损达到8%及时退出。
5. 加入简单的市场温度过滤：若沪深300指数过去63日收益为负，则空仓以规避系统性风险。

复制到聚宽策略编辑器后，可直接回测。
"""

from jqdata import *  # noqa: F401  聚宽平台会自动提供依赖

import numpy as np
import pandas as pd


def _get_latest_close(security):
    """使用 get_price 获取最新复权收盘价，兼容 Python3 版本。"""
    price = get_price(security, count=1, frequency='daily', fields=['close'], fq='pre')
    if price is None or price.empty:
        return None
    return float(price['close'][-1])


# 初始化函数，回测/实盘启动时运行一次
def initialize(context):
    # 设定基准与滑点等基础参数
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)  # 使用真实价格

    # 设定要轮动的ETF池，可按需增删
    g.etf_pool = [
        '510300.XSHG',  # 沪深300ETF
        '510500.XSHG',  # 中证500ETF
        '510900.XSHG',  # 300价值ETF
        '159915.XSHE',  # 创业板ETF
        '512100.XSHG',  # 中证1000ETF
        '512000.XSHG',  # 券商ETF
        '510880.XSHG',  # 红利ETF
    ]

    g.max_hold_num = 2             # 最大同时持有数量
    g.take_profit = 1.15           # 止盈线：收益≥15%
    g.stop_loss = 0.92             # 止损线：亏损≥8%
    g.market_filter_days = 63      # 市场温度窗口
    g.cash_to_use = 50000          # 策略可用资金（回测时在参数里设置一样的初始资金即可）

    # 调仓：每月第1个交易日开盘时执行
    run_monthly(rebalance, 1, 'open')
    # 风险控制：每日收盘前检查持仓
    run_daily(risk_control, time='14:50')


def rebalance(context):
    """月度调仓函数：根据动量得分决定持仓"""
    # 市场过滤，如果沪深300指数动量为负则空仓
    if not is_market_positive():
        for stock in list(context.portfolio.positions.keys()):
            order_target(stock, 0)
        log.info('市场动量疲弱，执行空仓防御。')
        return

    # 计算ETF池动量得分
    score_series = calc_momentum_score(g.etf_pool)
    if score_series.empty:
        log.info('无法计算动量得分，本次调仓跳过。')
        return

    # 仅保留动量为正的ETF，并按得分排序
    score_series = score_series[score_series > 0].sort_values(ascending=False)
    targets = list(score_series.index[:g.max_hold_num])
    log.info('本期入选ETF: {}'.format(targets))

    # 目标仓位 = 可用资金在目标ETF之间平均分配
    if targets:
        each_cash = g.cash_to_use / len(targets)
    else:
        each_cash = 0

    # 先卖出不在目标名单中的ETF
    for stock in list(context.portfolio.positions.keys()):
        if stock not in targets:
            order_target(stock, 0)

    # 再买入/调仓目标ETF
    for stock in targets:
        price = _get_latest_close(stock)
        if not price or price <= 0:
            continue
        target_amount = int(each_cash / price / 100) * 100  # ETF按100份整数买卖
        if target_amount > 0:
            order_target_value(stock, target_amount * price)


def calc_momentum_score(security_list):
    """计算ETF的动量得分：多周期收益率/波动率"""
    scores = {}
    for security in security_list:
        try:
            # 获取近160日收盘价
            hist = get_price(security, count=160, frequency='daily', fields=['close'], fq='pre')
            close = hist['close']
            if close.isnull().any() or len(close) < 126:
                continue

            # 计算1/3/6个月收益率（21、63、126个交易日）
            r1 = close.iloc[-1] / close.iloc[-21] - 1
            r3 = close.iloc[-1] / close.iloc[-63] - 1
            r6 = close.iloc[-1] / close.iloc[-126] - 1

            # 使用近3个月的收益率波动作为风险因子
            returns = close.pct_change().dropna()
            volatility = returns.iloc[-63:].std()
            if volatility == 0 or np.isnan(volatility):
                continue

            momentum = (0.3 * r1 + 0.5 * r3 + 0.2 * r6) / volatility
            scores[security] = momentum
        except Exception as e:
            log.warning('计算{}动量失败: {}'.format(security, e))
    if not scores:
        return pd.Series(dtype=float)
    return pd.Series(scores)


def is_market_positive():
    """判断沪深300指数过去3个月收益是否为正"""
    price = get_price('000300.XSHG', count=g.market_filter_days + 1, frequency='daily', fields=['close'], fq='pre')
    close = price['close']
    if len(close) < g.market_filter_days + 1:
        return True  # 数据不足时默认不空仓
    market_return = close.iloc[-1] / close.iloc[0] - 1
    log.info('沪深300 {}日收益率：{:.2%}'.format(g.market_filter_days, market_return))
    return market_return > 0


def risk_control(context):
    """每日风控：执行止盈止损"""
    for stock, position in list(context.portfolio.positions.items()):
        if position.closeable_amount <= 0:
            continue
        current_price = _get_latest_close(stock)
        cost = position.avg_cost
        if not current_price or cost <= 0 or current_price <= 0:
            continue

        # 止盈
        if current_price >= cost * g.take_profit:
            log.info('{} 触发止盈，价格 {:.2f} >= 成本 {:.2f}'.format(stock, current_price, cost))
            order_target(stock, 0)
            continue

        # 止损
        if current_price <= cost * g.stop_loss:
            log.info('{} 触发止损，价格 {:.2f} <= 成本 {:.2f}'.format(stock, current_price, cost))
            order_target(stock, 0)


def after_trading_end(context):
    """每日收盘后打印持仓信息，便于回顾。"""
    log.info('当前总资产：{:.2f} 元，持仓：{}'.format(context.portfolio.total_value, context.portfolio.positions))
