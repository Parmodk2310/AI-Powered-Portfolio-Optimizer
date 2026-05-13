"""
risk.py
Portfolio risk metrics — volatility, correlation, Value at Risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def calculate_volatility(returns: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate annualized volatility for each stock.

    Args:
        returns: DataFrame of daily returns

    Returns:
        Dict mapping ticker to annualized volatility percentage
    """
    volatility = returns.std() * np.sqrt(252)
    return {ticker: round(vol * 100, 2) for ticker, vol in volatility.items()}


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix between all stocks.
    Values close to 1 = highly correlated (less diversification benefit).
    Values close to 0 = uncorrelated (good diversification).
    """
    return returns.corr().round(3)


def calculate_var(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    confidence: float = 0.95,
    investment: float = 100000
) -> Dict:
    """
    Calculate Value at Risk (VaR) — the maximum expected loss
    over one day at a given confidence level.

    Args:
        returns: DataFrame of daily returns
        weights: Portfolio weights dict
        confidence: Confidence level (0.95 = 95%)
        investment: Portfolio value in currency units

    Returns:
        Dict with VaR amount and percentage
    """
    weight_array = np.array([weights.get(t, 0) for t in returns.columns])
    portfolio_returns = returns.dot(weight_array)

    var_pct = np.percentile(portfolio_returns, (1 - confidence) * 100)
    var_amount = abs(var_pct * investment)

    return {
        "var_percentage": round(abs(var_pct) * 100, 3),
        "var_amount": round(var_amount, 2),
        "confidence_level": confidence,
        "interpretation": f"With {confidence*100:.0f}% confidence, maximum daily loss is {abs(var_pct)*100:.2f}% (₹{var_amount:,.0f} on ₹{investment:,.0f} portfolio)"
    }


def calculate_max_drawdown(returns: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """
    Calculate maximum drawdown — the largest peak-to-trough decline.

    Args:
        returns: DataFrame of daily returns
        weights: Portfolio weights dict

    Returns:
        Dict with max drawdown percentage
    """
    weight_array = np.array([weights.get(t, 0) for t in returns.columns])
    portfolio_returns = returns.dot(weight_array)

    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "max_drawdown_percentage": round(abs(max_drawdown) * 100, 2),
        "interpretation": f"Worst historical decline from peak: {abs(max_drawdown)*100:.2f}%"
    }


def full_risk_report(returns: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """
    Generate complete risk report for a portfolio.
    """
    return {
        "volatility_by_stock": calculate_volatility(returns),
        "value_at_risk": calculate_var(returns, weights),
        "max_drawdown": calculate_max_drawdown(returns, weights),
        "correlation_matrix": calculate_correlation_matrix(returns).to_dict()
    }


if __name__ == "__main__":
    from src.data.stock_fetcher import fetch_stock_data, calculate_returns
    from src.optimization.portfolio import optimize_portfolio

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    prices = fetch_stock_data(tickers, period="1y")
    returns = calculate_returns(prices)

    result = optimize_portfolio(returns)
    weights = result["weights"]

    report = full_risk_report(returns, weights)

    print("Volatility per stock:")
    for t, v in report["volatility_by_stock"].items():
        print(f"  {t}: {v}%")

    print(f"\nValue at Risk: {report['value_at_risk']['interpretation']}")
    print(f"Max Drawdown: {report['max_drawdown']['interpretation']}")