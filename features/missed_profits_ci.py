from typing import Tuple
import pandas as pd
import numpy as np

def week_missed_profits(
    df: pd.DataFrame,
    sales_col: str,
    forecast_col: str,
    date_col: str = "day",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Calculates the missed profits every week for the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed profits for.
        (Must contain columns "sku_id", "date", "price", "sales" and "forecast")
    sales_col : str
        The column with the actual sales.
    forecast_col : str
        The column with the forecasted sales.
    price_col : str, optional
        The column with the price, by default "price".

    Returns
    -------
    pd.DataFrame
        The DataFrame with the missed profits.
        (Contains columns "day", "revenue", "missed_profits")
    """
    
    df['revenue'] = df[sales_col] * df[price_col]
    df['missed_profits'] = (df[forecast_col]*df[price_col] - df['revenue']).\
        apply(lambda x: x if x>0 else 0)
    df = df.groupby([pd.Grouper(key=date_col, freq='W')])\
        .agg({'revenue':'sum', 'missed_profits':'sum'}).reset_index()
    return df

def missed_profits_ci(
    df: pd.DataFrame,
    missed_profits_col: str, 
    confidence_level: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]:
    """
    Estimates the missed profits for the given DataFrame.
    Calculates average missed_profits per week and estimates the 95% confidence interval.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to calculate the missed_profits for.
 
    missed_profits_col : str
        The column with the missed_profits.

    confidence_level : float, optional
        The confidence level for the confidence interval, by default 0.95.

    n_bootstraps : int, optional
        The number of bootstrap samples to use for the confidence interval, by default 1000.

    Returns
    -------
    Tuple[Tuple[float, Tuple[float, float]], Tuple[float, Tuple[float, float]]]
        Returns a tuple of tuples, where the first tuple is the absolute average 
        missed profits with its CI, and the second is the relative average missed 
        profits with its CI.

    Example:
    -------
    ((1200000, (1100000, 1300000)), (0.5, (0.4, 0.6)))
    """

    abs_values = np.random.choice(a=df[missed_profits_col].values,
                                  size=(df.shape[0], n_bootstraps)).mean(axis=0)
    rel_values = abs_values/df['revenue'].mean()
    q1, q2 = (1 - confidence_level)/2, 0.5 + confidence_level/2 
    return (abs_values.mean(), tuple(np.quantile(abs_values,q=[q1, q2]).tolist())),\
            (rel_values.mean(), tuple(np.quantile(rel_values,q=[q1, q2]).tolist())),\

