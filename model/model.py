from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from tqdm import tqdm


def split_train_test(
    df: pd.DataFrame,
    test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets.

    The last `test_days` days are held out for testing.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        test_days (int): The number of days to include in the test set (default: 30).
            use ">=" sign for df_test

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
        A tuple containing the train and test DataFrames.
    """
    df['day'] = pd.to_datetime(df['day'])
    date_for_split = df['day'].max() - pd.Timedelta(test_days, 'd')
    df_train = df[df['day'] < date_for_split]
    df_test = df[df['day'] >= date_for_split]

    return df_train, df_test


class MultiTargetModel:
    def __init__(
        self,
        features: List[str],
        horizons: List[int] = [7, 14, 21],
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """
        Parameters
        ----------
        features : List[str]
            List of features columns.
        horizons : List[int]
            List of horizons.
        quantiles : List[float]
            List of quantiles.

        Attributes
        ----------
        fitted_models_ : dict
            Dictionary with fitted models for each sku_id.
            Example:
            {
                sku_id_1: {
                    (quantile_1, horizon_1): model_1,
                    (quantile_1, horizon_2): model_2,
                    ...
                },
                sku_id_2: {
                    (quantile_1, horizon_1): model_3,
                    (quantile_1, horizon_2): model_4,
                    ...
                },
                ...
            }

        """
        self.quantiles = quantiles
        self.horizons = horizons
        self.sku_col = "sku_id"
        self.date_col = "day"
        self.features = features
        self.targets = [f"next_{horizon}d" for horizon in self.horizons]

        self.fitted_models_ = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False) -> None:
        """Fit model on data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit on.
        verbose : bool, optional
            Whether to show progress bar, by default False
            Optional to implement, not used in grading.
        """
        data = data.dropna()

        for grouped_data in data.groupby(self.sku_col):
            sku_data = grouped_data[1]
            sku = sku_data[self.sku_col].iloc[0]
            self.fitted_models_[sku] = {}
            
            for horizon in self.horizons:
                for q in self.quantiles:
                    model = QuantileRegressor(quantile=q, alpha=0, solver='highs')
                    model.fit(sku_data[self.features], sku_data[f"next_{horizon}d"])
                    self.fitted_models_[sku][(q, horizon)] = model 


    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on data.

        Predict 0 values for a new sku_id.

        Parameters
        ----------
        data : pd.DataFrame
            Data to predict on.

        Returns
        -------
        pd.DataFrame
            Predictions.
        """
        predictions = pd.DataFrame(data=dict(zip(
            [self.sku_col, self.date_col] + [f'pred_{h}d_q{int(q*100)}' for h in self.horizons for q in self.quantiles],
            [[] for _ in range(len(self.horizons)*len(self.quantiles)+2)]
        )))
        for grouped_data in data.groupby(self.sku_col):
            sku_data = grouped_data[1]
            sku = sku_data[self.sku_col].iloc[0]
            tmp_predictions = {}
            tmp_predictions[sku] =  {}
            tmp_predictions[sku]['day'] = sku_data['day'].values
            for horizon in self.horizons:
                for q in self.quantiles:
                    try:
                        model = self.fitted_models_.get(sku)[(q, horizon)]        
                        sku_predictions = model.predict(sku_data[self.features])
                        tmp_predictions[sku][f'pred_{horizon}d_q{int(q*100)}'] = sku_predictions

                    except TypeError:
                        sku_predictions = np.zeros(shape=(sku_data.shape[0], 1)) ## zeros if sku not presented in train data
                        tmp_predictions[sku][f'pred_{horizon}d_q{int(q*100)}'] = sku_predictions
            
            tmp_predictions = pd.DataFrame.from_dict(tmp_predictions, orient='index')\
            .explode(list(tmp_predictions[sku].keys())).reset_index().rename(columns={'index':'sku_id'})
            predictions = pd.concat([predictions, tmp_predictions], axis=0)
        
        predictions[self.sku_col] = predictions[self.sku_col].astype('int')

        return predictions



def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the quantile loss between the true and predicted values.

    The quantile loss measures the deviation between the true
        and predicted values at a specific quantile.

    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    quantile : float
        The quantile to calculate the loss for.

    Returns
    -------
    float
        The quantile loss.
    """
    loss = (quantile * np.clip(y_true - y_pred, a_min=0, a_max=None) \
            + (1-quantile)*np.clip(y_pred - y_true, a_min=0, a_max=None)).mean()
    
    return loss
