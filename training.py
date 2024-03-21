from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import fire
import pandas as pd
from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator
from model.model import MultiTargetModel


@PipelineDecorator.component(
    return_values=["orders"],
    task_type=TaskTypes.data_processing,
)
def fetch_orders(orders_url: str) -> pd.DataFrame:
    import requests
    from urllib.parse import urlencode
    import pandas as pd
    from clearml import StorageManager

    print(f"Downloading orders data from {orders_url}...")

    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    full_url = base_url + urlencode(dict(public_key=orders_url))
    response = requests.get(full_url)
    download_url = response.json()["href"]

    local_path = StorageManager.get_local_copy(remote_url=download_url)
    df_orders = pd.read_csv(
        local_path,
        parse_dates=["timestamp"],
        dayfirst=True,
    )

    print(f"Orders data downloaded. orders.csv shape: {df_orders.shape}")

    return df_orders


@PipelineDecorator.component(
    return_values=["sales"],
    task_type=TaskTypes.data_processing,
)
def extract_sales(df_orders: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    print("Extracting sales data...")

    df_orders["timestamp"] = pd.to_datetime(df_orders["timestamp"], dayfirst=True)
    df_sales = df_orders.copy()

    df_sales["day"] = df_sales["timestamp"].dt.floor("D")

    df_sales = (
        df_sales.groupby(["day", "sku_id", "sku", "price"])["qty"].sum().reset_index()
    )

    all_sku_ids = df_sales["sku_id"].unique()
    all_dates = pd.date_range(
        df_sales["day"].min(),
        df_sales["day"].max(),
        freq="D",
    )

    all_dates_sku_df = pd.DataFrame(
        {
            "day": np.repeat(all_dates, len(all_sku_ids)),
            "sku_id": np.tile(all_sku_ids, len(all_dates)),
        }
    )

    df_sales = pd.merge(all_dates_sku_df, df_sales, how="left", on=["day", "sku_id"])
    df_sales["qty"] = df_sales["qty"].fillna(0).astype(int)

    # fill missing sku and price from df
    df = df_orders[["sku_id", "sku", "price"]].drop_duplicates().reset_index(drop=True)
    df_sales = pd.merge(
        df_sales, df[["sku_id", "sku", "price"]], how="left", on="sku_id"
    )
    df_sales["sku"] = df_sales["sku_x"].fillna(df_sales["sku_y"])
    df_sales["price"] = df_sales["price_x"].fillna(df_sales["price_y"])
    df_sales.drop(columns=["sku_x", "sku_y", "price_x", "price_y"], inplace=True)

    df_sales = df_sales[["day", "sku_id", "sku", "price", "qty"]]

    df_sales.sort_values(by=["sku_id", "day"], inplace=True)
    df_sales.reset_index(drop=True, inplace=True)

    print(f"Sales data extracted. sales.csv shape: {df_sales.shape}")

    return df_sales


@PipelineDecorator.component(
    return_values=["features"],
    task_type=TaskTypes.data_processing,
)
def extract_features(
    df_sales: pd.DataFrame,
    features: Dict[str, Tuple[str, int, str, Optional[int]]],
    targets: Dict[str, Tuple[str, int]],
) -> pd.DataFrame:
    from features import add_features, add_targets

    print("Extracting features...")

    df_features = df_sales.copy()

    add_features(df_features, features)
    add_targets(df_features, targets)

    df_features.sort_values(["sku_id", "day"], inplace=True)

    print(f"Features extracted. features.csv shape: {df_features.shape}")

    return df_features


@PipelineDecorator.component(
    return_values=["df_train, df_test"],
    cache=True,
    task_type=TaskTypes.data_processing,
)
def split_train_test(
    df_features: pd.DataFrame,
    test_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import pandas as pd  # noqa
    from model.model import split_train_test

    print("Splitting train and test data...")

    df_features['day'] = pd.to_datetime(df_features['day'])
    date_for_split = df_features['day'].max() - pd.Timedelta(test_days, 'd')
    df_train = df_features[df_features['day'] < date_for_split]
    df_test = df_features[df_features['day'] >= date_for_split]

    print("Train and test data splitted.")

    return df_train, df_test


@PipelineDecorator.component(
    return_values=["model"],
    cache=True,
    task_type=TaskTypes.training,
)
def fit_model(
    df_features: pd.DataFrame,
    features: List[str],
    quantiles: List[float],
    horizons: List[int],
) -> MultiTargetModel:
    from model.model import MultiTargetModel

    print("Training production model...")
    model = MultiTargetModel(features=features, quantiles=quantiles, horizons=horizons)
    model.fit(df_features)

    print("Production model trained.")

    return model


@PipelineDecorator.component(
    return_values=["eval_model"],
    cache=True,
    task_type=TaskTypes.training,
)
def fit_eval_model(
    df_train: pd.DataFrame,
    features: List[str],
    quantiles: List[float],
    horizons: List[int],
) -> MultiTargetModel:
    from model.model import MultiTargetModel

    print("Training evaluation model...")

    from model.model import MultiTargetModel

    print("Training production model...")
    model = MultiTargetModel(features=features, quantiles=quantiles, horizons=horizons)
    model.fit(df_train)

    print("Evaluation model trained.")

    return model


@PipelineDecorator.component(
    return_values=["losses, df_pred"],
    task_type=TaskTypes.qc,
)
def evaluate(
    eval_model: MultiTargetModel,
    df_test: pd.DataFrame,
    quantiles: List[float],
    horizons: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from model.evaluate import evaluate_model

    print("Evaluating model...")
    df_pred = eval_model.predict(data=df_test)
    losses = evaluate_model(df_pred=df_pred, df_true=df_test)

    print("Model evaluated.")

    return losses, df_pred


@PipelineDecorator.component(
    task_type=TaskTypes.custom,
)
def deploy_model(
    model: MultiTargetModel,
    model_path: str,
    losses: pd.DataFrame,
    df_pred: pd.DataFrame,
) -> None:
    import pickle
    from model.evaluate import test_losses

    print("Check model quality...")

    print(f"Losses: {losses}")


    print("Quality checked. Saving production model...")

    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    print("Production model saved!")


@PipelineDecorator.pipeline(
    name="Training Pipeline",
    project="Stock Management System Task",
    version="1.0.0",
)
def run_pipeline(
    orders_url: str,
    test_days: int,
    model_path: str,
    features: Dict[str, Tuple[str, int, str, Optional[int]]],
    targets: Dict[str, Tuple[str, int]],
    quantiles: List[float],
    horizons: List[int],
) -> None:
    orders_df = fetch_orders(orders_url)

    df_sales = extract_sales(orders_df)

    df_features = extract_features(df_sales, features, targets)
    df_features = df_features.dropna()
    df_train, df_test = split_train_test(df_features, test_days)

    model_features = ["price", "qty"] + list(features.keys())
    
    prod_model = fit_model(df_features, model_features, quantiles, horizons)
    eval_model = fit_eval_model(df_train, model_features, quantiles, horizons)
    eval_losses, df_pred = evaluate(eval_model, df_test, quantiles, horizons)
    deploy_model(prod_model, model_path, eval_losses, df_pred)


def main(
    orders_url: str = "https://disk.yandex.ru/d/NUDMAdBMe9sbLw",
    model_path: str = "model.pkl",
    debug: bool = False,
) -> None:
    """Main function

    Args:
        orders_url (str): URL to the orders data on Yandex Disk
        model_path (str): Local path of production model
        debug (bool, optional): Run the pipeline in debug mode.
            In debug mode no Taska are created, so it is running faster.
            Defaults to False.
    """

    if debug:
        PipelineDecorator.debug_pipeline()
    else:
        PipelineDecorator.run_locally()

    test_days = 30

    features = {
        "qty_7d_avg": ("qty", 7, "avg", None),
        "qty_7d_q10": ("qty", 7, "quantile", 10),
        "qty_7d_q50": ("qty", 7, "quantile", 50),
        "qty_7d_q90": ("qty", 7, "quantile", 90),
        "qty_14d_avg": ("qty", 14, "avg", None),
        "qty_14d_q10": ("qty", 14, "quantile", 10),
        "qty_14d_q50": ("qty", 14, "quantile", 50),
        "qty_14d_q90": ("qty", 14, "quantile", 90),
        "qty_21d_avg": ("qty", 21, "avg", None),
        "qty_21d_q10": ("qty", 21, "quantile", 10),
        "qty_21d_q50": ("qty", 21, "quantile", 50),
        "qty_21d_q90": ("qty", 21, "quantile", 90),
    }

    targets = {
        "next_7d": ("qty", 7),
        "next_14d": ("qty", 14),
        "next_21d": ("qty", 21),
    }

    quantiles = [0.1, 0.5, 0.9]

    horizons = [7, 14, 21]

    run_pipeline(
        orders_url=orders_url,
        test_days=test_days,
        model_path=model_path,
        features=features,
        targets=targets,
        quantiles=quantiles,
        horizons=horizons,
    )


if __name__ == "__main__":
    fire.Fire(main)
