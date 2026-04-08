from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        parts: tuple[str, ...] = ("year", "month", "day", "dayofweek"),
        fill_value: int = -1,
    ) -> None:
        self.parts = parts
        self.fill_value = fill_value
        self.feature_names_in_: list[str] = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"datetime_{idx}" for idx in range(X.shape[1])]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        output = pd.DataFrame(index=X.index)

        for column in self.feature_names_in_:
            series = pd.to_datetime(X[column], errors="coerce")

            if "year" in self.parts:
                output[f"{column}__year"] = series.dt.year.fillna(self.fill_value).astype(int)
            if "month" in self.parts:
                output[f"{column}__month"] = series.dt.month.fillna(self.fill_value).astype(int)
            if "day" in self.parts:
                output[f"{column}__day"] = series.dt.day.fillna(self.fill_value).astype(int)
            if "dayofweek" in self.parts:
                output[f"{column}__dayofweek"] = series.dt.dayofweek.fillna(self.fill_value).astype(int)
            if "quarter" in self.parts:
                output[f"{column}__quarter"] = series.dt.quarter.fillna(self.fill_value).astype(int)

        return output

    def get_feature_names_out(self, input_features=None):
        input_features = list(input_features) if input_features is not None else self.feature_names_in_
        names: list[str] = []

        for column in input_features:
            if "year" in self.parts:
                names.append(f"{column}__year")
            if "month" in self.parts:
                names.append(f"{column}__month")
            if "day" in self.parts:
                names.append(f"{column}__day")
            if "dayofweek" in self.parts:
                names.append(f"{column}__dayofweek")
            if "quarter" in self.parts:
                names.append(f"{column}__quarter")

        return names


def build_datetime_pipeline(
    *,
    parts: tuple[str, ...] = ("year", "month", "day", "dayofweek"),
) -> DatetimeFeatureExtractor:
    return DatetimeFeatureExtractor(parts=parts)