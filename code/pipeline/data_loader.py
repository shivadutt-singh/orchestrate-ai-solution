import os
import pandas as pd
from typing import Dict, Any


class DataLoader:
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        self.profiles_df = pd.DataFrame()
        self.events_df = pd.DataFrame()
        self.rates_df = pd.DataFrame()
        self.requests_df = pd.DataFrame()
        self.options_df = pd.DataFrame()
        self.messages_df = pd.DataFrame()
        self.images_df = pd.DataFrame()

    def load_all(self) -> None:
        self.profiles_df = pd.read_csv(
            os.path.join(self.data_dir, "financial_profiles.csv")
        )
        self.events_df = pd.read_csv(
            os.path.join(self.data_dir, "financial_events.csv")
        )
        self.rates_df = pd.read_csv(os.path.join(self.data_dir, "exchange_rates.csv"))
        self.requests_df = pd.read_csv(os.path.join(self.data_dir, "requests.csv"))
        self.options_df = pd.read_csv(
            os.path.join(self.data_dir, "request_payment_options.csv")
        )
        self.messages_df = pd.read_csv(os.path.join(self.data_dir, "messages.csv"))
        self.images_df = pd.read_csv(os.path.join(self.data_dir, "images.csv"))

        self._standardize_dates()
        self._join_profiles()
        self._apply_currency_conversion()

    def _standardize_dates(self) -> None:
        # Standardize all dates to YYYY-MM-DD
        if "settlement_date" in self.events_df.columns:
            self.events_df["settlement_date"] = pd.to_datetime(
                self.events_df["settlement_date"]
            ).dt.strftime("%Y-%m-%d")
        if "request_date" in self.requests_df.columns:
            self.requests_df["request_date"] = pd.to_datetime(
                self.requests_df["request_date"]
            ).dt.strftime("%Y-%m-%d")
        if "desired_completion_date" in self.requests_df.columns:
            self.requests_df["desired_completion_date"] = pd.to_datetime(
                self.requests_df["desired_completion_date"]
            ).dt.strftime("%Y-%m-%d")
        if "rate_date" in self.rates_df.columns:
            self.rates_df["rate_date"] = pd.to_datetime(
                self.rates_df["rate_date"]
            ).dt.strftime("%Y-%m-%d")

    def _join_profiles(self) -> None:
        self.events_df = self.events_df.merge(
            self.profiles_df[["user_id", "home_currency", "minimum_balance_to_keep"]],
            on="user_id",
            how="left",
        )

    def _apply_currency_conversion(self) -> None:
        def convert_row(row: pd.Series) -> float:
            amt = row.get("amount")
            if pd.isna(amt) or str(amt).strip() == "":
                return amt
            amt = float(amt)
            currency = row.get("currency", row["home_currency"])
            home = row["home_currency"]

            if currency == home:
                return amt

            date = row["settlement_date"]
            rate_row = self.rates_df[
                (self.rates_df["rate_date"] == date)
                & (self.rates_df["from_currency"] == currency)
                & (self.rates_df["to_currency"] == home)
            ]
            if not rate_row.empty:
                return amt * float(rate_row.iloc[0]["rate"])

            return amt

        self.events_df["amount"] = self.events_df.apply(convert_row, axis=1)
        self.events_df["currency"] = self.events_df["home_currency"]
