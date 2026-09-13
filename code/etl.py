"""Data ingestion and sanitization pipeline."""

import os
import logging
import pandas as pd
from typing import Optional

from config import DATA_DIR


class ETL:
    """Extract, Transform, and Load module for standardizing hackathon datasets."""

    def __init__(self, data_dir: str = DATA_DIR):
        """Initializes the ETL pipeline.

        Args:
            data_dir (str): The directory containing the dataset CSVs.
        """
        self.data_dir = data_dir
        self.profiles_df: Optional[pd.DataFrame] = None
        self.events_df: Optional[pd.DataFrame] = None
        self.rates_df: Optional[pd.DataFrame] = None
        self.requests_df: Optional[pd.DataFrame] = None
        self.options_df: Optional[pd.DataFrame] = None
        self.messages_df: Optional[pd.DataFrame] = None
        self.images_df: Optional[pd.DataFrame] = None

    def run(self) -> None:
        """Executes the ingestion and cleansing process."""
        logging.info("Starting ETL pipeline...")
        try:
            self.profiles_df = pd.read_csv(
                os.path.join(self.data_dir, "financial_profiles.csv")
            )
            self.events_df = pd.read_csv(
                os.path.join(self.data_dir, "financial_events.csv")
            )
            self.rates_df = pd.read_csv(
                os.path.join(self.data_dir, "exchange_rates.csv")
            )
            self.requests_df = pd.read_csv(os.path.join(self.data_dir, "requests.csv"))

            # Additional datasets
            options_path = os.path.join(self.data_dir, "request_payment_options.csv")
            self.options_df = (
                pd.read_csv(options_path)
                if os.path.exists(options_path)
                else pd.DataFrame()
            )

            messages_path = os.path.join(self.data_dir, "messages.csv")
            self.messages_df = (
                pd.read_csv(messages_path)
                if os.path.exists(messages_path)
                else pd.DataFrame()
            )

            images_path = os.path.join(self.data_dir, "images.csv")
            self.images_df = (
                pd.read_csv(images_path)
                if os.path.exists(images_path)
                else pd.DataFrame()
            )

            # Merge home currency onto events
            self.events_df = self.events_df.merge(
                self.profiles_df[["user_id", "home_currency"]], on="user_id", how="left"
            )

            # Normalize dates
            self.events_df["settlement_date"] = pd.to_datetime(
                self.events_df["settlement_date"]
            )
            self.rates_df["rate_date"] = pd.to_datetime(self.rates_df["rate_date"])

            # Convert currency
            merged = self.events_df.merge(
                self.rates_df,
                left_on=["currency", "home_currency", "settlement_date"],
                right_on=["from_currency", "to_currency", "rate_date"],
                how="left",
            )

            # Apply rate to amount
            mask = merged["rate"].notna() & (
                merged["currency"] != merged["home_currency"]
            )
            merged.loc[mask, "amount"] = (
                merged.loc[mask, "amount"] * merged.loc[mask, "rate"]
            )
            merged.loc[mask, "currency"] = merged.loc[mask, "home_currency"]

            self.events_df["amount"] = merged["amount"].astype("float64")
            self.events_df["currency"] = merged["currency"]

            # Clean ledger
            drop_mask = (
                self.events_df["status"]
                .str.lower()
                .isin(["failed", "cancelled", "unrealized"])
            )
            self.events_df = self.events_df[~drop_mask].copy()
            logging.info("ETL pipeline completed successfully.")
        except Exception as e:
            logging.error(f"ETL Error during pandas processing: {e}")
            raise e
