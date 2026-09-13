"""Vectorized Engine for 90-day Ledger Simulations."""

import pandas as pd
import numpy as np
from typing import List, Tuple


class Simulator:
    """Simulates a 90-day cash flow ledger using strict vectorization."""

    def __init__(self, profile: pd.Series, events: pd.DataFrame):
        """Initializes the Simulator.

        Args:
            profile (pd.Series): The user's financial profile.
            events (pd.DataFrame): The sanitized, converted events dataframe.
        """
        self.profile = profile
        self.events = events.copy()
        self.min_balance = float(self.profile.get("minimum_balance_to_keep", 0.0))
        self.ledger: pd.DataFrame = pd.DataFrame()

    def generate_ledger(self, request_date: pd.Timestamp) -> pd.DataFrame:
        """Generates the 90-day forward-looking ledger delta map.

        Args:
            request_date (pd.Timestamp): The origin date for the 90-day projection.

        Returns:
            pd.DataFrame: The generated daily balance ledger.
        """
        dates = pd.date_range(start=request_date, periods=90, freq="D")
        ledger = pd.DataFrame({"date": dates})
        ledger["delta"] = 0.0

        # Add initial balance to day 0
        ledger.loc[0, "delta"] = float(
            self.profile.get("current_available_balance", 0.0)
        )

        # Map event impacts
        for _, row in self.events.iterrows():
            evt_date = pd.to_datetime(row["settlement_date"])
            if request_date <= evt_date <= dates[-1]:
                amt = float(row["amount"])
                if str(row["direction"]).lower() == "debit":
                    amt = -amt

                # Exclude pending credits
                if (
                    str(row["direction"]).lower() == "credit"
                    and str(row["status"]).lower() == "pending"
                ):
                    continue

                # Vectorized assignment
                ledger.loc[ledger["date"] == evt_date, "delta"] += amt

        # Vectorized cumsum
        ledger["daily_balance"] = ledger["delta"].cumsum()
        self.ledger = ledger
        return ledger

    def is_safe(self) -> bool:
        """Validates if the current ledger safely maintains the minimum balance.

        Returns:
            bool: True if safe, False otherwise.
        """
        return (self.ledger["daily_balance"] >= self.min_balance).all()

    def get_max_safe_deduction(self) -> float:
        """Calculates the absolute maximum safe deduction on Day 0.

        Returns:
            float: Maximum safe immediate deduction.
        """
        min_headroom = (self.ledger["daily_balance"] - self.min_balance).min()
        return max(0.0, float(min_headroom))

    def check_schedule_safe(self, schedule: List[Tuple[pd.Timestamp, float]]) -> bool:
        """Applies a virtual payment schedule and checks safety boundaries.

        Args:
            schedule (List[Tuple[pd.Timestamp, float]]): The payment schedule to test.

        Returns:
            bool: True if the resulting ledger is safe, False otherwise.
        """
        temp_ledger = self.ledger.copy()
        for p_date, p_amt in schedule:
            mask = temp_ledger["date"] >= p_date
            temp_ledger.loc[mask, "daily_balance"] -= p_amt
        return (temp_ledger["daily_balance"] >= self.min_balance).all()
