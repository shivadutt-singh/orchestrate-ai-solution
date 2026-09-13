import pandas as pd
import os


class DataLoader:
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir

        # Datasets
        self.requests_df = None
        self.profiles_df = None
        self.events_df = None
        self.exchange_rates_df = None
        self.payment_options_df = None
        self.messages_df = None
        self.images_df = None

    def load_all(self, confirmed_pending_event_ids=None):
        """
        Loads all 7 CSV files and performs the required preprocessing:
        1. Parsing dates strictly in YYYY-MM-DD
        2. Joining requests with profiles
        3. Converting currencies in financial_events
        4. Filtering out invalid event statuses
        """
        self.requests_df = pd.read_csv(os.path.join(self.data_dir, "requests.csv"))
        self.profiles_df = pd.read_csv(
            os.path.join(self.data_dir, "financial_profiles.csv")
        )
        self.events_df = pd.read_csv(
            os.path.join(self.data_dir, "financial_events.csv")
        )
        self.exchange_rates_df = pd.read_csv(
            os.path.join(self.data_dir, "exchange_rates.csv")
        )
        self.payment_options_df = pd.read_csv(
            os.path.join(self.data_dir, "request_payment_options.csv")
        )
        self.messages_df = pd.read_csv(os.path.join(self.data_dir, "messages.csv"))
        self.images_df = pd.read_csv(os.path.join(self.data_dir, "images.csv"))

        self._parse_dates()
        self._join_requests_profiles()

        # Integration: Apply AI-extracted JSON actions and Vision updates to mutate financial_events.csv
        from ai_extractors import apply_extracted_actions

        self.events_df = apply_extracted_actions(
            self.events_df, self.images_df, self.messages_df, self.data_dir
        )

        self._convert_currencies()
        self._filter_events(confirmed_pending_event_ids)

    def _parse_dates(self):
        """Parse all dates strictly in YYYY-MM-DD format."""
        date_cols = {
            "requests_df": ["request_date", "desired_completion_date"],
            "events_df": ["event_date", "settlement_date"],
            "exchange_rates_df": ["rate_date"],
        }
        for df_name, cols in date_cols.items():
            df = getattr(self, df_name)
            for col in cols:
                if col in df.columns:
                    # Convert to datetime then to exact string representation of YYYY-MM-DD
                    df[col] = pd.to_datetime(
                        df[col], format="%Y-%m-%d", errors="coerce"
                    ).dt.strftime("%Y-%m-%d")
            setattr(self, df_name, df)

    def _join_requests_profiles(self):
        """
        Join requests.csv with financial_profiles.csv on user_id to attach
        home_currency, minimum_balance_to_keep, and payment_methods_user_will_consider.
        """
        cols_to_add = [
            "user_id",
            "home_currency",
            "minimum_balance_to_keep",
            "payment_methods_user_will_consider",
        ]
        profiles_subset = self.profiles_df[cols_to_add]
        self.requests_df = self.requests_df.merge(
            profiles_subset, on="user_id", how="left"
        )

    def _convert_currencies(self):
        """
        Convert all amounts in financial_events.csv to the user's home_currency based on the event date.
        """
        # Attach home_currency to events
        events_with_profile = self.events_df.merge(
            self.profiles_df[["user_id", "home_currency"]], on="user_id", how="left"
        )

        # Use event_date as the basis for the exchange rate
        events_with_profile["date_for_rate"] = events_with_profile["event_date"]

        # Merge with exchange rates
        merged = events_with_profile.merge(
            self.exchange_rates_df,
            left_on=["date_for_rate", "currency", "home_currency"],
            right_on=["rate_date", "from_currency", "to_currency"],
            how="left",
        )

        # Where currency == home_currency, rate is 1.0
        same_currency = merged["currency"] == merged["home_currency"]
        merged.loc[same_currency, "rate"] = 1.0

        # Convert amounts
        merged["amount_in_home_currency"] = merged["amount"] * merged["rate"]

        # Replace original amount and currency with the converted values
        self.events_df = merged.drop(
            columns=[
                "date_for_rate",
                "rate_date",
                "from_currency",
                "to_currency",
                "rate",
            ]
        )
        self.events_df["original_amount"] = self.events_df["amount"]
        self.events_df["original_currency"] = self.events_df["currency"]
        self.events_df["amount"] = self.events_df["amount_in_home_currency"]
        self.events_df["currency"] = self.events_df["home_currency"]

        # Clean up temporary column
        self.events_df = self.events_df.drop(
            columns=["amount_in_home_currency", "home_currency"]
        )

    def _filter_events(self, confirmed_pending_ids=None):
        """
        Filter out any records in financial_events.csv where status is failed, cancelled,
        pending (unless explicitly confirmed by a message), or unrealized.
        """
        if confirmed_pending_ids is None:
            confirmed_pending_ids = set()

            # Phase 1 simplistic heuristic: if a pending event is explicitly referenced by a message,
            # we consider it potentially confirmed (until the LLM extraction module overrides this).
            if (
                self.messages_df is not None
                and "related_event_id" in self.messages_df.columns
            ):
                has_message = self.messages_df["related_event_id"].dropna().unique()
                pending_events = self.events_df[self.events_df["status"] == "pending"][
                    "event_id"
                ]
                confirmed_pending_ids = set(pending_events).intersection(
                    set(has_message)
                )

        drop_statuses = ["failed", "cancelled", "unrealized"]
        mask_drop_statuses = self.events_df["status"].isin(drop_statuses)

        mask_unconfirmed_pending = (self.events_df["status"] == "pending") & (
            ~self.events_df["event_id"].isin(confirmed_pending_ids)
        )

        mask_to_keep = ~(mask_drop_statuses | mask_unconfirmed_pending)

        self.events_df = self.events_df[mask_to_keep].copy()


if __name__ == "__main__":
    loader = DataLoader()
    loader.load_all()
    print(
        f"Data loading complete. Loaded {len(loader.requests_df)} requests and {len(loader.events_df)} valid events."
    )
