import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta


class LedgerSimulator:
    def __init__(self, profile, events_df):
        """
        profile: pd.Series or dict containing 'current_available_balance' and 'minimum_balance_to_keep'
        events_df: pd.DataFrame of the user's financial events (already filtered and currency-converted)
        """
        self.profile = profile
        self.events_df = events_df.copy()

        # Ensure dates are datetime objects for easier math
        self.events_df["settlement_date"] = pd.to_datetime(
            self.events_df["settlement_date"]
        ).dt.date

        self.daily_balances = {}
        self.minimum_balance = float(self.profile.get("minimum_balance_to_keep", 0.0))
        self.start_date = None
        self.end_date = None

    def _detect_and_project_recurrences(self, start_date, end_date):
        """
        Simplistic recurrence projection:
        If multiple events have the same description and category and occur roughly a month apart,
        project them forward into the 90-day window.
        """
        projected_events = []

        # Group by description to find recurring patterns
        for desc, group in self.events_df.groupby("description"):
            if len(group) >= 2:
                # Sort by date
                group = group.sort_values("settlement_date")
                dates = group["settlement_date"].tolist()

                # Check if they are roughly monthly
                diffs = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
                # If all diffs are between 27 and 33 days, assume monthly recurrence
                if all(27 <= d <= 33 for d in diffs):
                    last_event = group.iloc[-1]
                    next_date = last_event["settlement_date"] + relativedelta(months=1)

                    while next_date <= end_date:
                        if next_date >= start_date:
                            new_event = last_event.copy()
                            new_event["settlement_date"] = next_date
                            new_event["status"] = "projected"
                            projected_events.append(new_event)
                        next_date += relativedelta(months=1)

        if projected_events:
            projected_df = pd.DataFrame(projected_events)
            self.events_df = pd.concat(
                [self.events_df, projected_df], ignore_index=True
            )

    def simulate_90_days(self, start_date_str):
        """
        Generates an array (or dictionary mapping date to balance) of 90 daily balances.
        """
        if isinstance(start_date_str, str):
            self.start_date = datetime.datetime.strptime(
                start_date_str, "%Y-%m-%d"
            ).date()
        else:
            self.start_date = start_date_str

        self.end_date = self.start_date + datetime.timedelta(days=89)

        # 1. Project recurrences for the window
        self._detect_and_project_recurrences(self.start_date, self.end_date)

        # 2. Filter events to the 90-day window
        window_events = self.events_df[
            (self.events_df["settlement_date"] >= self.start_date)
            & (self.events_df["settlement_date"] <= self.end_date)
        ]

        # 3. Map available_balance to day 0 (start_date)
        current_balance = float(self.profile.get("current_available_balance", 0.0))

        # Aggregate net cash flow per day in the window
        # credit is positive, debit is negative
        daily_cashflows = {}
        for _, row in window_events.iterrows():
            amt = float(row["amount"])
            direction = row["direction"].lower()
            date = row["settlement_date"]

            if direction == "debit":
                amt = -amt

            daily_cashflows[date] = daily_cashflows.get(date, 0.0) + amt

        # 4. Generate the 90-day balance array
        self.daily_balances = {}

        for i in range(90):
            current_day = self.start_date + datetime.timedelta(days=i)
            # Apply cash flow for the day
            net_flow = daily_cashflows.get(current_day, 0.0)
            current_balance += net_flow
            self.daily_balances[current_day] = current_balance

        return self.daily_balances

    def is_simulation_safe(self):
        """
        A simulation is ONLY safe if the daily balance is >= minimum_balance_to_keep
        on every single day of the 90-day array.
        """
        if not self.daily_balances:
            raise ValueError("Must run simulate_90_days first.")

        return all(bal >= self.minimum_balance for bal in self.daily_balances.values())

    def get_safe_deduction(self, deduction_date_str):
        """
        Calculate the maximum float value that can be subtracted on a specific date
        without pushing any subsequent day below the minimum balance.
        """
        if not self.daily_balances:
            raise ValueError("Must run simulate_90_days first.")

        if isinstance(deduction_date_str, str):
            deduction_date = datetime.datetime.strptime(
                deduction_date_str, "%Y-%m-%d"
            ).date()
        else:
            deduction_date = deduction_date_str

        if deduction_date < self.start_date or deduction_date > self.end_date:
            raise ValueError("Deduction date is outside the 90-day simulation window.")

        # The maximum safe deduction is the minimum of (balance - minimum_balance)
        # for all days from deduction_date to the end of the 90-day window.

        min_headroom = float("inf")

        for i in range(90):
            current_day = self.start_date + datetime.timedelta(days=i)
            if current_day >= deduction_date:
                headroom = self.daily_balances[current_day] - self.minimum_balance
                if headroom < min_headroom:
                    min_headroom = headroom

        return max(0.0, min_headroom)
