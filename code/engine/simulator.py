import pandas as pd
import datetime
from typing import Dict, List, Any
from dateutil.relativedelta import relativedelta


class Simulator:
    def __init__(self, profile: pd.Series, events: pd.DataFrame):
        self.profile = profile
        self.events = events.copy()
        self.minimum_balance = float(self.profile.get("minimum_balance_to_keep", 0.0))
        self.ledger_array: List[float] = []
        self.start_date: datetime.date = None

    def _project_recurrences(self, end_date: datetime.date) -> None:
        projected = []
        for desc, group in self.events.groupby("description"):
            if len(group) >= 2:
                group = group.sort_values("settlement_date")
                dates = pd.to_datetime(group["settlement_date"]).dt.date.tolist()
                diffs = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
                if all(27 <= d <= 33 for d in diffs):
                    last_event = group.iloc[-1].copy()
                    next_date = dates[-1] + relativedelta(months=1)
                    while next_date <= end_date:
                        if next_date >= self.start_date:
                            new_event = last_event.copy()
                            new_event["settlement_date"] = next_date.strftime(
                                "%Y-%m-%d"
                            )
                            new_event["status"] = "projected"
                            projected.append(new_event)
                        next_date += relativedelta(months=1)

        if projected:
            self.events = pd.concat(
                [self.events, pd.DataFrame(projected)], ignore_index=True
            )

    def generate_90_day_ledger(self, start_date: datetime.date) -> List[float]:
        self.start_date = start_date
        end_date = start_date + datetime.timedelta(days=89)

        self._project_recurrences(end_date)

        # Aggregate net cash flow
        daily_cashflows: Dict[datetime.date, float] = {}
        for _, row in self.events.iterrows():
            evt_date = pd.to_datetime(row["settlement_date"]).date()
            if start_date <= evt_date <= end_date:
                amt = float(row["amount"])
                if str(row["direction"]).lower() == "debit":
                    amt = -amt
                # "Ignore pending credits"
                if (
                    str(row["direction"]).lower() == "credit"
                    and str(row["status"]).lower() == "pending"
                ):
                    continue
                daily_cashflows[evt_date] = daily_cashflows.get(evt_date, 0.0) + amt

        current_balance = float(self.profile.get("current_available_balance", 0.0))
        self.ledger_array = []

        for i in range(90):
            day = start_date + datetime.timedelta(days=i)
            current_balance += daily_cashflows.get(day, 0.0)
            self.ledger_array.append(current_balance)

        return self.ledger_array

    def is_path_safe(self, ledger_array: List[float], min_balance: float) -> bool:
        return all(bal >= min_balance for bal in ledger_array)

    def calculate_max_safe_deduction(self) -> float:
        min_headroom = min(bal - self.minimum_balance for bal in self.ledger_array)
        return max(0.0, min_headroom)

    def find_earliest_full_payment_date(self, requested_amount: float) -> datetime.date:
        for i in range(90):
            # Safe deduction at day i means from day i to 89, the minimum headroom >= requested_amount
            headroom = min(
                self.ledger_array[j] - self.minimum_balance for j in range(i, 90)
            )
            if headroom >= requested_amount:
                return self.start_date + datetime.timedelta(days=i)
        return None
