"""Evaluator logic for analyzing payment plan permutations."""

import pandas as pd
from typing import List, Tuple, Optional
from itertools import combinations
import logging


class PaymentPlan:
    """Represents a potential payment strategy."""

    def __init__(
        self,
        method: str,
        schedule: List[Tuple[pd.Timestamp, float]],
        option_id: str = "",
        spending_changes: Optional[List[str]] = None,
    ):
        """Initializes a Payment Plan configuration.

        Args:
            method (str): The payment method ('full_payment', 'installments', etc.)
            schedule (List[Tuple[pd.Timestamp, float]]): The chronological payment schedule.
            option_id (str): The optional installment ID.
            spending_changes (List[str], optional): Required spending changes.
        """
        self.method = method
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.option_id = option_id
        self.spending_changes = spending_changes or []

    @property
    def total_amount(self) -> float:
        """Returns total amount paid across the schedule."""
        return sum(amt for _, amt in self.schedule)

    @property
    def completion_date(self) -> pd.Timestamp:
        """Returns the date of the final payment."""
        return self.schedule[-1][0] if self.schedule else pd.Timestamp.max

    @property
    def start_date(self) -> pd.Timestamp:
        """Returns the date of the first payment."""
        return self.schedule[0][0] if self.schedule else pd.Timestamp.max

    @property
    def total_payments(self) -> int:
        """Returns the number of payments."""
        return len(self.schedule)

    def format_plan(self) -> str:
        """Formats the schedule to the specific 'YYYY-MM-DD:Amount' pipe-delimited standard."""
        if not self.schedule:
            return "none"
        return "|".join(
            [
                (
                    f"{date.strftime('%Y-%m-%d')}:{amt:.2f}".rstrip("0").rstrip(".")
                    if "." in f"{amt:.2f}"
                    else f"{date.strftime('%Y-%m-%d')}:{amt:.0f}"
                )
                for date, amt in self.schedule
            ]
        )


class Evaluator:
    """Evaluates and ranks valid payment permutations."""

    def __init__(
        self,
        simulator: Any,
        request: pd.Series,
        profile: pd.Series,
        options: pd.DataFrame,
    ):
        """Initializes the Evaluator.

        Args:
            simulator (Simulator): The initialized simulator engine.
            request (pd.Series): The active request context.
            profile (pd.Series): The active user profile.
            options (pd.DataFrame): The available installment options.
        """
        self.simulator = simulator
        self.request = request
        self.profile = profile
        self.options = options

        self.req_date = pd.to_datetime(self.request["request_date"])
        self.req_amount = float(self.request["requested_amount"])
        self.desired_date = pd.to_datetime(self.request["desired_completion_date"])

        methods_str = str(self.profile.get("payment_methods_user_will_consider", ""))
        self.allowed_methods = [m.strip() for m in methods_str.split("|")]

    def generate_permutations(self) -> List[PaymentPlan]:
        """Generates all structurally valid payment plan permutations.

        Returns:
            List[PaymentPlan]: Array of unvalidated payment plans.
        """
        plans = []

        if "full_payment" in self.allowed_methods:
            plans.append(
                PaymentPlan("full_payment", [(self.req_date, self.req_amount)])
            )

        if "wait" in self.allowed_methods or "full_payment" in self.allowed_methods:
            for i in range(1, 90):
                d = self.req_date + pd.Timedelta(days=i)
                plans.append(PaymentPlan("wait", [(d, self.req_amount)]))

        if (
            "partial_payment" in self.allowed_methods
            and str(self.request.get("allows_partial_payment", "false")).lower()
            == "true"
        ):
            safe_today = min(self.simulator.get_max_safe_deduction(), self.req_amount)
            if 0 < safe_today < self.req_amount:
                remainder = self.req_amount - safe_today
                for i in range(1, 90):
                    d = self.req_date + pd.Timedelta(days=i)
                    if d <= self.desired_date:
                        plans.append(
                            PaymentPlan(
                                "partial_payment",
                                [(self.req_date, safe_today), (d, remainder)],
                            )
                        )

        if "installments" in self.allowed_methods:
            req_opts = self.options[
                self.options["request_id"] == self.request["request_id"]
            ]
            max_months_val = self.profile.get("max_installment_months")
            max_months = (
                int(max_months_val)
                if pd.notna(max_months_val) and str(max_months_val).strip() != ""
                else float("inf")
            )

            for _, opt in req_opts.iterrows():
                num_payments = int(opt["number_of_payments"])
                if num_payments <= max_months:
                    first_date = pd.to_datetime(opt["first_payment_date"])
                    freq_days = (
                        int(opt["payment_frequency_days"])
                        if pd.notna(opt["payment_frequency_days"])
                        else 30
                    )
                    amt_per_payment = float(opt["payment_amount"])

                    schedule = [
                        (first_date + pd.Timedelta(days=i * freq_days), amt_per_payment)
                        for i in range(num_payments)
                    ]
                    plans.append(
                        PaymentPlan(
                            "installments",
                            schedule,
                            option_id=str(opt["payment_option_id"]),
                        )
                    )

        return plans

    def evaluate(self) -> Optional[PaymentPlan]:
        """Ranks all safe valid configurations to identify the optimal plan.

        Raises:
            ValueError: If contradictory spending modifications attempt logic overlap.

        Returns:
            Optional[PaymentPlan]: The optimal plan, or None if completely unaffordable.
        """
        all_plans = self.generate_permutations()

        # Filter purely safe baseline plans
        safe_plans = [
            p for p in all_plans if self.simulator.check_schedule_safe(p.schedule)
        ]

        # Fallback layer: Generate logical spending modifications
        if not safe_plans:
            from simulation import Simulator

            flexible = self.simulator.events[
                (self.simulator.events["flexibility"] == "flexible")
                & (self.simulator.events["direction"].str.lower() == "debit")
            ]

            prot_str = str(self.profile.get("expense_categories_to_protect", ""))
            protected = set(c.strip() for c in prot_str.split("|") if c.strip())

            stop_str = str(
                self.profile.get("expense_categories_user_is_willing_to_stop", "")
            )
            can_stop = set(c.strip() for c in stop_str.split("|") if c.strip())

            reduce_str = str(
                self.profile.get("expense_categories_user_is_willing_to_reduce", "")
            )
            can_reduce = set(c.strip() for c in reduce_str.split("|") if c.strip())

            actions = []
            for _, row in flexible.iterrows():
                cat = row["category"]
                if cat in protected:
                    continue
                if cat in can_stop:
                    actions.append(
                        {
                            "type": "stop",
                            "event_id": row["event_id"],
                            "desc": f"stop:{row['event_id']}",
                        }
                    )
                if cat in can_reduce and pd.notna(row.get("minimum_allowed_amount")):
                    min_amt = float(row["minimum_allowed_amount"])
                    if float(row["amount"]) > min_amt:
                        amt_str = (
                            f"{min_amt:.2f}".rstrip("0").rstrip(".")
                            if "." in f"{min_amt:.2f}"
                            else f"{min_amt:.0f}"
                        )
                        actions.append(
                            {
                                "type": "reduce",
                                "event_id": row["event_id"],
                                "new_amount": min_amt,
                                "desc": f"reduce_to:{row['event_id']}:{amt_str}",
                            }
                        )

            combos = []
            for r in range(1, 4):
                for combo in combinations(actions, r):
                    ids = [a["event_id"] for a in combo]
                    if len(ids) != len(set(ids)):
                        logging.error(
                            f"Validation Error: logic conflict with event IDs {ids}"
                        )
                        raise ValueError(
                            f"Validation Error: logic conflict with event IDs {ids}"
                        )
                    combos.append(combo)
            combos.sort(key=len)

            for combo in combos:
                temp_sim = Simulator(self.profile, self.simulator.events)
                temp_sim.generate_ledger(self.req_date)

                for a in combo:
                    evt_id = a["event_id"]
                    targets = self.simulator.events[
                        self.simulator.events["event_id"] == evt_id
                    ]
                    for _, row in targets.iterrows():
                        evt_date = pd.to_datetime(row["settlement_date"])
                        if (
                            self.req_date
                            <= evt_date
                            <= self.req_date + pd.Timedelta(days=89)
                        ):
                            orig_amt = float(row["amount"])
                            savings = (
                                orig_amt
                                if a["type"] == "stop"
                                else (orig_amt - a["new_amount"])
                            )
                            temp_sim.ledger.loc[
                                temp_sim.ledger["date"] >= evt_date, "daily_balance"
                            ] += savings

                for p in all_plans:
                    if temp_sim.check_schedule_safe(p.schedule):
                        p.spending_changes = [a["desc"] for a in combo]
                        safe_plans.append(p)
                        break
                if safe_plans:
                    break

        if not safe_plans:
            return None

        def sort_key(p: PaymentPlan) -> Tuple:
            """Sorts plans strictly by HackerRank requested priority heuristics."""
            opt_id_str = p.option_id
            opt_id = 999999
            if opt_id_str and "opt" in opt_id_str.lower():
                try:
                    opt_id = int("".join(filter(str.isdigit, opt_id_str)))
                except ValueError:
                    pass

            start_date_int = int(p.start_date.timestamp())

            return (
                0 if p.completion_date <= self.desired_date else 1,
                len(p.spending_changes) > 0,
                p.total_amount,
                start_date_int,
                p.total_payments,
                opt_id,
            )

        safe_plans.sort(key=sort_key)
        return safe_plans[0]
