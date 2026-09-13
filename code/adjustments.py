import itertools
import pandas as pd


class AdjustmentsEngine:
    def __init__(self, simulator, profile):
        self.simulator = simulator
        self.profile = profile

        # Parse profile categories
        prot_str = str(profile.get("expense_categories_to_protect", ""))
        self.protected = set(c.strip() for c in prot_str.split("|") if c.strip())

        reduce_str = str(
            profile.get("expense_categories_user_is_willing_to_reduce", "")
        )
        self.can_reduce = set(c.strip() for c in reduce_str.split("|") if c.strip())

        stop_str = str(profile.get("expense_categories_user_is_willing_to_stop", ""))
        self.can_stop = set(c.strip() for c in stop_str.split("|") if c.strip())

    def get_possible_actions(self):
        """
        Identify all possible single adjustment actions based on the events and user profile.
        """
        actions = []
        events_df = self.simulator.events_df

        flexible_events = events_df[
            (events_df["flexibility"] == "flexible")
            & (events_df["direction"].str.lower() == "debit")
        ]

        for _, row in flexible_events.iterrows():
            event_id = row["event_id"]
            cat = row["category"]

            if cat in self.protected:
                continue

            if cat in self.can_stop:
                actions.append(
                    {"type": "stop", "event_id": event_id, "desc": f"stop:{event_id}"}
                )

            if cat in self.can_reduce and pd.notna(row.get("minimum_allowed_amount")):
                min_amt = float(row["minimum_allowed_amount"])
                if float(row["amount"]) > min_amt:
                    # Format strictly as required
                    # We want to avoid .0 if it's an integer, but format correctly.
                    # e.g., 100.0 -> 100, 100.5 -> 100.5
                    amt_str = (
                        f"{min_amt:.2f}".rstrip("0").rstrip(".")
                        if "." in f"{min_amt:.2f}"
                        else f"{min_amt:.0f}"
                    )
                    actions.append(
                        {
                            "type": "reduce_to",
                            "event_id": event_id,
                            "new_amount": min_amt,
                            "desc": f"reduce_to:{event_id}:{amt_str}",
                        }
                    )

        # Remove duplicates
        unique_actions = {a["desc"]: a for a in actions}
        return list(unique_actions.values())

    def get_valid_combinations(self, actions, max_len=3):
        """
        Generate combinations of 1 to `max_len` actions, ensuring we don't
        stop and reduce the same event.
        """
        valid_combos = []

        for r in range(1, max_len + 1):
            for combo in itertools.combinations(actions, r):
                event_ids = [a["event_id"] for a in combo]
                if len(event_ids) == len(set(event_ids)):
                    valid_combos.append(combo)

        # Sort combinations by length (try 1 change first, then 2, etc.)
        valid_combos.sort(key=len)
        return valid_combos

    def test_adjustment_plan(self, schedule, adjustment_combo):
        """
        Test if a payment schedule is safe under a specific combination of adjustments.
        """
        balances = self.simulator.daily_balances.copy()

        # Apply the savings from the adjustments
        for action in adjustment_combo:
            event_id = action["event_id"]

            target_events = self.simulator.events_df[
                (self.simulator.events_df["event_id"] == event_id)
                & (
                    self.simulator.events_df["settlement_date"]
                    >= self.simulator.start_date
                )
                & (
                    self.simulator.events_df["settlement_date"]
                    <= self.simulator.end_date
                )
            ]

            for _, row in target_events.iterrows():
                evt_date = row["settlement_date"]
                original_amt = float(row["amount"])

                if action["type"] == "stop":
                    savings = original_amt
                elif action["type"] == "reduce_to":
                    savings = original_amt - action["new_amount"]

                for day in balances:
                    if day >= evt_date:
                        balances[day] += savings

        # Apply the payment schedule deductions
        for p_date, p_amt in schedule:
            for day in balances:
                if day >= p_date:
                    balances[day] -= p_amt

        return all(bal >= self.simulator.minimum_balance for bal in balances.values())

    def find_safe_adjustments(self, payment_schedule):
        """
        Iteratively tests combinations of up to 3 adjustments to make the schedule safe.
        Returns the list of adjustment string descriptions, or None if impossible.
        """
        actions = self.get_possible_actions()
        if not actions:
            return None

        combos = self.get_valid_combinations(actions, max_len=3)

        for combo in combos:
            if self.test_adjustment_plan(payment_schedule, combo):
                return [a["desc"] for a in combo]

        return None
