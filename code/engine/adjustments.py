import itertools
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
from engine.evaluator import PaymentPlan, EvaluatorContext


class AdjustmentsSolver:
    def __init__(self, ctx: EvaluatorContext):
        self.ctx = ctx
        self.simulator = ctx.simulator
        self.profile = ctx.profile

        prot_str = str(self.profile.get("expense_categories_to_protect", ""))
        self.protected = set(c.strip() for c in prot_str.split("|") if c.strip())

        reduce_str = str(
            self.profile.get("expense_categories_user_is_willing_to_reduce", "")
        )
        self.can_reduce = set(c.strip() for c in reduce_str.split("|") if c.strip())

        stop_str = str(
            self.profile.get("expense_categories_user_is_willing_to_stop", "")
        )
        self.can_stop = set(c.strip() for c in stop_str.split("|") if c.strip())

    def _get_candidates(self) -> List[Dict[str, Any]]:
        actions = []
        events_df = self.simulator.events

        flexible = events_df[
            (events_df["flexibility"] == "flexible")
            & (events_df["direction"].str.lower() == "debit")
        ]

        for _, row in flexible.iterrows():
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

        unique = {a["desc"]: a for a in actions}
        return list(unique.values())

    def _get_valid_combos(self, actions: List[Dict[str, Any]]) -> List[tuple]:
        combos = []
        for r in range(1, 4):
            for combo in itertools.combinations(actions, r):
                ids = [a["event_id"] for a in combo]
                if len(ids) != len(set(ids)):
                    raise ValueError(
                        f"Validation Error: Cannot stop and reduce the same event ID in a single adjustment plan. IDs: {ids}"
                    )
                combos.append(combo)
        combos.sort(key=len)
        return combos

    def _test_combo(self, schedule: List[tuple], combo: tuple) -> bool:
        ledger = self.simulator.ledger_array.copy()

        for action in combo:
            event_id = action["event_id"]
            targets = self.simulator.events[
                (self.simulator.events["event_id"] == event_id)
                & (
                    pd.to_datetime(self.simulator.events["settlement_date"]).dt.date
                    >= self.simulator.start_date
                )
                & (
                    pd.to_datetime(self.simulator.events["settlement_date"]).dt.date
                    <= self.simulator.start_date + datetime.timedelta(days=89)
                )
            ]
            for _, row in targets.iterrows():
                evt_date = pd.to_datetime(row["settlement_date"]).date()
                orig_amt = float(row["amount"])
                savings = (
                    orig_amt
                    if action["type"] == "stop"
                    else (orig_amt - action["new_amount"])
                )

                for j in range(90):
                    day = self.simulator.start_date + datetime.timedelta(days=j)
                    if day >= evt_date:
                        ledger[j] += savings

        for p_date, p_amt in schedule:
            for j in range(90):
                day = self.simulator.start_date + datetime.timedelta(days=j)
                if day >= p_date:
                    ledger[j] -= p_amt

        return self.simulator.is_path_safe(ledger, self.simulator.minimum_balance)

    def find_minimum_adjustments(self, schedule: List[tuple]) -> Optional[List[str]]:
        candidates = self._get_candidates()
        if not candidates:
            return None

        combos = self._get_valid_combos(candidates)
        for combo in combos:
            if self._test_combo(schedule, combo):
                return [a["desc"] for a in combo]
        return None
