from abc import ABC, abstractmethod
import datetime
from typing import List, Optional, Tuple
import pandas as pd


class PaymentPlan:
    def __init__(
        self,
        method: str,
        schedule: List[Tuple[datetime.date, float]],
        option_id: str = "",
        spending_changes: List[str] = None,
    ):
        self.method = method
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.option_id = option_id
        self.spending_changes = spending_changes or []

    @property
    def total_amount(self) -> float:
        return sum(amt for _, amt in self.schedule)

    @property
    def completion_date(self) -> datetime.date:
        return self.schedule[-1][0] if self.schedule else datetime.date.max

    @property
    def start_date(self) -> datetime.date:
        return self.schedule[0][0] if self.schedule else datetime.date.max

    @property
    def total_payments(self) -> int:
        return len(self.schedule)

    def format_plan(self) -> str:
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


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, context: "EvaluatorContext") -> List[PaymentPlan]:
        pass


class FullPaymentStrategy(EvaluationStrategy):
    def evaluate(self, ctx: "EvaluatorContext") -> List[PaymentPlan]:
        if "full_payment" not in ctx.allowed_methods:
            return []

        schedule = [(ctx.request_date, ctx.requested_amount)]
        if ctx.simulator.calculate_max_safe_deduction() >= ctx.requested_amount:
            return [PaymentPlan("full_payment", schedule)]
        else:
            adj = ctx.adjustments_solver.find_minimum_adjustments(schedule)
            if adj is not None:
                return [PaymentPlan("full_payment", schedule, spending_changes=adj)]
        return []


class WaitStrategy(EvaluationStrategy):
    def evaluate(self, ctx: "EvaluatorContext") -> List[PaymentPlan]:
        if "full_payment" not in ctx.allowed_methods:
            return []

        for i in range(1, 90):
            d = ctx.request_date + datetime.timedelta(days=i)
            schedule = [(d, ctx.requested_amount)]

            # Check natively safe
            test_ledger = ctx.simulator.ledger_array.copy()
            for j in range(90):
                day = ctx.request_date + datetime.timedelta(days=j)
                if day >= d:
                    test_ledger[j] -= ctx.requested_amount

            if ctx.simulator.is_path_safe(test_ledger, ctx.simulator.minimum_balance):
                return [PaymentPlan("wait", schedule)]
            else:
                adj = ctx.adjustments_solver.find_minimum_adjustments(schedule)
                if adj is not None:
                    return [PaymentPlan("wait", schedule, spending_changes=adj)]
        return []


class PartialPaymentStrategy(EvaluationStrategy):
    def evaluate(self, ctx: "EvaluatorContext") -> List[PaymentPlan]:
        if (
            "partial_payment" not in ctx.allowed_methods
            or str(ctx.request.get("allows_partial_payment", "false")).lower() != "true"
        ):
            return []

        safe_today = min(
            ctx.simulator.calculate_max_safe_deduction(), ctx.requested_amount
        )
        if 0 < safe_today < ctx.requested_amount:
            remainder = ctx.requested_amount - safe_today
            for i in range(1, 90):
                d = ctx.request_date + datetime.timedelta(days=i)
                if d > ctx.desired_date:
                    break

                schedule = [(ctx.request_date, safe_today), (d, remainder)]

                test_ledger = ctx.simulator.ledger_array.copy()
                for j in range(90):
                    test_day = ctx.request_date + datetime.timedelta(days=j)
                    if test_day >= ctx.request_date:
                        test_ledger[j] -= safe_today
                    if test_day >= d:
                        test_ledger[j] -= remainder

                if ctx.simulator.is_path_safe(
                    test_ledger, ctx.simulator.minimum_balance
                ):
                    return [PaymentPlan("partial_payment", schedule)]
                else:
                    adj = ctx.adjustments_solver.find_minimum_adjustments(schedule)
                    if adj is not None:
                        return [
                            PaymentPlan(
                                "partial_payment", schedule, spending_changes=adj
                            )
                        ]
        return []


class InstallmentsStrategy(EvaluationStrategy):
    def evaluate(self, ctx: "EvaluatorContext") -> List[PaymentPlan]:
        plans = []
        if "installments" not in ctx.allowed_methods:
            return plans

        req_opts = ctx.payment_options[
            ctx.payment_options["request_id"] == ctx.request["request_id"]
        ]
        max_months_val = ctx.profile.get("max_installment_months")
        max_months = (
            int(max_months_val)
            if pd.notna(max_months_val) and str(max_months_val).strip() != ""
            else float("inf")
        )

        for _, opt in req_opts.iterrows():
            num_payments = int(opt["number_of_payments"])
            if num_payments > max_months:
                continue

            first_date = datetime.datetime.strptime(
                str(opt["first_payment_date"]), "%Y-%m-%d"
            ).date()
            freq_days = (
                int(opt["payment_frequency_days"])
                if pd.notna(opt["payment_frequency_days"])
                else 30
            )
            amt_per_payment = float(opt["payment_amount"])

            schedule = []
            for i in range(num_payments):
                schedule.append(
                    (
                        first_date + datetime.timedelta(days=i * freq_days),
                        amt_per_payment,
                    )
                )

            test_ledger = ctx.simulator.ledger_array.copy()
            for p_date, p_amt in schedule:
                for j in range(90):
                    day = ctx.request_date + datetime.timedelta(days=j)
                    if day >= p_date:
                        test_ledger[j] -= p_amt

            if ctx.simulator.is_path_safe(test_ledger, ctx.simulator.minimum_balance):
                plans.append(
                    PaymentPlan(
                        "installments",
                        schedule,
                        option_id=str(opt["payment_option_id"]),
                    )
                )
            else:
                adj = ctx.adjustments_solver.find_minimum_adjustments(schedule)
                if adj is not None:
                    plans.append(
                        PaymentPlan(
                            "installments",
                            schedule,
                            option_id=str(opt["payment_option_id"]),
                            spending_changes=adj,
                        )
                    )

        return plans


class EvaluatorContext:
    def __init__(
        self,
        request: pd.Series,
        profile: pd.Series,
        simulator: "Simulator",
        options: pd.DataFrame,
    ):
        self.request = request
        self.profile = profile
        self.simulator = simulator
        self.payment_options = options

        self.request_date = pd.to_datetime(self.request["request_date"]).date()
        self.desired_date = pd.to_datetime(
            self.request["desired_completion_date"]
        ).date()
        self.requested_amount = float(self.request["requested_amount"])

        methods_str = str(self.profile.get("payment_methods_user_will_consider", ""))
        self.allowed_methods = [m.strip() for m in methods_str.split("|")]
        self.strategies = [
            FullPaymentStrategy(),
            WaitStrategy(),
            PartialPaymentStrategy(),
            InstallmentsStrategy(),
        ]
        from engine.adjustments import AdjustmentsSolver

        self.adjustments_solver = AdjustmentsSolver(self)

    def evaluate_all(self) -> Optional[PaymentPlan]:
        safe_plans = []
        for strategy in self.strategies:
            safe_plans.extend(strategy.evaluate(self))

        if not safe_plans:
            return None

        def rank_key(plan: PaymentPlan):
            return (
                0 if plan.completion_date <= self.desired_date else 1,
                len(plan.spending_changes),
                plan.total_amount,
                plan.start_date,
                plan.total_payments,
                plan.option_id,
            )

        safe_plans.sort(key=rank_key)
        return safe_plans[0]
