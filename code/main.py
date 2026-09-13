"""Orchestration Module for Buy or Wait."""

import os
import sys
import logging
import pandas as pd
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl import ETL
from ai_layer import AILayer
from simulation import Simulator
from evaluator import Evaluator
from config import OUTPUT_CSV_PATH, USAGE_REPORT_PATH, OUTPUT_COLUMNS
from config import NVIDIA_BASE_URL, NVIDIA_API_KEY, TEXT_MODELS_TO_TRY, VISION_MODELS_TO_TRY


class TokenTracker:
    """Tracks token consumption and API costs across all remote model inferences."""

    def __init__(self):
        """Initializes usage tracking."""
        self.usage: List[Dict[str, Any]] = []

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Appends a completed inference cycle to the tracker.

        Args:
            model (str): The requested model identifier.
            input_tokens (int): The prompt payload length.
            output_tokens (int): The generated string length.
        """
        if "llama-3.1-70b" in model or "llama-3.3-70b" in model or "nemotron" in model or "llama3-70b" in model:
            cost = (input_tokens / 1_000_000) * 0.88 + (
                output_tokens / 1_000_000
            ) * 0.88
        elif "llama-3.2-90b" in model or "llama-3.2-11b" in model:
            cost = (input_tokens / 1_000_000) * 1.50 + (
                output_tokens / 1_000_000
            ) * 1.50
        elif "8b" in model or "3b" in model or "7b" in model:
            cost = (input_tokens / 1_000_000) * 0.15 + (
                output_tokens / 1_000_000
            ) * 0.15
        else:
            cost = 0.0

        self.usage.append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
            }
        )

    def write_report(self, filepath: str, total_requests: int) -> None:
        """Generates the Markdown-formatted financial telemetry summary.

        Args:
            filepath (str): Target output file path.
            total_requests (int): The denominator of total payload cycles.
        """
        total_input = sum(u["input_tokens"] for u in self.usage)
        total_output = sum(u["output_tokens"] for u in self.usage)
        total_cost = sum(u["cost"] for u in self.usage)
        models = list(set(u["model"] for u in self.usage))

        avg_tokens = (total_input + total_output) / max(1, total_requests)
        avg_cost = total_cost / max(1, total_requests)

        report = f"""# Token Usage Report

| Metric | Value |
| --- | --- |
| Models Used | {', '.join(models) if models else 'None'} |
| Total Input Tokens | {total_input} |
| Total Output Tokens | {total_output} |
| Total Tokens | {total_input + total_output} |
| Average Tokens per Request | {avg_tokens:.2f} |
| Estimated Cost | ${total_cost:.4f} |
| Average Cost per Request | ${avg_cost:.4f} |
"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)


tracker = TokenTracker()


def patch_openai() -> None:
    """Intercepts standard OpenAI payload routes to track bandwidth costs."""
    try:
        import openai

        original_create = openai.resources.chat.completions.Completions.create

        def wrapped_create(*args: Any, **kwargs: Any) -> Any:
            res = original_create(*args, **kwargs)
            if hasattr(res, "usage") and res.usage:
                tracker.add(
                    kwargs.get("model", "unknown"),
                    res.usage.prompt_tokens,
                    res.usage.completion_tokens,
                )
            return res

        openai.resources.chat.completions.Completions.create = wrapped_create
    except ImportError:
        pass


def generate_explanation(
    request_text: str,
    plan_method: str,
    amount_safe: float,
    earliest_date: str,
    changes: str,
    req_date: str,
    model_id: str,
) -> str:
    """Generates a contextual, one-sentence justification.

    Args:
        request_text (str): The initial user payload prompt.
        plan_method (str): The computed optimal method.
        amount_safe (float): Immediate affordable cash flow.
        earliest_date (str): Projected settlement date.
        changes (str): Identified adjustment permutations.
        req_date (str): The context origin date.
        model_id (str): The active NLP model ID.

    Returns:
        str: Concise decision explanation.
    """
    if not model_id:
        return "The recommendation was determined based on your projected 90-day cash flow and preferences."

    try:
        from openai import OpenAI

        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

        prompt = (
            f"Request: {request_text}\nRecommendation: {plan_method}\nAmount safe today: {amount_safe}\n"
            f"Earliest full payment date: {earliest_date}\nSpending changes required: {changes}\nRequest Date: {req_date}\n\n"
            "Write exactly ONE concise sentence explaining this financial decision based strictly on the facts provided."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a financial explainer. Output exactly 1 sentence. Do not offer financial advice.",
            },
            {"role": "user", "content": prompt},
        ]

        res = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=100,
            temperature=0.0,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        logging.warning(f"Explanation generator error: {e}")
        return "The recommendation was determined based on your projected 90-day cash flow and preferences."


def get_working_model(client, model_list: List[str]) -> str:
    """Iterates through an array of models and pings the API to find the first working model.
    
    Args:
        client: The OpenAI client pointing to NIM.
        model_list (List[str]): Array of model strings to attempt.
        
    Returns:
        str: The first model string that executes successfully.
        
    Raises:
        RuntimeError: If all models fail.
    """
    for model_name in model_list:
        try:
            logging.info(f"Pinging model {model_name}...")
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            logging.info(f"Successfully connected to {model_name}.")
            return model_name
        except Exception as e:
            error_str = str(e)
            if "410" in error_str or "404" in error_str or "401" in error_str:
                logging.warning(f"Model {model_name} ping failed: {e}")
                continue
            logging.warning(f"Model {model_name} ping failed with unexpected error: {e}")
            
    raise RuntimeError("No available models found for this API key.")


def main() -> None:
    """Main Orchestrator Entrypoint."""
    logging.info("Initializing Orchestration Engine...")
    patch_openai()

    try:
        from openai import OpenAI
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        ACTIVE_TEXT_MODEL = get_working_model(client, TEXT_MODELS_TO_TRY)
        ACTIVE_VISION_MODEL = get_working_model(client, VISION_MODELS_TO_TRY)
        logging.info(f"Selected Active Text Model: {ACTIVE_TEXT_MODEL}")
        logging.info(f"Selected Active Vision Model: {ACTIVE_VISION_MODEL}")
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}")
        ACTIVE_TEXT_MODEL, ACTIVE_VISION_MODEL = None, None

    etl = ETL()
    etl.run()

    ai_layer = AILayer(ACTIVE_TEXT_MODEL, ACTIVE_VISION_MODEL)
    etl.events_df = ai_layer.process_vision(etl.events_df, etl.images_df)
    etl.events_df = ai_layer.process_messages(etl.events_df, etl.messages_df)

    output_rows = []
    total_reqs = len(etl.requests_df)
    logging.info(f"Processing {total_reqs} standard requests...")

    for _, req in etl.requests_df.iterrows():
        req_id = req["request_id"]
        req_date = pd.to_datetime(req["request_date"])
        req_amount = float(req["requested_amount"])

        user_id = req["user_id"]
        profile = etl.profiles_df[etl.profiles_df["user_id"] == user_id].iloc[0]
        user_events = etl.events_df[etl.events_df["user_id"] == user_id]

        simulator = Simulator(profile, user_events)
        simulator.generate_ledger(req_date)

        safe_today = simulator.get_max_safe_deduction()
        amount_safe_to_pay = min(req_amount, safe_today)

        evaluator = Evaluator(simulator, req, profile, etl.options_df)
        best_plan = evaluator.evaluate()

        earliest_full_date = None
        for i in range(90):
            d = req_date + pd.Timedelta(days=i)
            temp_ledger = simulator.ledger.copy()
            temp_ledger.loc[temp_ledger["date"] >= d, "daily_balance"] -= req_amount
            if (temp_ledger["daily_balance"] >= simulator.min_balance).all():
                earliest_full_date = d
                break

        if not best_plan:
            affordability_status = "not_affordable"
            recommended_payment_method = "not_recommended"
            payment_plan_str = "none"
            spending_changes_str = "none"
        else:
            recommended_payment_method = best_plan.method
            payment_plan_str = best_plan.format_plan()
            spending_changes_str = (
                "|".join(best_plan.spending_changes)
                if best_plan.spending_changes
                else "none"
            )

            if best_plan.method == "full_payment":
                affordability_status = (
                    "affordable_now"
                    if amount_safe_to_pay == req_amount
                    else "affordable_with_plan"
                )
            elif best_plan.method == "wait":
                affordability_status = "affordable_later"
            else:
                affordability_status = "affordable_with_plan"

        earliest_date_str = (
            earliest_full_date.strftime("%Y-%m-%d")
            if pd.notna(earliest_full_date)
            else ""
        )
        if affordability_status == "affordable_now":
            earliest_date_str = req_date.strftime("%Y-%m-%d")

        explanation = generate_explanation(
            req["request_text"],
            recommended_payment_method,
            amount_safe_to_pay,
            earliest_date_str,
            spending_changes_str,
            req_date.strftime("%Y-%m-%d"),
            ACTIVE_TEXT_MODEL,
        )

        amount_safe_str = (
            f"{amount_safe_to_pay:.2f}".rstrip("0").rstrip(".")
            if "." in f"{amount_safe_to_pay:.2f}"
            else f"{amount_safe_to_pay:.0f}"
        )

        output_rows.append(
            {
                "request_id": req_id,
                "amount_safe_to_pay": amount_safe_str,
                "affordability_status": affordability_status,
                "recommended_payment_method": recommended_payment_method,
                "payment_plan": payment_plan_str,
                "earliest_date_for_full_payment": earliest_date_str,
                "spending_changes_needed": spending_changes_str,
                "decision_explanation": explanation,
            }
        )

    out_df = pd.DataFrame(output_rows)
    # Ensure exact column match requirement
    out_df = out_df[OUTPUT_COLUMNS]
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)

    tracker.write_report(USAGE_REPORT_PATH, total_reqs)
    logging.info("Run successfully completed. Data successfully exported.")


if __name__ == "__main__":
    main()
