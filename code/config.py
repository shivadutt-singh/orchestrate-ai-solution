"""Configuration module for the orchestration system."""

import os
import logging

# Set up global logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# API Endpoints
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "dummy")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
TEXT_MODELS_TO_TRY = [
    "deepseek-ai/deepseek-v3",
    "deepseek-ai/deepseek-r1",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "nvidia/nemotron-4-340b-instruct"
]
VISION_MODELS_TO_TRY = [
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct"
]

# Paths
DATA_DIR = "dataset"
MEDIA_DIR = os.path.join(DATA_DIR, "media", "images")
OUTPUT_CSV_PATH = "output.csv"
USAGE_REPORT_PATH = os.path.join("code", "evaluation", "usage_report.md")

# Final Columns Order
OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]
