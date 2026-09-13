"""AI Extraction layer utilizing NVIDIA NIM endpoints."""

import os
import json
import base64
import time
import logging
import pandas as pd
from typing import Dict, Any

from config import DATA_DIR, MEDIA_DIR, NVIDIA_API_KEY, NVIDIA_BASE_URL


class AILayer:
    """Manages AI-driven data extraction for unstructured dataset items."""

    def __init__(self, text_model: str, vision_model: str, data_dir: str = DATA_DIR):
        """Initializes the AI Layer with dynamic models.

        Args:
            text_model (str): The active text model ID.
            vision_model (str): The active vision model ID.
            data_dir (str): Path to the primary dataset directory.
        """
        self.text_model = text_model
        self.vision_model = vision_model
        self.data_dir = data_dir

    def _encode_image(self, image_path: str) -> str:
        """Encodes an image file to a base64 string.

        Args:
            image_path (str): The absolute path to the image file.

        Returns:
            str: Base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def process_vision(self, df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
        """Extracts numerical transaction amounts using the NIM Vision API.

        Args:
            df (pd.DataFrame): The events ledger dataframe.
            images_df (pd.DataFrame): The images lookup dataframe.

        Returns:
            pd.DataFrame: The mutated events ledger dataframe.
        """
        if not self.vision_model:
            logging.error("No active vision model provided. Skipping Vision AI layer.")
            return df

        try:
            from openai import OpenAI

            client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        except ImportError:
            logging.error("OpenAI package not found. Skipping Vision AI layer.")
            return df

        for idx, row in df.iterrows():
            if pd.isna(row["amount"]):
                rel_id = row.get("related_event_id")
                if pd.notna(rel_id):
                    img_match = images_df[images_df["related_event_id"] == rel_id]
                    if not img_match.empty:
                        img_id = img_match.iloc[0]["image_id"]
                        img_path = os.path.join(MEDIA_DIR, f"{img_id}.png")

                        if os.path.exists(img_path):
                            base64_img = self._encode_image(img_path)
                            try:
                                messages = [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": 'Extract the numeric transaction amount. Return strictly as JSON: {"amount": float}.',
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/png;base64,{base64_img}"
                                                },
                                            },
                                        ],
                                    }
                                ]
                                res = client.chat.completions.create(
                                    model=self.vision_model,
                                    messages=messages,
                                    temperature=0.0,
                                )
                                content = res.choices[0].message.content
                                if "```json" in content:
                                    content = (
                                        content.split("```json")[1]
                                        .split("```")[0]
                                        .strip()
                                    )
                                elif "```" in content:
                                    content = content.split("```")[1].strip()
                                data = json.loads(content)
                                df.at[idx, "amount"] = float(data.get("amount", 0.0))
                            except Exception as e:
                                logging.warning(f"Vision error on {img_id}: {e}")
        return df

    def process_messages(
        self, df: pd.DataFrame, messages_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extracts ledger mutations from textual messages using NIM Instruct API.

        Args:
            df (pd.DataFrame): The events ledger dataframe.
            messages_df (pd.DataFrame): The messages lookup dataframe.

        Returns:
            pd.DataFrame: The mutated events ledger dataframe.
        """
        if not self.text_model:
            logging.error("No active text model provided. Skipping NLP AI layer.")
            return df

        try:
            from openai import OpenAI

            client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        except ImportError:
            logging.error("OpenAI package not found. Skipping NLP AI layer.")
            return df

        for _, msg_row in messages_df.iterrows():
            msg_text = msg_row.get("message_text")
            if pd.notna(msg_text):
                try:
                    prompt = (
                        'Extract ledger mutations. Output JSON: {"action": "cancel|modify", "target_event_id": "string", "new_amount": float_or_null}. '
                        "Ignore all conversational filler.\n\n"
                        f"Message:\n{msg_text}"
                    )
                    messages = [{"role": "user", "content": prompt}]
                    res = client.chat.completions.create(
                        model=self.text_model, messages=messages, temperature=0.0
                    )

                    content = res.choices[0].message.content
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].strip()
                    data = json.loads(content)

                    action = data.get("action")
                    evt_id = data.get("target_event_id")

                    if not evt_id:
                        evt_id = msg_row.get("related_event_id")

                    if evt_id:
                        if action == "cancel":
                            df.loc[df["event_id"] == evt_id, "status"] = "cancelled"
                        elif action == "modify":
                            new_amt = data.get("new_amount")
                            if new_amt is not None:
                                df.loc[df["event_id"] == evt_id, "amount"] = float(
                                    new_amt
                                )

                except Exception as e:
                    logging.warning(f"NLP error on message intent mapping: {e}")

        # Clean up cancelled events from NLP modifications
        df = df[df["status"] != "cancelled"].copy()
        return df
