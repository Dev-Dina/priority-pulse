import re
from typing import Dict, List

import pandas as pd

from app.config import settings
from app.core.logger import data_logger


class DataCleaner:
    """Clean and filter Twitter customer support data"""

    def __init__(self, target_airlines: List[str] = None):
        self.target_airlines = target_airlines or [
            "americanair",
            "delta",
            "southwestair",
            "jetblue",
            "united",
            "usairways",
        ]
        # Populated by clean_pipeline — read by run_data_pipeline.py to save stats JSON
        self.stats: dict = {}

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw dataset"""
        data_logger.info(f"Loading raw data from {settings.RAW_DATASET_PATH}")

        try:
            df = pd.read_csv(settings.RAW_DATASET_PATH)
            data_logger.info(f"Loaded {len(df):,} total tweets")
            data_logger.info(f"Columns found: {df.columns.tolist()}")
            return df

        except FileNotFoundError:
            data_logger.error(f"File not found: {settings.RAW_DATASET_PATH}")
            raise

        except Exception as e:
            data_logger.error(f"Error loading data: {str(e)}")
            raise

    def extract_target_airline(self, text: str) -> str:
        if not isinstance(text, str):
            return None

        match = re.match(r"^@([A-Za-z0-9_]+)", text.strip())
        return match.group(1) if match else None

    def filter_airlines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter inbound tweets directed at target airlines"""

        data_logger.info(f"Filtering for airlines: {self.target_airlines}")

        # inbound tweets only
        df_inbound = df[df["inbound"] == True].copy()
        data_logger.info(f"Found {len(df_inbound):,} inbound tweets")
        self.stats["raw_inbound_count"] = len(df_inbound)

        # extract mention
        df_inbound["target_airline"] = df_inbound["text"].apply(
            self.extract_target_airline
        )

        # capture non-selected handles for the frontend "what else is in TWCS" display
        all_mention_counts = df_inbound["target_airline"].value_counts()
        normalized_targets_set = {a.lower() for a in self.target_airlines}
        self.stats["other_top_handles"] = [
            {"handle": str(h), "count": int(c)}
            for h, c in all_mention_counts.items()
            if pd.notna(h) and str(h).lower() not in normalized_targets_set and c >= 200
        ][:25]

        data_logger.info(
            f"Top extracted mentions:\n"
            f"{all_mention_counts.head(20)}"
        )

        # normalize for matching only
        normalized_targets = [a.lower() for a in self.target_airlines]

        df_filtered = df_inbound[
            df_inbound["target_airline"].str.lower().isin(normalized_targets)
        ].copy()

        data_logger.info(f"Filtered to {len(df_filtered):,} airline tweets")
        self.stats["after_airline_filter"] = len(df_filtered)

        airline_counts = df_filtered["target_airline"].value_counts()
        data_logger.info(f"Distribution:\n{airline_counts}")

        return df_filtered

    def clean_text(self, text: str) -> str:
        """Clean customer tweet text for ML / embeddings."""
        if not isinstance(text, str):
            return ""

        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"^@[A-Za-z0-9_]+\s*", "", text)   # strip leading airline @mention
        text = text.replace("&amp;", "&")
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def clean_agent_text(self, text: str) -> str:
        """
        Clean agent (outbound) response text.
        Strips the leading @CustomerHandle that agents use to address the
        customer, removes URLs, and normalises whitespace. Preserves the
        substance of the resolution so RAG context is useful.
        """
        if not isinstance(text, str):
            return ""

        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"^@[A-Za-z0-9_]+\s*", "", text)   # strip leading @CustomerHandle
        text = text.replace("&amp;", "&")
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def build_agent_response_map(self, df_raw: pd.DataFrame) -> Dict[int, str]:
        """
        Build a mapping from customer tweet_id → cleaned agent response text.

        In TWCS, every agent tweet (inbound=False) carries an
        in_response_to_tweet_id that points to the customer tweet it answered.
        Multiple agent replies to the same ticket are joined with " | ".

        This map is built from the *full* raw dataset before any filtering so
        no agent reply is accidentally discarded.
        """
        df_agent = df_raw[df_raw["inbound"] != True].copy()
        data_logger.info(f"Found {len(df_agent):,} agent (outbound) tweets to map")

        df_agent["agent_text"] = df_agent["text"].apply(self.clean_agent_text)

        # Drop rows with no parent reference or trivially short responses
        df_agent = df_agent.dropna(subset=["in_response_to_tweet_id"])
        df_agent = df_agent[df_agent["agent_text"].str.len() >= 10]

        # Safe integer conversion — in_response_to_tweet_id is float64 due to NaN
        df_agent["parent_id"] = pd.to_numeric(
            df_agent["in_response_to_tweet_id"], errors="coerce"
        )
        df_agent = df_agent.dropna(subset=["parent_id"])
        df_agent["parent_id"] = df_agent["parent_id"].astype(int)

        # Aggregate multiple agent replies per customer tweet
        response_map: Dict[int, str] = (
            df_agent.groupby("parent_id")["agent_text"]
            .apply(lambda texts: " | ".join(texts.tolist()))
            .to_dict()
        )

        data_logger.info(
            f"Agent response map built: {len(response_map):,} customer tweets have a matched response"
        )
        return response_map

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate tweets (deduplicate on cleaned customer text)"""
        initial_count = len(df)

        df = df.drop_duplicates(subset=["text"], keep="first")

        removed = initial_count - len(df)
        self.stats["removed_duplicates"] = int(removed)
        data_logger.info(f"Removed {removed:,} duplicate tweets")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing text"""
        initial_count = len(df)

        df = df.dropna(subset=["text"])

        removed = initial_count - len(df)
        if removed > 0:
            data_logger.info(f"Removed {removed:,} rows with missing text")

        return df

    def remove_low_quality_tweets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove useless tweets:
        - too short
        - only DM sent / thanks
        """
        initial_count = len(df)

        low_quality_patterns = [
            r"^dm sent$",
            r"^thanks$",
            r"^thank you$",
            r"^help$",
        ]

        pattern = "|".join(low_quality_patterns)

        df = df[
            (df["text"].str.len() >= 8)
            & (~df["text"].str.lower().str.match(pattern))
        ]

        removed = initial_count - len(df)
        self.stats["removed_low_quality"] = int(removed)
        data_logger.info(f"Removed {removed:,} low-quality tweets")

        return df

    def clean_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full cleaning pipeline"""
        data_logger.info("=" * 50)
        data_logger.info("STARTING DATA CLEANING PIPELINE")
        data_logger.info("=" * 50)

        self.stats["raw_total"] = len(df)

        # Build agent response map from the FULL raw dataset before any filtering
        # so no outbound reply is discarded before we can join it
        agent_map = self.build_agent_response_map(df)

        # Filter to inbound tweets for target airlines
        # (also populates self.stats["raw_inbound_count"], "after_airline_filter", "other_top_handles")
        df = self.filter_airlines(df)

        # Preserve original customer text before cleaning
        df["original_text"] = df["text"]

        data_logger.info("Cleaning customer text...")
        df["text"] = df["text"].apply(self.clean_text)

        # Remove empty results of cleaning
        before = len(df)
        df = df[df["text"].str.len() > 0]
        removed = before - len(df)
        self.stats["removed_empty"] = int(removed)
        if removed > 0:
            data_logger.info(f"Removed {removed:,} empty texts after cleaning")

        # also populates self.stats["removed_low_quality"] and "removed_duplicates"]
        df = self.remove_low_quality_tweets(df)
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)

        # Join agent responses using tweet_id as the foreign key
        data_logger.info("Joining agent responses to customer tickets...")
        tweet_id_numeric = pd.to_numeric(df["tweet_id"], errors="coerce")
        df["agent_response"] = tweet_id_numeric.map(agent_map).fillna("")

        has_response = (df["agent_response"].str.len() > 0).sum()
        self.stats["with_agent_response"] = int(has_response)
        self.stats["agent_response_pct"] = round(100 * has_response / len(df), 1)
        self.stats["final_count"] = len(df)
        data_logger.info(
            f"Tickets with an agent response: {has_response:,} "
            f"({100 * has_response / len(df):.1f}%)"
        )

        # Select final columns
        keep_columns = [
            "tweet_id",
            "author_id",
            "target_airline",
            "created_at",
            "inbound",
            "original_text",
            "text",
            "agent_response",
        ]

        keep_columns = [col for col in keep_columns if col in df.columns]
        df = df[keep_columns].copy()

        df.rename(
            columns={
                "author_id": "customer_id",
                "target_airline": "airline",
            },
            inplace=True,
        )

        data_logger.info(f"Cleaning complete. Final dataset: {len(df):,} tweets")

        return df

    def save_cleaned_data(self, df: pd.DataFrame):
        """Save cleaned dataset"""
        output_path = settings.CLEANED_DATASET_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        data_logger.info(f"Saved cleaned data to {output_path}")
        data_logger.info(f"Rows: {len(df):,}, Columns: {len(df.columns)}")