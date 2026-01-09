"""
Base Strategy Module
==================

Base classes for trading strategies.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    name: str
    description: str = ""
    max_weight_per_stock: float = 0.1  # Maximum weight per individual stock
    min_weight_per_stock: float = 0.0  # Minimum weight per individual stock
    max_sector_weight: float = 0.3      # Maximum weight per sector
    min_sector_weight: float = 0.0      # Minimum weight per sector


@dataclass
class StrategyResult:
    """Result from strategy execution."""
    strategy_name: str
    weights: pd.DataFrame
    metadata: Dict[str, Any]


class BaseStrategy:
    """Base class for trading strategies."""

    def __init__(self, config: StrategyConfig):
        """
        Initialize base strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def apply_risk_limits(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk limits to portfolio weights.

        Args:
            weights_df: DataFrame with columns ['gvkey', 'weight', ...]
                       May include 'sector' column for sector limits

        Returns:
            DataFrame with risk-adjusted weights that sum to 1.0
        """
        if weights_df.empty:
            return weights_df

        df = weights_df.copy()

        # Apply per-stock weight limits
        if 'weight' in df.columns:
            # Clip individual stock weights
            df['weight'] = df['weight'].clip(
                lower=self.config.min_weight_per_stock,
                upper=self.config.max_weight_per_stock
            )

        # Apply sector limits if sector column exists
        if 'sector' in df.columns and 'weight' in df.columns:
            sector_weights = df.groupby('sector')['weight'].sum()

            # Find sectors that exceed limits
            over_limit_sectors = sector_weights[sector_weights > self.config.max_sector_weight]

            for sector in over_limit_sectors.index:
                sector_mask = df['sector'] == sector
                sector_total = df.loc[sector_mask, 'weight'].sum()

                if sector_total > 0:
                    # Scale down weights in this sector proportionally
                    scale_factor = self.config.max_sector_weight / sector_total
                    df.loc[sector_mask, 'weight'] *= scale_factor

        # Re-normalize to ensure weights sum to 1.0
        total_weight = df['weight'].sum()
        if total_weight > 0:
            df['weight'] = df['weight'] / total_weight

        # Remove stocks with zero or very small weights
        df = df[df['weight'] > 1e-6].copy()

        # Final normalization
        total_weight = df['weight'].sum()
        if total_weight > 0:
            df['weight'] = df['weight'] / total_weight

        self.logger.info(f"Applied risk limits: {len(df)} positions, total weight: {df['weight'].sum():.6f}")

        return df

    def validate_weights(self, weights_df: pd.DataFrame) -> bool:
        """
        Validate that weights meet strategy constraints.

        Args:
            weights_df: DataFrame with weights

        Returns:
            True if weights are valid
        """
        if weights_df.empty:
            return True

        # Check weight sum
        total_weight = weights_df['weight'].sum()
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            self.logger.warning(f"Total weight {total_weight:.6f} does not sum to 1.0")
            return False

        # Check individual stock limits
        if 'weight' in weights_df.columns:
            max_weight = weights_df['weight'].max()
            min_weight = weights_df['weight'].min()

            if max_weight > self.config.max_weight_per_stock:
                self.logger.warning(f"Maximum weight {max_weight:.4f} exceeds limit {self.config.max_weight_per_stock}")
                return False

            if min_weight < self.config.min_weight_per_stock:
                self.logger.warning(f"Minimum weight {min_weight:.4f} below limit {self.config.min_weight_per_stock}")
                return False

        # Check sector limits
        if 'sector' in weights_df.columns and 'weight' in weights_df.columns:
            sector_weights = weights_df.groupby('sector')['weight'].sum()
            max_sector_weight = sector_weights.max()

            if max_sector_weight > self.config.max_sector_weight:
                self.logger.warning(f"Maximum sector weight {max_sector_weight:.4f} exceeds limit {self.config.max_sector_weight}")
                return False

        return True
