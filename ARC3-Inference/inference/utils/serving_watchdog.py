"""Archimedes-3 Serving Watchdog and Concurrency Headroom Governor (§14.2).

Guards Qwen3.8-Flash-Next on SGLang/vLLM against:
1. SGLang Issue #36537 (infinite token-0 loops)
2. In-flight speculative acceptance collapse under scheduler saturation (spec_accept_length < 1.15)
3. Concurrency Headroom Governor: limits active concurrent streams M_active <= max_running_requests - 2
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


class ConcurrencyHeadroomGovernor:
    """Enforces M_active <= max_running_requests - 2 to prevent MTP speculative wedging."""

    @staticmethod
    def compute_safe_concurrency(configured_concurrency: int, max_running_requests: int = 8) -> int:
        """Caps concurrent streams to maintain at least 2 slots of scheduler headroom (§14.2)."""
        safe_ceiling = max(1, max_running_requests - 2)
        return min(max(1, configured_concurrency), safe_ceiling)


class ServingStabilityWatchdog:
    """Monitors serving telemetry (/metrics) and detects speculative collapse or hung streams."""

    def __init__(
        self,
        metrics_url: str = "http://127.0.0.1:1234/metrics",
        min_spec_accept_length: float = 1.15,
        consecutive_threshold: int = 20,
    ) -> None:
        self.metrics_url = metrics_url
        self.min_spec_accept_length = min_spec_accept_length
        self.consecutive_threshold = consecutive_threshold
        self.consecutive_low_spec_count: int = 0
        self.last_check_time: float = 0.0

    def parse_spec_accept_length(self, metrics_text: str) -> float | None:
        """Parses sglang:spec_accept_length or vllm:spec_accept_length from Prometheus metrics format."""
        for line in metrics_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "spec_accept_length" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[-1])
                    except ValueError:
                        pass
        return None

    def record_metrics_sample(self, spec_accept_length: float | None) -> bool:
        """Records a sample of spec_accept_length.
        
        Returns:
            True if worker soft-reset is triggered due to persistent speculative collapse.
        """
        if spec_accept_length is None:
            return False

        if spec_accept_length < self.min_spec_accept_length:
            self.consecutive_low_spec_count += 1
            if self.consecutive_low_spec_count >= self.consecutive_threshold:
                log.warning(
                    "ServingStabilityWatchdog: spec_accept_length=%.3f collapsed below %.2f across %d batches. "
                    "Triggering worker soft-reset to recover speculative acceptance.",
                    spec_accept_length,
                    self.min_spec_accept_length,
                    self.consecutive_threshold,
                )
                self.consecutive_low_spec_count = 0
                return True
        else:
            self.consecutive_low_spec_count = max(0, self.consecutive_low_spec_count - 1)

        return False
