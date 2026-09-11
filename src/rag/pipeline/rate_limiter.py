"""Token-Bucket Rate Limiter with Jittered Exponential Backoff.

Protects embedding endpoints from 429 Rate Limits by controlling:
    - Requests Per Minute (RPM)
    - Tokens Per Minute (TPM)
"""

import asyncio
import time
import random

class TokenBucketRateLimiter:
    """Async token-bucket rate limiter for embedding API calls."""

    def __init__(self, requests_per_minute: int = 600, tokens_per_minute: int = 200_000):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        
        self._request_tokens = float(requests_per_minute)
        self._word_tokens = float(tokens_per_minute)
        self._last_refill = time.time()
        self._lock = asyncio.Lock()

    async def _refill(self):
        """Refills available tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._last_refill = now

        # Add proportional capacity
        self._request_tokens = min(float(self.rpm), self._request_tokens + elapsed * (self.rpm / 60.0))
        self._word_tokens = min(float(self.tpm), self._word_tokens + elapsed * (self.tpm / 60.0))

    async def acquire(self, estimated_tokens: int = 500):
        """Waits asynchronously until rate capacity is available."""
        while True:
            async with self._lock:
                await self._refill()
                if self._request_tokens >= 1.0 and self._word_tokens >= float(estimated_tokens):
                    self._request_tokens -= 1.0
                    self._word_tokens -= float(estimated_tokens)
                    return

            # Backpressure sleep with jitter
            await asyncio.sleep(0.05 + random.uniform(0.01, 0.04))

# Singleton instance
rate_limiter = TokenBucketRateLimiter()
