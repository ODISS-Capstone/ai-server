"""TurboQuant runtime integration for ai-server.

Ensures every HuggingFace Transformers model loaded inside the server is
wrapped with the TurboQuant compressed KV cache by default, unless the
operator explicitly disables it via settings.

The integration is deliberately defensive: if the TurboQuantWrapper
package or a CUDA device is missing the helpers just log and no-op so
CPU-only CI jobs still pass.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


def _sync_env_from_settings() -> None:
    """Propagate Settings flags to env vars consumed by turboquant.runtime."""
    os.environ.setdefault(
        "TURBOQUANT_AUTO_WRAP",
        "1" if settings.turboquant_auto_wrap else "0",
    )
    os.environ.setdefault(
        "TURBOQUANT_KEY_BITS",
        str(int(settings.turboquant_key_bits)),
    )
    os.environ.setdefault(
        "TURBOQUANT_VALUE_BITS",
        str(int(settings.turboquant_value_bits)),
    )
    os.environ.setdefault(
        "TURBOQUANT_COMPRESS_VALUES",
        "1" if settings.turboquant_compress_values else "0",
    )
    os.environ.setdefault(
        "TURBOQUANT_REQUIRE_CUDA",
        "1" if settings.turboquant_require_cuda else "0",
    )


def install() -> bool:
    """Activate TurboQuant auto-wrap for Transformers in this process."""
    if not settings.turboquant_auto_wrap:
        logger.info("TurboQuant auto-wrap disabled via settings")
        return False

    _sync_env_from_settings()

    try:
        from turboquant.runtime import install_hf_autowrap
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "TurboQuantWrapper not importable, skipping auto-wrap: %s", exc
        )
        return False

    installed = install_hf_autowrap(force=True)
    if installed:
        logger.info(
            "TurboQuant auto-wrap active (key_bits=%d, compress_values=%s)",
            settings.turboquant_key_bits,
            settings.turboquant_compress_values,
        )
    else:
        logger.warning("TurboQuant auto-wrap requested but hook install failed")
    return installed


def wrap(model: Any) -> Any:
    """Explicit fallback entry point for code paths that bypass Auto*."""
    if not settings.turboquant_auto_wrap:
        return model
    try:
        from turboquant.runtime import auto_wrap
    except Exception as exc:  # noqa: BLE001
        logger.debug("TurboQuantWrapper unavailable: %s", exc)
        return model
    return auto_wrap(model)


__all__ = ["install", "wrap"]
