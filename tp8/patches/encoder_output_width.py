# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder output widths for sizing EC transfers.

An encoder can emit features wider than the LM embedding (DeepStack packs
several layers on the last dim) by an amount only the model knows. Producers
size transfers before the first real item is encoded, so the engine measures
the width once at startup by encoding a dummy item per modality.
"""

from collections.abc import Callable, Mapping
import gc
import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_SM12 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def _reclaim_dummy_encode_uma(device: torch.device) -> None:
    """Release startup dummy encoder activations and pinned host cache before KV sizing."""
    if (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_current_stream_capturing")
        and torch.cuda.is_current_stream_capturing()
    ):
        return
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if hasattr(torch, "_C") and hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        pass
    if getattr(device, "type", "") == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    try:
        from vllm.models.inkling2.ops import host_pin as _hp_uma

        _hp_uma.reclaim_host_uma(empty_cuda=False, tag="encoder_output_width")
    except Exception:
        pass


@torch.inference_mode()
def _measure_on_worker(worker: Any) -> dict[str, int]:
    """Encode one dummy item per modality; runs on every worker.

    Returns {} on ranks that hold no encoder.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import MultiModalBudget
    from vllm.multimodal.utils import group_and_batch_mm_kwargs
    from vllm.utils.torch_utils import PIN_MEMORY

    runner = worker.model_runner
    if not getattr(runner, "supports_mm_inputs", False) or not get_pp_group().is_first_rank:
        return {}
    pin_override = os.environ.get(
        "INKLING_VISION_PIN_META", "0" if _SM12 else "1"
    ).strip().lower() in ("1", "true", "on", "yes")
    use_pin = bool(PIN_MEMORY) and getattr(runner.device, "type", "") == "cuda" and pin_override
    budget = MultiModalBudget(
        worker.vllm_config, MULTIMODAL_REGISTRY, enable_cache=False
    )
    model = runner.get_model()
    widths: dict[str, int] = {}
    try:
        mm_max = budget.mm_max_toks_per_item
        for modality in mm_max:
            if isinstance(mm_max, Mapping) and int(mm_max[modality]) <= 0:
                continue
            items = None
            batch = None
            mm_kwargs = None
            outputs = None
            try:
                items = budget.get_dummy_encoder_profile_inputs(modality, 1)
                if not items:
                    continue
                batch = next(
                    iter(
                        group_and_batch_mm_kwargs(
                            items, device=runner.device, pin_memory=use_pin
                        )
                    ),
                    None,
                )
                if batch is None:
                    continue
                _, _, mm_kwargs = batch
                outputs = model.embed_multimodal(**mm_kwargs)
                if isinstance(outputs, (list, tuple)):
                    if not outputs:
                        continue
                    first = outputs[0]
                else:
                    first = outputs
                if first is None or getattr(first, "ndim", 0) < 1:
                    raise ValueError(
                        f"Encoder returned invalid output for modality {modality!r}: "
                        f"{type(first).__name__}"
                    )
                widths[modality] = int(first.shape[-1])
            finally:
                del outputs, mm_kwargs, batch, items
    finally:
        del budget
        _reclaim_dummy_encode_uma(runner.device)
    return widths


def measure_encoder_output_widths(
    vllm_config: "VllmConfig", collective_rpc: Callable[..., list[Any]]
) -> None:
    """Record the encoder output widths on an EC producer's config.

    Call after the model is loaded and before the KV cache is allocated: the
    Scheduler's connector reads the widths, and once the KV cache holds the
    memory budget there is no room left for a dummy encode.

    Raises:
        ValueError: No worker could encode a dummy item, or workers disagree.

    """
    ec_config = vllm_config.ec_transfer_config
    if ec_config is None or not ec_config.is_ec_producer:
        return
    # sm121 patch: collective_rpc exists only on the driver (world rank 0);
    # followers would assert. The scheduler (the widths' only consumer)
    # lives on the driver, so followers skip the measurement entirely.
    import torch.distributed as _dist

    if _dist.is_initialized() and _dist.get_rank() != 0:
        return
    reports = [widths for widths in collective_rpc(_measure_on_worker) if widths]
    if not reports:
        raise ValueError(
            "EC producer could not measure the encoder output width: the model "
            "encoded no dummy multimodal input."
        )
    if any(widths != reports[0] for widths in reports):
        raise ValueError(f"Workers disagree on the encoder output widths: {reports}")
    logger.info("EC producer encoder output widths: %s", reports[0])
    ec_config.mm_encoder_output_widths = reports[0]


def get_encoder_output_width(vllm_config: "VllmConfig", modality: str) -> int:
    """Encoder output width for `modality`, as measured at producer startup.

    Raises:
        ValueError: No width was measured for `modality`, so a transfer
            cannot be sized.

    """
    ec_config = vllm_config.ec_transfer_config
    if ec_config is None:
        raise ValueError(
            f"Cannot get encoder output width for modality {modality!r}: "
            "ec_transfer_config is None."
        )
    widths = ec_config.mm_encoder_output_widths or {}
    if modality not in widths:
        raise ValueError(
            f"No encoder output width was measured for modality {modality!r}; "
            f"measured: {sorted(widths)}"
        )
    return widths[modality]
