import asyncio
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from src import crud
from src.config import ConfiguredModelSettings, ModelOverrideSettings, settings
from src.crud.representation import RepresentationManager
from src.dependencies import tracked_db
from src.llm import honcho_llm_call
from src.llm.types import LLMTelemetryContext
from src.models import Message
from src.schemas import ResolvedConfiguration
from src.telemetry import prometheus_metrics
from src.telemetry.events import RepresentationCompletedEvent, emit
from src.telemetry.events.llm import CallPurpose
from src.telemetry.logging import accumulate_metric, log_performance_metrics
from src.telemetry.prometheus.metrics import (
    DeriverComponents,
    DeriverTaskTypes,
    TokenTypes,
)
from src.telemetry.sentry import with_sentry_transaction
from src.utils.config_helpers import get_configuration
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.representation import PromptRepresentation, Representation
from src.utils.tokens import track_deriver_input_tokens

from .prompts import estimate_deriver_prompt_tokens, minimal_deriver_prompt

logger = logging.getLogger(__name__)


def _get_deriver_model_config() -> ConfiguredModelSettings:
    return settings.DERIVER.MODEL_CONFIG


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, raw, default)
        return default


def _env_json_dict(name: str) -> dict[str, Any]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return {}
    try:
        import json

        value = json.loads(raw)
    except Exception as exc:
        logger.warning("Invalid JSON for %s: %s", name, exc)
        return {}
    if not isinstance(value, dict):
        logger.warning("Invalid JSON for %s: expected object, got %s", name, type(value).__name__)
        return {}
    return value


def _build_fast_deriver_model_config(
    base_model_config: ConfiguredModelSettings,
) -> ConfiguredModelSettings | None:
    """Build optional fast deriver model config from env.

    This intentionally lives outside AppSettings so operators can canary a fast
    backend without changing Honcho's public config schema. Missing transport,
    API key env, token cap, etc. inherit from the safe primary config.
    """
    model = os.getenv("DERIVER_ROUTER_FAST_MODEL")
    base_url = os.getenv("DERIVER_ROUTER_FAST_BASE_URL")
    if not model or not base_url:
        return None

    return ConfiguredModelSettings(
        transport=os.getenv("DERIVER_ROUTER_FAST_TRANSPORT") or base_model_config.transport,
        model=model,
        temperature=base_model_config.temperature,
        top_p=base_model_config.top_p,
        top_k=base_model_config.top_k,
        frequency_penalty=base_model_config.frequency_penalty,
        presence_penalty=base_model_config.presence_penalty,
        seed=base_model_config.seed,
        thinking_effort=base_model_config.thinking_effort,
        thinking_budget_tokens=base_model_config.thinking_budget_tokens,
        max_output_tokens=_env_int(
            "DERIVER_ROUTER_FAST_MAX_OUTPUT_TOKENS",
            base_model_config.max_output_tokens or settings.LLM.DEFAULT_MAX_TOKENS,
        ),
        stop_sequences=base_model_config.stop_sequences,
        cache_policy=base_model_config.cache_policy,
        overrides=ModelOverrideSettings(
            api_key=os.getenv("DERIVER_ROUTER_FAST_API_KEY") or None,
            api_key_env=os.getenv("DERIVER_ROUTER_FAST_API_KEY_ENV")
            or base_model_config.overrides.api_key_env,
            base_url=base_url,
            provider_params=_env_json_dict("DERIVER_ROUTER_FAST_PROVIDER_PARAMS"),
        ),
    )


def _choose_deriver_model_config(
    *,
    base_model_config: ConfiguredModelSettings,
    messages: list[Message],
    queued_message_count: int,
    messages_tokens: int,
    prompt_message_tokens: int,
    hit_batch_token_cap: bool,
    had_previous_error: bool,
) -> tuple[ConfiguredModelSettings, str]:
    """Route deriver work by shape.

    Safe model remains default. Fast model is used only for small, fresh,
    first-attempt work units. Large, backfill/old, token-cap-hit, and retry/error
    work goes to the safe structured-output model.
    """
    if not _env_bool("DERIVER_ROUTER_ENABLED", False):
        return base_model_config, "safe:router-disabled"

    fast_model_config = _build_fast_deriver_model_config(base_model_config)
    if fast_model_config is None:
        return base_model_config, "safe:fast-config-missing"

    if had_previous_error:
        return base_model_config, "safe:retry-or-error"

    max_prompt_tokens = _env_int("DERIVER_ROUTER_FAST_MAX_PROMPT_MESSAGE_TOKENS", 2048)
    if prompt_message_tokens > max_prompt_tokens:
        return base_model_config, f"safe:prompt-message-tokens>{max_prompt_tokens}"

    max_messages = _env_int("DERIVER_ROUTER_FAST_MAX_QUEUED_MESSAGES", 1)
    if queued_message_count > max_messages:
        return base_model_config, f"safe:queued-messages>{max_messages}"

    max_tokens = _env_int("DERIVER_ROUTER_FAST_MAX_MESSAGE_TOKENS", 512)
    if messages_tokens > max_tokens:
        return base_model_config, f"safe:message-tokens>{max_tokens}"

    newest_allowed_age_hours = _env_int("DERIVER_ROUTER_FAST_MAX_AGE_HOURS", 24)
    if newest_allowed_age_hours > 0 and messages:
        latest_created_at = max(m.created_at for m in messages)
        if latest_created_at.tzinfo is None:
            latest_created_at = latest_created_at.replace(tzinfo=UTC)
        if latest_created_at < datetime.now(UTC) - timedelta(hours=newest_allowed_age_hours):
            return base_model_config, f"safe:older-than-{newest_allowed_age_hours}h"

    return fast_model_config, "fast:small-fresh-first-attempt"


@with_sentry_transaction("minimal_deriver_batch", op="deriver")
async def process_representation_tasks_batch(
    messages: list[Message],
    message_level_configuration: ResolvedConfiguration | None,
    *,
    observers: list[str],
    observed: str,
    queue_item_message_ids: list[int],
    hit_batch_token_cap: bool = False,
    was_flush_enabled: bool = False,
    batch_max_tokens: int = 0,
    had_previous_error: bool = False,
) -> None:
    """
    Process messages with minimal overhead - single LLM call, save to multiple collections.

    Args:
        messages: List of messages to process (includes interleaving context).
        message_level_configuration: Optional configuration override.
        observers: List of observer peer IDs (collections to save to).
        observed: The observed peer ID.
        queue_item_message_ids: Message IDs from queue items being processed
        hit_batch_token_cap: queue batcher clamped this batch to fit
        was_flush_enabled: DERIVER.FLUSH_ENABLED snapshot at batch time
        batch_max_tokens: DERIVER.REPRESENTATION_BATCH_MAX_TOKENS snapshot
        had_previous_error: True when any queue item in this batch previously errored.
    """
    if not messages:
        return

    overall_start = time.perf_counter()

    messages.sort(key=lambda x: x.id)
    latest_message = messages[-1]
    earliest_message = messages[0]

    # Get configuration if not provided
    # TODO: this appears to be a very rare edge case coming out of `get_queue_item_batch` in queue_manager.py,
    # possible that we can remove this and require configuration to come through with the payload.
    if message_level_configuration is None:
        async with tracked_db("minimal_deriver.get_config") as db:
            message_level_configuration = get_configuration(
                None,
                await crud.get_session(
                    db, latest_message.session_name, latest_message.workspace_name
                ),
                await crud.get_workspace(
                    db, workspace_name=latest_message.workspace_name
                ),
            )

    # Skip if disabled
    if message_level_configuration.reasoning.enabled is False:
        return

    custom_instructions = message_level_configuration.reasoning.custom_instructions

    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "starting_message_id",
        earliest_message.id,
        "id",
    )
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "ending_message_id",
        latest_message.id,
        "id",
    )

    # Format messages with timestamps
    formatted_messages = "\n".join(
        format_new_turn_with_timestamp(msg.content, msg.created_at, msg.peer_name)
        for msg in messages
    )

    # Track token usage - count only tokens from messages being processed
    prompt_tokens = estimate_deriver_prompt_tokens(custom_instructions)
    queue_item_message_ids_set = set(queue_item_message_ids)
    messages_tokens = sum(
        msg.token_count for msg in messages if msg.id in queue_item_message_ids_set
    )
    track_deriver_input_tokens(
        task_type=DeriverTaskTypes.INGESTION,
        components={
            DeriverComponents.PROMPT: prompt_tokens,
            DeriverComponents.MESSAGES: messages_tokens,
        },
    )

    # Build prompt
    prompt = minimal_deriver_prompt(
        peer_id=observed,
        messages=formatted_messages,
        custom_instructions=custom_instructions,
    )

    context_prep_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "context_preparation",
        context_prep_duration,
        "ms",
    )

    # validation on settings means max_tokens will always be > 0
    base_model_config = _get_deriver_model_config()
    model_config, router_decision = _choose_deriver_model_config(
        base_model_config=base_model_config,
        messages=messages,
        queued_message_count=len(queue_item_message_ids),
        messages_tokens=messages_tokens,
        prompt_message_tokens=sum(msg.token_count for msg in messages),
        hit_batch_token_cap=hit_batch_token_cap,
        had_previous_error=had_previous_error,
    )
    max_tokens = model_config.max_output_tokens or settings.LLM.DEFAULT_MAX_TOKENS
    logger.info(
        "Deriver router selected %s/%s at %s for messages %s:%s (%s, queued=%d, tokens=%d)",
        model_config.transport,
        model_config.model,
        model_config.overrides.base_url,
        earliest_message.id,
        latest_message.id,
        router_decision,
        len(queue_item_message_ids),
        messages_tokens,
    )
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "router_decision",
        router_decision,
        "label",
    )
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "router_model",
        model_config.model,
        "label",
    )

    # Single LLM call
    llm_start = time.perf_counter()
    llm_call = honcho_llm_call(
        model_config=model_config,
        prompt=prompt,
        max_tokens=max_tokens,
        track_name="Minimal Deriver",
        response_model=PromptRepresentation,
        json_mode=True,
        max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
        enable_retry=True,
        retry_attempts=3,
        trace_name="minimal_deriver",
        telemetry=LLMTelemetryContext(
            workspace_name=latest_message.workspace_name,
            call_purpose=CallPurpose.DERIVER_REPRESENTATION.value,
            parent_category="representation",
            observed=observed,
        ),
    )
    if router_decision.startswith("fast:"):
        response = await asyncio.wait_for(
            llm_call,
            timeout=_env_int("DERIVER_ROUTER_FAST_TIMEOUT_SECONDS", 30),
        )
    else:
        response = await llm_call
    llm_duration = (time.perf_counter() - llm_start) * 1000

    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "llm_call_duration",
        llm_duration,
        "ms",
    )

    # Prometheus metrics
    if settings.METRICS.ENABLED:
        prometheus_metrics.record_deriver_tokens(
            count=response.output_tokens,
            task_type=DeriverTaskTypes.INGESTION.value,
            token_type=TokenTypes.OUTPUT.value,
            component=DeriverComponents.OUTPUT_TOTAL.value,
        )

    message_ids = [m.id for m in messages if m.peer_name == observed]

    # Convert to Representation and save
    observations = Representation.from_prompt_representation(
        response.content,
        message_ids,
        latest_message.session_name,
        latest_message.created_at,
    )

    successful_observer_count = 0
    if observations.is_empty() or not message_ids:
        logger.warning(
            "Deriver generated zero observations for messages %s:%s in %s/%s!",
            earliest_message.id,
            latest_message.id,
            latest_message.workspace_name,
            latest_message.session_name,
        )
    else:
        # Save to all observer collections
        for observer in observers:
            representation_manager = RepresentationManager(
                workspace_name=latest_message.workspace_name,
                observer=observer,
                observed=observed,
            )

            try:
                await representation_manager.save_representation(
                    observations,
                    message_ids,
                    latest_message.session_name,
                    latest_message.created_at,
                    message_level_configuration,
                )
                successful_observer_count += 1
            except Exception as e:
                logger.error(
                    "Failed to save representation for observer %s: %s", observer, e
                )

    # Log metrics
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = len(observations.explicit) + len(observations.deductive)
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "observation_count",
        total_observations,
        "count",
    )

    if settings.DERIVER.LOG_OBSERVATIONS:
        # Log messages fed into deriver
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observed}",
            "messages",
            formatted_messages,
            "blob",
        )
        # Log actual observations created as blob metrics
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observed}",
            "explicit_observations",
            "\n".join(f" • {obs}" for obs in observations.explicit),
            "blob",
        )

    log_performance_metrics("minimal_deriver", f"{latest_message.id}_{observed}")

    # token-breakdown fields derived from messages + cap snapshots.
    queued_message_count = len(queue_item_message_ids)
    prompt_message_count = len(messages)
    prompt_message_tokens = sum(msg.token_count for msg in messages)
    extra_context_message_count = max(prompt_message_count - queued_message_count, 0)
    extra_context_tokens = max(prompt_message_tokens - messages_tokens, 0)

    # Data-quality invariants. Best-effort — telemetry never bleeds into the
    # deriver path — but log loudly when violated so analytics alerting catches
    # silent estimator failures (provider tokenization drift, scaffold helper
    # returning 0) at the source instead of as drift in BigQuery later.
    if response.input_tokens < messages_tokens:
        logger.warning(
            "token-breakdown invariant violated: response.input_tokens (%d) < messages_tokens (%d) for observed=%s, latest=%s — provider tokenization drift or wrong messages_tokens computation?",
            response.input_tokens,
            messages_tokens,
            observed,
            latest_message.public_id,
        )
    if prompt_tokens <= 0:
        logger.warning(
            "prompt_scaffold_tokens estimated as %d for observed=%s, latest=%s — estimate_deriver_prompt_tokens may have failed silently",
            prompt_tokens,
            observed,
            latest_message.public_id,
        )

    # Emit telemetry event
    emit(
        RepresentationCompletedEvent(
            workspace_name=latest_message.workspace_name,
            session_name=latest_message.session_name,
            observed=observed,
            queue_items_processed=len(queue_item_message_ids),
            earliest_message_id=earliest_message.public_id,
            latest_message_id=latest_message.public_id,
            message_count=len(messages),
            explicit_conclusion_count=len(observations.explicit),
            context_preparation_ms=context_prep_duration,
            llm_call_ms=llm_duration,
            total_duration_ms=overall_duration,
            input_tokens=messages_tokens,
            total_input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            # additive fields
            queued_message_count=queued_message_count,
            prompt_message_count=prompt_message_count,
            prompt_message_tokens=prompt_message_tokens,
            extra_context_message_count=extra_context_message_count,
            extra_context_tokens=extra_context_tokens,
            prompt_scaffold_tokens=prompt_tokens,
            batch_max_tokens=batch_max_tokens,
            max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
            was_flush_enabled=was_flush_enabled,
            hit_batch_token_cap=hit_batch_token_cap,
            hit_input_token_cap=response.hit_input_token_cap,
            observer_count=successful_observer_count,
        )
    )
