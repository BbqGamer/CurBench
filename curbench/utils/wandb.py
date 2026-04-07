import os
from typing import Any, Optional


_WANDB_RUN = None


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def wandb_enabled() -> bool:
    return _env_flag("CURBENCH_WANDB")


def wandb_config() -> dict[str, Any]:
    config = {}
    for key, env_name in (
        ("project", "CURBENCH_WANDB_PROJECT"),
        ("entity", "CURBENCH_WANDB_ENTITY"),
        ("mode", "CURBENCH_WANDB_MODE"),
        ("name", "CURBENCH_WANDB_NAME"),
        ("group", "CURBENCH_WANDB_GROUP"),
    ):
        value = os.environ.get(env_name)
        if value:
            config[key] = value
    return config


def init_wandb(*, config: dict[str, Any], **kwargs: Any):
    global _WANDB_RUN
    if not wandb_enabled():
        return None
    if _WANDB_RUN is not None:
        return _WANDB_RUN

    try:
        import wandb
    except Exception:
        return None

    init_kwargs = wandb_config()
    init_kwargs.update(kwargs)
    init_kwargs.setdefault("project", "CurBench")
    init_kwargs.setdefault("mode", "online")
    _WANDB_RUN = wandb.init(config=config, **init_kwargs)
    return _WANDB_RUN


def log_wandb(metrics: dict[str, Any], step: Optional[int] = None):
    if not wandb_enabled() or _WANDB_RUN is None:
        return
    _WANDB_RUN.log(metrics, step=step)


def finish_wandb():
    global _WANDB_RUN
    if not wandb_enabled() or _WANDB_RUN is None:
        return
    try:
        _WANDB_RUN.finish()
    finally:
        _WANDB_RUN = None
