import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    data_root: str = "data/waterbird_complete95_forest2water2"
    seg_root: str = "data/CUB_200_2011/segmentations"
    output_root: str = "outputs/pilot"
    device: str = "cuda"
    batch_size: int = 32
    eval_batch_size: int = 32
    accumulation_steps: int = 4
    num_workers: int = 4
    num_threads: int = 4
    epochs: int = 100
    teacher_epochs: int = 30
    seed: int = 0
    lr: float = 5e-4
    teacher_lr: float = 5e-5
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    drop_path: float = 0.1
    label_smoothing: float = 0.1
    kd_alpha: float = 0.5
    temperature: float = 1.0
    keep_tokens: int = 98
    foreground_threshold: float = 0.5
    probe_per_group: int = 50
    diagnostic_epochs: list = field(default_factory=lambda: [0, 5, 10, 20, 40, 60, 80, 100])
    student_init: str = "scratch"
    teacher_pretrained: bool = True
    amp: bool = True
    grad_clip: float = 1.0
    max_train_batches: int | None = None
    model_scale: str = "deit"

    @classmethod
    def load(cls, path=None, overrides=None):
        values = json.loads(Path(path).read_text()) if path else {}
        values.update({k: v for k, v in (overrides or {}).items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        cfg = cls(**values)
        for name in ("data_root", "seg_root", "output_root"):
            if getattr(cfg, name):
                setattr(cfg, name, str(Path(getattr(cfg, name)).expanduser()))
        for k in ("batch_size", "eval_batch_size", "accumulation_steps", "num_threads", "epochs", "teacher_epochs", "probe_per_group"):
            if getattr(cfg, k) < 1:
                raise ValueError(f"{k} must be positive")
        if cfg.num_workers < 0 or cfg.warmup_epochs < 0:
            raise ValueError("num_workers and warmup_epochs must be nonnegative")
        if not 1 <= cfg.keep_tokens <= 196:
            raise ValueError("keep_tokens must be in [1, 196]")
        if not 0 < cfg.foreground_threshold <= 1 or cfg.temperature <= 0:
            raise ValueError("foreground_threshold must be in (0,1]; temperature must be positive")
        if not 0 <= cfg.kd_alpha <= 1 or not 0 <= cfg.label_smoothing < 1:
            raise ValueError("Invalid KD alpha or label smoothing")
        if cfg.student_init not in ("scratch", "imagenet") or cfg.model_scale not in ("deit", "debug"):
            raise ValueError("Invalid initialization or model_scale")
        if cfg.max_train_batches is not None and cfg.max_train_batches < 1:
            raise ValueError("max_train_batches must be positive or null")
        if cfg.lr <= 0 or cfg.teacher_lr <= 0 or cfg.min_lr < 0:
            raise ValueError("Learning rates must be positive (min_lr may be zero)")
        if not 0 <= cfg.drop_path < 1 or cfg.weight_decay < 0 or cfg.grad_clip <= 0:
            raise ValueError("Invalid drop_path, weight_decay, or grad_clip")
        return cfg

    def to_dict(self):
        return asdict(self)


METHODS = ["ce", "full", "random", "student", "foreground_rescue_5",
           "foreground_rescue_10", "foreground_rescue_20", "random_rescue_10"]
GROUP_NAMES = ["landbird_land", "landbird_water", "waterbird_land", "waterbird_water"]
