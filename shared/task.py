"""
Task abstraction: makes RADAR problem-agnostic.

A Task defines EVERYTHING about a specific research problem:
  - What files are mutable vs frozen
  - How to execute an experiment
  - What metrics to extract and how
  - What the Pareto objectives are (and their directions)
  - Domain-specific knowledge for the reflector
  - Constraints and anti-patterns

Built-in tasks:
  - ml_training: Minimize val_bpb on a training script (default)

Custom tasks: subclass TaskSpec or create one from a YAML/dict.
"""

from __future__ import annotations

import os
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Objective:
    """One axis of the Pareto front."""

    name: str
    pattern: str
    lower_is_better: bool = True
    weight: float = 1.0
    primary: bool = False
    default: float = float("inf")

    def extract(self, trace: str) -> Optional[float]:
        match = re.search(self.pattern, trace)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                return None
        return None

    def normalize_for_pareto(self, value: float) -> float:
        return value if self.lower_is_better else -value


@dataclass
class TaskSpec:
    """Complete specification of a research task."""

    name: str = "unnamed_task"
    description: str = "No description provided."
    target_file: str = "submission.py"
    frozen_files: list[str] = field(default_factory=lambda: ["prepare.py"])
    run_command: str = "python {target}"
    eval_command: str = ""
    time_budget: int = 300
    kill_timeout: int = 600
    objectives: list[Objective] = field(default_factory=list)
    domain_system_prompt: str = ""
    constraints: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    example_hypotheses: list[str] = field(default_factory=list)
    seed_file: str = ""
    runner_dir: str = ""
    docker_memory: str = "8Gi"
    docker_cpus: str = "2"

    @property
    def primary_objective(self) -> Optional[Objective]:
        for obj in self.objectives:
            if obj.primary:
                return obj
        return self.objectives[0] if self.objectives else None

    @property
    def primary_metric_pattern(self) -> str:
        primary = self.primary_objective
        return primary.pattern if primary else r"metric:\s*([\d.]+)"

    @property
    def primary_lower_is_better(self) -> bool:
        primary = self.primary_objective
        return primary.lower_is_better if primary else True

    def extract_all_objectives(self, trace: str) -> dict[str, float]:
        result = {}
        for obj in self.objectives:
            value = obj.extract(trace)
            result[obj.name] = value if value is not None else obj.default
        return result

    def objective_vector(self, objectives_dict: dict[str, float]) -> tuple[float, ...]:
        vec = []
        for obj in self.objectives:
            raw = objectives_dict.get(obj.name, obj.default)
            vec.append(obj.normalize_for_pareto(raw))
        return tuple(vec)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "target_file": self.target_file,
            "frozen_files": self.frozen_files,
            "run_command": self.run_command,
            "eval_command": self.eval_command,
            "time_budget": int(os.getenv("RADAR_TIME_BUDGET", str(self.time_budget))),
            "kill_timeout": self.kill_timeout,
            "objectives": [
                {
                    "name": o.name, "pattern": o.pattern,
                    "lower_is_better": o.lower_is_better, "weight": o.weight,
                    "primary": o.primary, "default": o.default,
                }
                for o in self.objectives
            ],
            "domain_system_prompt": self.domain_system_prompt,
            "constraints": self.constraints,
            "anti_patterns": self.anti_patterns,
            "example_hypotheses": self.example_hypotheses,
            "seed_file": self.seed_file,
            "runner_dir": self.runner_dir,
            "docker_memory": self.docker_memory,
            "docker_cpus": self.docker_cpus,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskSpec:
        objectives = [Objective(**o) for o in d.get("objectives", [])]
        return cls(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
            target_file=d.get("target_file", "submission.py"),
            frozen_files=d.get("frozen_files", []),
            run_command=d.get("run_command", "python {target}"),
            eval_command=d.get("eval_command", ""),
            time_budget=d.get("time_budget", 300),
            kill_timeout=d.get("kill_timeout", 600),
            objectives=objectives,
            domain_system_prompt=d.get("domain_system_prompt", ""),
            constraints=d.get("constraints", []),
            anti_patterns=d.get("anti_patterns", []),
            example_hypotheses=d.get("example_hypotheses", []),
            seed_file=d.get("seed_file", ""),
            runner_dir=d.get("runner_dir", ""),
            docker_memory=d.get("docker_memory", "8Gi"),
            docker_cpus=d.get("docker_cpus", "2"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> TaskSpec:
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        task = cls.from_dict(d)
        # Resolve seed_file relative to the YAML's directory
        if task.seed_file and not Path(task.seed_file).is_absolute():
            resolved = Path(path).parent / task.seed_file
            if resolved.exists():
                task.seed_file = str(resolved)
        return task

    def save_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def ml_training_task() -> TaskSpec:
    """Default: time-series forecasting with trusted execution."""
    return TaskSpec(
        name="ml_training",
        description=(
            "Train a time series foundation model to minimize CRPS on "
            "held-out forecasting tasks across diverse domains and frequencies."
        ),
        target_file="submission.py",
        frozen_files=["prepare.py", "evaluate.py", "harness.py", "auth.py"],
        run_command="python /workspace/frozen/harness.py {target}",
        eval_command="python /workspace/frozen/evaluate.py",
        time_budget=300,
        kill_timeout=600,
        objectives=[
            Objective(
                name="crps", pattern=r"crps:\s*([\d.]+)",
                lower_is_better=True, weight=1.0, primary=True,
            ),
            Objective(
                name="mase", pattern=r"mase:\s*([\d.]+)",
                lower_is_better=True, weight=0.5,
            ),
            Objective(
                name="exec_time", pattern=r"training_seconds:\s*([\d.]+)",
                lower_is_better=True, weight=0.2,
            ),
            Objective(
                name="memory_mb", pattern=r"peak_vram_mb:\s*([\d.]+)",
                lower_is_better=True, weight=0.1,
            ),
        ],
        domain_system_prompt=(
            "You are an expert autonomous ML researcher building time-series "
            "foundation models. The model receives multivariate context windows "
            "and must output probabilistic forecasts (quantile predictions). "
            "The code runs on a single GPU under a fixed time budget — throughput "
            "matters as much as per-step quality."
        ),
        constraints=[
            "Every experiment runs for a fixed time budget (wall clock)",
            "Code must be runnable on a single GPU",
            "Implement build_model() and build_optimizer() — the harness runs training",
            "Model input: (batch, context_len, num_variates) float tensor",
            "Model output: (batch, prediction_len, num_variates, num_quantiles) float tensor",
            "Preserve metric output format (runner needs to extract crps and mase)",
            "Code must be syntactically valid and crash-free",
        ],
        anti_patterns=[
            "Tweaking a single hyperparameter by a tiny amount",
            "Increasing model size without compensating for time budget",
            "Breaking the evaluation protocol or metric reporting",
            "Adding external dependencies not already available",
        ],
        example_hypotheses=[
            "Using patching (PatchTST-style) will improve efficiency and allow more steps",
            "Channel-independent processing reduces parameters while maintaining quality",
            "A reversible instance normalization layer will improve cross-domain generalization",
        ],
        seed_file="tasks/ts_forecasting/submission.py",
        runner_dir="runner/timeseries_forecast",
        docker_memory="8Gi",
        docker_cpus="2",
    )


def ts_forecasting_task() -> TaskSpec:
    """Alias for ml_training_task (time-series forecasting)."""
    return ml_training_task()


BUILT_IN_TASKS = {
    "ml_training": ml_training_task,
    "ts_forecasting": ts_forecasting_task,
}


def load_task(name_or_path: str) -> TaskSpec:
    """
    Load a task by name (built-in) or by file path (custom YAML).

    Usage:
        task = load_task("ml_training")
        task = load_task("tasks/ml_training/ml_training.yaml")
    """
    if name_or_path in BUILT_IN_TASKS:
        return BUILT_IN_TASKS[name_or_path]()

    path = Path(name_or_path)
    if path.exists() and path.suffix in (".yaml", ".yml"):
        return TaskSpec.from_yaml(str(path))

    if path.exists() and path.suffix == ".json":
        import json
        with open(path) as f:
            return TaskSpec.from_dict(json.load(f))

    raise ValueError(
        f"Unknown task: '{name_or_path}'. "
        f"Built-in tasks: {list(BUILT_IN_TASKS.keys())}. "
        f"Or provide a path to a .yaml/.json task definition."
    )
