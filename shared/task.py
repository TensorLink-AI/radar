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
    """Complete specification of a research task.

    Three independent time budgets, each scoped to one phase:

      agent_seconds     Phase A — wall-clock for the miner's agent pod
                        (how long the agent has to think / design an
                        architecture). Falls back to Config.AGENT_TIMEOUT
                        when unset.

      time_budget       Phase B — wall-clock for the *training loop* inside
                        the trainer container (harness kills the loop after
                        this many seconds). Nothing to do with the agent.

      kill_timeout      Phase B — hard subprocess kill-signal timeout for
                        trainer run/eval commands (see env.py). Always >=
                        time_budget; this is the outer safety net.
    """

    name: str = "unnamed_task"
    description: str = "No description provided."
    target_file: str = "submission.py"
    frozen_files: list[str] = field(default_factory=lambda: ["prepare.py"])
    run_command: str = "python {target}"
    eval_command: str = ""
    agent_seconds: int = 0          # 0 = inherit Config.AGENT_TIMEOUT
    time_budget: int = 300          # trainer training-loop wall-clock
    kill_timeout: int = 600         # trainer subprocess hard-kill
    objectives: list[Objective] = field(default_factory=list)
    domain_system_prompt: str = ""
    constraints: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    example_hypotheses: list[str] = field(default_factory=list)
    seed_file: str = ""
    runner_dir: str = ""
    docker_memory: str = "8Gi"
    docker_cpus: str = "2"
    task_params: dict = field(default_factory=dict)

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
            "agent_seconds": self.agent_seconds,
            "time_budget": self.time_budget,
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
            "task_params": self.task_params,
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
            agent_seconds=d.get("agent_seconds", 0),
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
            task_params=d.get("task_params", {}),
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
        task_params={
            "context_len": 512,
            "prediction_len": 96,
            "num_variates": 1,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
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


def load_enabled_tasks(enabled_csv: str = "") -> dict[str, TaskSpec]:
    """Load all enabled tasks as a {name: TaskSpec} dict.

    Args:
        enabled_csv: Comma-separated task names. Empty or "all" loads all
                     built-in tasks.

    Returns:
        Dict mapping task name to TaskSpec, with at least one entry.
    """
    if not enabled_csv or enabled_csv.strip().lower() == "all":
        names = list(BUILT_IN_TASKS.keys())
    else:
        names = [n.strip() for n in enabled_csv.split(",") if n.strip()]

    tasks: dict[str, TaskSpec] = {}
    for name in names:
        tasks[name] = load_task(name)

    if not tasks:
        raise ValueError(f"No valid tasks in ENABLED_TASKS='{enabled_csv}'")

    return tasks
