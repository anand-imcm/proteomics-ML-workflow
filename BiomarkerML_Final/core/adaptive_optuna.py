from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import optuna
from optuna.trial import TrialState


MIN_VALID_COMPLETE_TRIALS = 50
MAX_TOTAL_ATTEMPTS = 200
PATIENCE_VALID_COMPLETE_TRIALS = 30

CONVERGED = "CONVERGED"
MAX_TRIALS_REACHED = "MAX_TRIALS_REACHED"
INSUFFICIENT_VALID_TRIALS = "INSUFFICIENT_VALID_TRIALS"


class AdaptiveTuningError(RuntimeError):
    pass


class InsufficientValidTrialsError(AdaptiveTuningError):
    pass


class LegacyTrialBudgetError(AdaptiveTuningError):
    pass


@dataclass(frozen=True)
class ImprovementTolerance:
    mode: str
    value: float
    scale_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"absolute", "relative"}:
            raise ValueError("Tolerance mode must be 'absolute' or 'relative'.")
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("Improvement tolerance must be finite and nonnegative.")
        if not math.isfinite(self.scale_floor) or self.scale_floor < 0:
            raise ValueError("Tolerance scale floor must be finite and nonnegative.")

    def threshold(self, reference: float) -> float:
        if self.mode == "absolute":
            return self.value
        return self.value * max(abs(reference), self.scale_floor)


_OBJECTIVE_TOLERANCE_CONTRACT: Mapping[str, tuple[str, float]] = {
    "weighted_f1": ("absolute", 1e-4),
    "vae_classification_f1": ("absolute", 1e-4),
    "mlpvae_classification_f1": ("absolute", 1e-4),
    "neg_root_mean_squared_error": ("relative", 1e-4),
    "vae_regression_mse": ("relative", 1e-4),
    "mlpvae_regression_rmse": ("relative", 1e-4),
    "umap_embedding_variance": ("relative", 1e-4),
    "tsne_embedding_variance": ("relative", 1e-4),
    "kernel_pca_embedding_variance": ("relative", 1e-4),
    "elastic_net_neg_mean_squared_error": ("relative", 1e-4),
}


def tolerance_for_objective(objective_name: str, *, scale_floor: float = 1.0) -> ImprovementTolerance:
    try:
        mode, value = _OBJECTIVE_TOLERANCE_CONTRACT[objective_name]
    except KeyError as exc:
        raise ValueError(f"No adaptive-tuning tolerance is defined for objective {objective_name!r}.") from exc
    return ImprovementTolerance(mode, value, 0.0 if mode == "absolute" else float(scale_floor))


def reject_legacy_n_trials(n_trials: int | None) -> None:
    if n_trials is not None:
        raise LegacyTrialBudgetError(
            "Fixed n_trials budgets are no longer supported. BiomarkerML uses 50 valid finite COMPLETE "
            "trials, 30 valid-trial patience, and a 200-attempt cap."
        )


class AdaptiveStudyController:
    def __init__(
        self,
        *,
        task: str,
        model: str,
        study_scope: str,
        objective_name: str,
        direction: str,
        tolerance: ImprovementTolerance,
        seed: int,
        study_id: str,
        outer_fold: int | None = None,
    ) -> None:
        if direction not in {"maximize", "minimize"}:
            raise ValueError("Direction must be 'maximize' or 'minimize'.")
        self.task = task
        self.model = model
        self.study_scope = study_scope
        self.objective_name = objective_name
        self.direction = direction
        self.tolerance = tolerance
        self.seed = int(seed)
        self.study_id = study_id
        self.outer_fold = outer_fold
        self.attempted_trials = 0
        self.valid_complete_trials = 0
        self.pruned_trials = 0
        self.failed_trials = 0
        self.nonfinite_complete_trials = 0
        self.best_objective: float | None = None
        self.best_trial_number: int | None = None
        self.material_reference: float | None = None
        self.consecutive_without_material_improvement = 0
        self.stop_reason: str | None = None
        self._seen_trial_numbers: set[int] = set()

    def _improvement(self, candidate: float, reference: float) -> float:
        return candidate - reference if self.direction == "maximize" else reference - candidate

    def _is_true_best(self, candidate: float, reference: float) -> bool:
        return candidate > reference if self.direction == "maximize" else candidate < reference

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.stop_reason is not None:
            return
        if trial.number in self._seen_trial_numbers:
            return
        self._seen_trial_numbers.add(trial.number)
        self.attempted_trials += 1

        if trial.state == TrialState.PRUNED:
            self.pruned_trials += 1
        elif trial.state == TrialState.FAIL:
            self.failed_trials += 1
        elif trial.state == TrialState.COMPLETE:
            value = trial.value
            if value is None or not math.isfinite(float(value)):
                self.nonfinite_complete_trials += 1
            else:
                candidate = float(value)
                self.valid_complete_trials += 1
                if self.best_objective is None or self._is_true_best(candidate, self.best_objective):
                    self.best_objective = candidate
                    self.best_trial_number = int(trial.number)
                if self.material_reference is None:
                    self.material_reference = candidate
                    self.consecutive_without_material_improvement = 0
                elif self._improvement(candidate, self.material_reference) > self.tolerance.threshold(
                    self.material_reference
                ):
                    self.material_reference = candidate
                    self.consecutive_without_material_improvement = 0
                else:
                    self.consecutive_without_material_improvement += 1
        else:
            self.failed_trials += 1

        if (
            self.valid_complete_trials >= MIN_VALID_COMPLETE_TRIALS
            and self.consecutive_without_material_improvement >= PATIENCE_VALID_COMPLETE_TRIALS
        ):
            self.stop_reason = CONVERGED
            study.stop()
        elif self.attempted_trials >= MAX_TOTAL_ATTEMPTS:
            self.stop_reason = (
                INSUFFICIENT_VALID_TRIALS
                if self.valid_complete_trials < MIN_VALID_COMPLETE_TRIALS
                else MAX_TRIALS_REACHED
            )
            study.stop()

    def finalize(self, study: optuna.Study) -> dict[str, Any]:
        if self.stop_reason is None:
            if self.attempted_trials >= MAX_TOTAL_ATTEMPTS:
                self.stop_reason = (
                    INSUFFICIENT_VALID_TRIALS
                    if self.valid_complete_trials < MIN_VALID_COMPLETE_TRIALS
                    else MAX_TRIALS_REACHED
                )
            elif (
                self.valid_complete_trials >= MIN_VALID_COMPLETE_TRIALS
                and self.consecutive_without_material_improvement >= PATIENCE_VALID_COMPLETE_TRIALS
            ):
                self.stop_reason = CONVERGED
            else:
                raise AdaptiveTuningError(
                    f"Study {self.study_id} ended before satisfying an adaptive stopping condition."
                )
        sampler = type(study.sampler).__name__
        pruner = type(study.pruner).__name__
        return {
            "StudyId": self.study_id,
            "StudyScope": self.study_scope,
            "OuterFold": self.outer_fold,
            "Task": self.task,
            "Model": self.model,
            "ObjectiveName": self.objective_name,
            "Direction": self.direction,
            "MinValidCompleteTrials": MIN_VALID_COMPLETE_TRIALS,
            "MaxTotalAttempts": MAX_TOTAL_ATTEMPTS,
            "PatienceValidCompleteTrials": PATIENCE_VALID_COMPLETE_TRIALS,
            "ImprovementTolerance": self.tolerance.value,
            "ToleranceType": self.tolerance.mode,
            "ToleranceScaleFloor": self.tolerance.scale_floor,
            "ImprovementToleranceContract": asdict(self.tolerance),
            "AttemptedTrials": self.attempted_trials,
            "ValidCompleteTrials": self.valid_complete_trials,
            "PrunedTrials": self.pruned_trials,
            "FailedTrials": self.failed_trials,
            "NonfiniteCompleteTrials": self.nonfinite_complete_trials,
            "BestObjective": self.best_objective,
            "BestTrialNumber": self.best_trial_number,
            "MaterialReferenceObjective": self.material_reference,
            "ConsecutiveValidCompleteWithoutMaterialImprovement": (
                self.consecutive_without_material_improvement
            ),
            "StopReason": self.stop_reason,
            "Sampler": sampler,
            "Pruner": pruner,
            "Seed": self.seed,
            "StudyLevelNJobs": 1,
        }


def run_adaptive_study(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    *,
    task: str,
    model: str,
    study_scope: str,
    objective_name: str,
    direction: str,
    tolerance: ImprovementTolerance,
    seed: int,
    study_id: str,
    outer_fold: int | None = None,
    provenance_path: str | Path | None = None,
    catch: tuple[type[Exception], ...] = (),
    show_progress_bar: bool = False,
) -> dict[str, Any]:
    controller = AdaptiveStudyController(
        task=task,
        model=model,
        study_scope=study_scope,
        objective_name=objective_name,
        direction=direction,
        tolerance=tolerance,
        seed=seed,
        study_id=study_id,
        outer_fold=outer_fold,
    )
    study.optimize(
        objective,
        n_trials=MAX_TOTAL_ATTEMPTS,
        n_jobs=1,
        callbacks=[controller],
        catch=catch,
        show_progress_bar=show_progress_bar,
    )
    report = controller.finalize(study)
    study.set_user_attr("biomarkerml_adaptive_tuning", report)
    if provenance_path is not None:
        path = Path(provenance_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if report["StopReason"] == INSUFFICIENT_VALID_TRIALS:
        raise InsufficientValidTrialsError(
            f"{task}/{model}/{study_scope} ({study_id}) exhausted {report['AttemptedTrials']} attempts "
            f"with only {report['ValidCompleteTrials']} valid finite COMPLETE trials; "
            f"pruned={report['PrunedTrials']}, failed={report['FailedTrials']}, "
            f"nonfinite_complete={report['NonfiniteCompleteTrials']}."
        )
    return report
