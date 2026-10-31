"""Fail closed on invalid numbers or effectively identical weight populations."""

from functools import wraps
import json
from pathlib import Path
import sys

import numpy as np


DEFAULT_MIN_DIVERSITY = 1e-6


class GuardrailStop(Exception):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.reason = {"code": code, "message": message, **details}


def check_finite(values, *, stage, **context):
    for name, value in values.items():
        array = np.asarray(value)
        invalid = int(np.count_nonzero(~np.isfinite(array)))
        if invalid:
            raise GuardrailStop("nonfinite", f"NaN/Inf detected in {name}",
                                stage=stage, field=name, invalid_count=invalid, **context)


def check_population(genome, minimum, **context):
    fields = genome._asdict()
    check_finite(fields, **context)
    size = next(iter(fields.values())).shape[0]
    if size < 2:
        return None  # A singleton cannot provide evidence of population collapse.
    variance_sum = 0.0
    parameters = 0
    for value in fields.values():
        array = np.asarray(value, dtype=np.float64)
        variance = np.var(array, axis=0)
        variance_sum += float(np.sum(variance))
        parameters += variance.size
    diversity = float(np.sqrt(variance_sum / parameters))
    check_finite({"population_diversity": diversity}, **context)
    if diversity <= minimum:
        raise GuardrailStop("diversity_collapse", "Population weight diversity is below the safety threshold",
                            diversity=diversity, threshold=minimum, population_size=size, **context)
    return diversity


def add_guardrail_arguments(parser):
    parser.add_argument("--min-population-diversity", type=float, default=DEFAULT_MIN_DIVERSITY,
                        help="Stop if RMS weight standard deviation is at or below this positive threshold.")


def guarded_run(experiment):
    def decorate(run):
        @wraps(run)
        def checked(args):
            try:
                check_finite({key: value for key, value in vars(args).items() if isinstance(value, float)},
                             stage="configuration")
                if args.min_population_diversity <= 0:
                    raise GuardrailStop("invalid_configuration", "min_population_diversity must be positive",
                                        stage="configuration")
                return run(args)
            except GuardrailStop as error:
                payload = {"status": "stopped", "experiment": experiment, "seed": args.seed,
                           "stop_reason": error.reason}
                path = Path(args.output_json or "guardrail_stop.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                print(f"GUARDRAIL STOP: {error}; saved reason to {path}", file=sys.stderr, flush=True)
                raise SystemExit(2) from error
        return checked
    return decorate
