"""Strict input contract for the constrained formula DSL (no executable code)."""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator


MemoryOp = Literal["identity", "tanh", "softsign", "sin", "relu"]
InputOp = MemoryOp
CombineOp = Literal["add", "add_mul", "gated_input", "gated_mix", "max"]
FinalOp = Literal["tanh", "softsign", "clip", "tanh_plus_sin"]
MEMORY_OPS = get_args(MemoryOp)
INPUT_OPS = get_args(InputOp)
COMBINE_OPS = get_args(CombineOp)
FINAL_OPS = get_args(FinalOp)


class FormulaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    memory_op: MemoryOp
    input_op: InputOp
    combine_op: CombineOp
    final_op: FinalOp


class HypothesisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: StrictStr = ""
    name: StrictStr = ""
    source: StrictStr = ""
    rationale: StrictStr = ""
    formula: FormulaModel

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_flat_formula(cls, value):
        if isinstance(value, dict) and "formula" not in value:
            fields = set(FormulaModel.model_fields)
            if fields.intersection(value):
                value = dict(value)
                value["formula"] = {key: value.pop(key) for key in fields if key in value}
        return value
