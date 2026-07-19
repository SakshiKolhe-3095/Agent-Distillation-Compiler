"""
Trajectory JSON schema and validator.
Ensures every trajectory record (from any teacher source) has the expected
fields with the expected types, before it enters the merged dataset or
downstream compression/training steps.
"""

from typing import Any


REQUIRED_FIELDS = {
    "problem": str,
    "plan": str,
    "code": str,
    "passed": bool,
    "retries": int,
}

OPTIONAL_FIELDS = {
    "source": str,
    "test": str,
    "error": str,
}


class TrajectoryValidationError(Exception):
    """Raised when a trajectory record fails schema validation."""
    pass


def validate_trajectory(task_id: str, trajectory: dict) -> list:
    """
    Validates a single trajectory record against the schema.
    Returns a list of error strings (empty list means valid).
    Does not raise — caller decides whether to treat errors as fatal.
    """
    errors = []

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in trajectory:
            errors.append(f"{task_id}: missing required field '{field}'")
            continue
        value = trajectory[field]
        if not isinstance(value, expected_type):
            errors.append(
                f"{task_id}: field '{field}' expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    for field, expected_type in OPTIONAL_FIELDS.items():
        if field in trajectory and not isinstance(trajectory[field], expected_type):
            errors.append(
                f"{task_id}: optional field '{field}' expected {expected_type.__name__}, "
                f"got {type(trajectory[field]).__name__}"
            )

    if "problem" in trajectory and isinstance(trajectory["problem"], str) and not trajectory["problem"].strip():
        errors.append(f"{task_id}: field 'problem' is empty")

    if "code" in trajectory and isinstance(trajectory["code"], str) and not trajectory["code"].strip():
        errors.append(f"{task_id}: field 'code' is empty")

    return errors


def validate_dataset(dataset: dict) -> dict:
    """
    Validates an entire {task_id: trajectory} dataset.
    Returns {"valid": [...task_ids...], "invalid": {task_id: [errors]}}.
    """
    valid = []
    invalid = {}

    for task_id, trajectory in dataset.items():
        errors = validate_trajectory(task_id, trajectory)
        if errors:
            invalid[task_id] = errors
        else:
            valid.append(task_id)

    return {"valid": valid, "invalid": invalid}


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "datasets/raw/trajectories_merged.json"
    with open(path) as f:
        dataset = json.load(f)

    result = validate_dataset(dataset)
    print(f"Valid: {len(result['valid'])}")
    print(f"Invalid: {len(result['invalid'])}")
    for task_id, errors in list(result["invalid"].items())[:10]:
        for e in errors:
            print(f"  {e}")