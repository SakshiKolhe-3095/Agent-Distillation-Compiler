"""
Shared W&B project configuration for the Agent Distillation Compiler team.
All training scripts (Faiza's primary run, Yeshita's ablation, any future
runs) should import init_wandb_run() from here to log to the same shared
project, so results are comparable side by side.
"""

import wandb

PROJECT_NAME = "agent-distillation-compiler"
ENTITY = None  # set to your team/org name here if using a W&B Team; None uses personal account


def init_wandb_run(run_name: str, config: dict, tags: list = None):
    """
    Initializes a W&B run under the shared project.

    run_name: descriptive name, e.g. "faiza-primary-rank32" or "yeshita-ablation-rank8"
    config: dict of hyperparameters to log (rank, batch size, grad accum, etc.)
    tags: optional list of tags, e.g. ["primary"], ["ablation"], ["rank32"]
    """
    return wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        name=run_name,
        config=config,
        tags=tags or [],
    )


if __name__ == "__main__":
    # smoke test: logs a few dummy metrics to confirm the shared project works
    run = init_wandb_run(
        run_name="sakshi-wandb-setup-smoketest",
        config={"rank": 0, "batch_size": 0, "note": "setup verification run"},
        tags=["setup-test"],
    )
    for step in range(5):
        wandb.log({"dummy_loss": 1.0 / (step + 1)})
    run.finish()
    print(f"Smoke test run logged to project '{PROJECT_NAME}' -- check wandb.ai to confirm.")