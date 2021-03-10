import wandb

wandb.init("deit")

config = wandb.config
config.name = "baseline"

wandb.watch(model)
wandb.log({"loss": loss})
