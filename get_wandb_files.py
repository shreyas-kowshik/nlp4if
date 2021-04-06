import wandb
api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()