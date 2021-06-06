import torch
from ray import tune
from common_data import k_fold_datasets, model
from common_functions import cross_validate

# query to see if we have a GPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# update to your current directory
MODEL_PATH = "/gpfs/group/ipc5094/default/RML/data/DSHARP/HD143006/rml/model.pt"

# wrap the model + training loop into a 'trainable' function
def trainable(config):
    cross_validate(model, config, device, k_fold_datasets, MODEL_PATH)


analysis = tune.run(
    trainable,
    config={
        "lr": 0.3,
        "lambda_sparsity": tune.loguniform(1e-8, 1e-4),
        "lambda_TV": tune.loguniform(1e-4, 1e1),
        "entropy": tune.loguniform(1e-7, 1e-1),
        "prior_intensity": tune.loguniform(1e-8, 1e-4),
        "epochs": 1000,
    },
    num_samples=24,
    local_dir="./results",
    resources_per_trial={"gpu": 1, "cpu": 1},
    verbose=2,
)

print("Best config: ", analysis.get_best_config(metric="cv_score", mode="min"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
print(df)
