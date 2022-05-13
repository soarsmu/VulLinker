import math

import torch

from build_problem import MakeLabelVector, BuildProblem
from utils import training, validating_testing
import yaml
from torchtext import data
import shutil
from timeit import default_timer as timer
start = timer()
use_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model here
model_path = "models/initial_best_model.pkl"
model = torch.load("models/initial_best_model.pkl")

print("Model loading complete")

# Load Params
with open("params.yml") as f:
    params = yaml.safe_load(f)

common = params["common"]
hyper_params = params["hyper_params"]
normal_train = params["normal_train"]
params_search = params["params_search"]

# Show Common Params
term_size = shutil.get_terminal_size().columns
print("\n" + " Params ".center(term_size, "-"))
print([i for i in sorted(common.items())])
print("-" * shutil.get_terminal_size().columns)

# Set of automatically determined params
params = {"device": use_device, "params_search": False}

print("\n" + " Hyper Params ".center(term_size, "-"))
print([i for i in sorted(hyper_params.items())])
print("-" * shutil.get_terminal_size().columns)

params.update(common)
params.update(hyper_params)
params.update(normal_train)

# Build Problem
trainer = BuildProblem(params)
trainer.preprocess()
trainer.evaluate_model(MODEL_PATH = model)
end = timer()
print("Total Runtime: " + (end - start).__str__())