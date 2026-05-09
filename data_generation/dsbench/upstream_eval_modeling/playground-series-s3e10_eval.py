import argparse
import os.path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--answer_file", type=str, required=True)
parser.add_argument("--predict_file", type=str, required=True)

parser.add_argument("--value", type=str, default="Class")

args = parser.parse_args()

answers = pd.read_csv(args.answer_file)
predictions = pd.read_csv(args.predict_file)
answers.sort_values(by=["id"])
predictions.sort_values(by=["id"])
if "Strength" in predictions:
    performance = log_loss(answers[args.value], predictions["Strength"])
else:
    performance = log_loss(answers[args.value], predictions[args.value])


with open(os.path.join(args.path, args.name, "result.txt"), "w") as f:
    f.write(str(performance))
