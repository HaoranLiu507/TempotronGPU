import Tempotron as Tp
import torch as th
import time
from pathlib import Path

# Parameter setting

# tau and tau_s are calculated by the following formula:
# tau = 0.03 * T, tau_s = tau / 4
# where, T is the length of spike trains, 280 ms in the discrimination demo.
# Note that these two ratios (0.03, and 4) are recommended in "https://doi.org/10.1038/nn1643" but
# are not fixed, and can be adjusted according to the experiments.
tau = 8.4
tau_s = 2.1

# Load efficacies
EFFICACIES_PATH = Path("Efficacies")
EFFICACIES_PATH.mkdir(parents=True, exist_ok=True)
EFFICACIES_NAME = "Epoch299_efficacies.pt"
EFFICACIES_SAVE_PATH = EFFICACIES_PATH / EFFICACIES_NAME
efficacies = th.load(EFFICACIES_SAVE_PATH)

# Initialize Tempotron
# Make sure dendrites_num matches the efficacies you loaded
tempotron = Tp.Tempotron(0, tau, tau_s, efficacies, A=1, dendrites_num=25, echo=1, threshold=1)

# Start validation
start = time.time()
loss = tempotron.test_batch(batchsize=10)
end = time.time()

Correct_roit = (1 - loss)*100
print(f"Validation completed | Classification Accuracy: {Correct_roit:.4f} % | Time: {end - start:.2f} s ")