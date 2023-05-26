import Tempotron as Tp
import torch as th
import argparse

# Parameter setting

# tau and tau_s are calculated by the following formula:
# tau = 0.03 * T, tau_s = tau / 4
# where, T is the length of spike trains, 280 ms in the discrimination demo.
# Note that these two ratios (0.03, and 4) are recommended in "https://doi.org/10.1038/nn1643" but
# are not fixed, and can be adjusted according to the experiments.
tau = 8.4
tau_s = 2.1

parser = argparse.ArgumentParser("train")

parser.add_argument('--dendrites_num', type=int, default=25)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--batchsize', type=int, default=5)

# Optional: Any string in "Gaussian, jitter, adding&missing" or "off"
parser.add_argument('--noise_key', type=str, default="Gaussian, jitter, adding&missing")

parser.add_argument('--Gaussian_sigma', type=float, default=1e-04)
parser.add_argument('--jitter_sigma', type=float, default=1e-04)
parser.add_argument('--adding_missing_prob', type=float, default=1e-04)

args, _ = parser.parse_known_args()

# The learning rate begins with the upper limit of the given interval and is subsequently
# reduced by half of the current interval after every twenty epochs.
learning_rate = [1e-6, 1e-3]

# Initialize efficacies

# th.manual_seed(723)
# dtype=th.float64 is required if you choose to use threshold=1. Lower precision requires higher threshold.
# dtype=th.float32 can properly function with threshold over 10.
efficacies = th.rand(args.dendrites_num, dtype=th.float64) - 0.5

# Initialize the model
tempotron = Tp.Tempotron(0, tau, tau_s, efficacies, A=1, dendrites_num=args.dendrites_num, echo=1, threshold=1)

# Start training
tempotron.train(epoch=args.epoch, batchsize=args.batchsize, learning_rate=learning_rate, momentum="on",
                noise_key=args.noise_key, Gaussian_sigma=args.Gaussian_sigma, jitter_sigma=args.jitter_sigma,
                adding_missing_prob=args.adding_missing_prob)