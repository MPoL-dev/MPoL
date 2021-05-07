import argparse

parser = argparse.ArgumentParser(
    description="Save baselines and klambda conversions to CSV."
)
parser.add_argument("outfile", help="Destination to save CSV table.")
args = parser.parse_args()

import csv

import numpy as np

from mpol.constants import c_ms

header = ["baseline", "100 GHz (Band 3)", "230 GHz (Band 6)", "340 GHz (Band 7)"]

baselines = np.array([10, 50, 100, 500, 1000, 5000, 10000, 16000])
frequencies = np.array([100, 230, 340]) * 1e9  # Hz


def format_baseline(baseline_m):
    if baseline_m < 1e3:
        return "{:.0f} m".format(baseline_m)
    elif baseline_m < 1e6:
        return "{:.0f} km".format(baseline_m * 1e-3)


def format_lambda(lam):
    if lam < 1e3:
        return "{:.0f}".format(lam) + " :math:`\lambda`"
    elif lam < 1e6:
        return "{:.0f}".format(lam * 1e-3) + " :math:`\mathrm{k}\lambda`"
    else:
        return "{:.0f}".format(lam * 1e-6) + " :math:`\mathrm{M}\lambda`"


data = []
for baseline in baselines:
    row = [format_baseline(baseline)]
    for frequency in frequencies:
        lam = baseline / (c_ms / frequency)
        row.append(format_lambda(lam))
    data.append(row)

with open(args.outfile, "w", newline="") as f:
    mywriter = csv.writer(f)
    mywriter.writerow(header)
    mywriter.writerows(data)
