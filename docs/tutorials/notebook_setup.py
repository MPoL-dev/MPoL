# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# flake8: noqa
"""isort:skip_file"""

# # Notebook Setup

# set some plotting styles, inspired by DFM
# https://github.com/exoplanet-dev/exoplanet/blob/main/docs/tutorials/notebook_setup.py

get_ipython().magic('config InlineBackend.figure_format = "retina"')

import matplotlib.pyplot as plt

plt.style.use("default")
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 12
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.frameon"] = False
