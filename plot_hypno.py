"""Plot a hypnogram for a single subject's overnight."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils


utils.set_matplotlib_style("technical")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--participant", type=int, required=True, choices=utils.participant_values()
)
args = parser.parse_args()



################################################################################
# SETUP
################################################################################

bids_root = Path(utils.config["bids_root"])

participant = args.participant
participant_id = f"sub-{participant:03d}"
task_id = "task-sleep"

import_path_hypno = bids_root / "derivatives" / participant_id / f"{participant_id}_{task_id}_hypno.tsv"
import_path_events = bids_root / participant_id / "eeg" / f"{participant_id}_{task_id}_events.tsv"
export_path = import_path_hypno.with_suffix(".png")

hypno = pd.read_csv(import_path_hypno, sep="\t")
events = pd.read_csv(import_path_events, sep="\t")

# Convert hypnogram stages to ints to ensure proper order.
stage_order = ["N3", "N2", "N1", "R", "W"]
stage_labels = ["SWS", "N2", "N1", "REM", "Wake"]
n_stages = len(stage_order)

hypno_int = hypno["description"].map(stage_order.index).to_numpy()
hypno_secs = hypno["duration"].mul(hypno["epoch"]).to_numpy()
hypno_hrs = hypno_secs / 60 / 60

hypno_rem = np.ma.masked_not_equal(hypno_int, stage_order.index("R"))

figsize = (5, 2)
fig, (ax0, ax1) = plt.subplots(
    nrows=2,
    figsize=figsize,
    sharex=True,
    sharey=False,
    gridspec_kw={"height_ratios": [2, 1]},
)

step_kwargs = dict(color="black", linewidth=0.5, linestyle="solid")

### Normal hypnogram
ax0.step(hypno_hrs, hypno_int, **step_kwargs)

# proba.plot(kind="area", color=palette, figsize=(10, 5), alpha=0.8, stacked=True, lw=0)

# cue_events = events.query("description.str.contains('Bell')")
cue_events = events.query("description.str.startswith('relaxcue')")
cue_secs = cue_events["onset"].to_numpy()
cue_hrs = cue_secs / 60 / 60

ax0.eventplot(
    positions=cue_hrs,
    orientation="horizontal",
    lineoffsets=n_stages - 0.5,
    linelengths=1,
    linewidths=0.1,
    colors="mediumpurple",
    linestyles="solid",
)

ax0.text(
    0, 1, "Relaxation cues", color="mediumpurple", ha="left", va="bottom", transform=ax0.transAxes
)


## Probabilities
probas = hypno[["proba_N1", "proba_N2", "proba_N3", "proba_R", "proba_W"]].T.to_numpy()
blues = utils.cmap2hex("blues", 4)[1:]
colors = blues + ["indianred", "gray"]
ax1.stackplot(hypno_hrs, probas, colors=colors, alpha=0.9)

ax0.set_yticks(range(n_stages))
ax0.set_yticklabels(stage_labels)
ax0.set_ylabel("Sleep Stage")
ax0.spines[["top", "right"]].set_visible(False)
ax0.tick_params(axis="both", direction="out", top=False, right=False)
ax0.set_ybound(upper=n_stages)
ax0.set_xbound(lower=0, upper=hypno_hrs.max())

ax1.set_ylabel("Sleep Stage\nProbability")
ax1.set_xlabel("Time (hours)")
ax1.tick_params(axis="both", which="both", direction="out", top=False, right=False)
ax1.set_ylim(0, 1)
ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
ax1.yaxis.set_minor_locator(plt.MultipleLocator(1/n_stages))
ax1.grid(which="minor")

# Legends. (need 2, one for the button press type and one for accuracy)
legend_labels = ["Awake", "REM", "N1", "N2", "N3"]
legend_colors = ["gray", "indianred"] + blues
handles = [ plt.matplotlib.patches.Patch(label=l, facecolor=c,
        edgecolor="black", linewidth=0.5)
    for l, c in zip(legend_labels, legend_colors) ]
legend = ax1.legend(handles=handles,
    loc="upper left", bbox_to_anchor=(1, 1),
    # handlelength=1, handleheight=.3,
    # handletextpad=,
    borderaxespad=0,
    labelspacing=.01,
    # columnspacing=,
    ncol=1, fontsize=6,
)

fig.align_ylabels()


# Export.
utils.export_mpl(export_path)
