"""Plot all subject's hypnodensities (i.e., probabilistic hypnograms)."""
from pathlib import Path

from bids import BIDSLayout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils


utils.set_matplotlib_style("technical")


################################################################################
# SETUP
################################################################################

root_dir = Path(utils.config["bids_root"])

layout = BIDSLayout(root_dir, derivatives=True, validate=False)
hypno_bids_files = layout.get(task="sleep", suffix="hypno", extension="tsv")
events_bids_files = layout.get(task="sleep", suffix="events", extension="tsv")
assert len(hypno_bids_files) == len(events_bids_files)
n_subjects = len(hypno_bids_files)

export_path = root_dir / "derivatives" / "hypnodensities.png"

cue_palette = utils.config["cue_palette"]
participant_palette = utils.load_participant_palette()

participant_conditions = utils.load_participants_file()["tmr_condition"].to_dict()

# stage_order = ["N3", "N2", "N1", "R", "W"]
# stage_labels = ["SWS", "N2", "N1", "REM", "Wake"]
# n_stages = len(stage_order)

figsize = (5, n_subjects * 0.7)

fig, axes = plt.subplots(
    nrows=n_subjects,
    figsize=figsize,
    sharex=True,
    sharey=True,
    # gridspec_kw={"hspace": 0},
)

plot_order = ["N1", "N2", "N3", "R", "W"]
legend_order = ["W", "N1", "N2", "N3", "R"]
stage_labels = dict(N1="N1", N2="N2", N3="SWS", R="REM", W="Awake")
blues = utils.cmap2hex("blues", 4)
stage_palette = {
    "N1": blues[1],
    "N2": blues[2],
    "N3": blues[3],
    "R": "indianred",
    "W": "gray",
}

stage_colors = [stage_palette[s] for s in plot_order]
stage_columns = [f"proba_{s}" for s in plot_order]
alpha = 0.9
cue_height = 0.3
n_stages = len(plot_order)

for ax, h_bf, e_bf in zip(axes, hypno_bids_files, events_bids_files):
    hypno = h_bf.get_df()
    events = e_bf.get_df()

    participant_id = "sub-" + h_bf.entities["subject"]
    participant_color = participant_palette[participant_id]
    cue_condition = participant_conditions[participant_id]
    cue_color = cue_palette[cue_condition]

    # Extract time values.
    hypno_secs = hypno["duration"].mul(hypno["epoch"]).to_numpy()
    hypno_hrs = hypno_secs / 60 / 60

    # Extract stage probabilities.
    probas = hypno[stage_columns].T.to_numpy()
    ax.stackplot(hypno_hrs, probas, colors=stage_colors, alpha=alpha)

    ax.axvline(hypno_hrs.max(), color="black", linewidth=1)

    # Events/cues
    cue_secs = events.query("description.str.endswith('.wav')")["onset"].to_numpy()
    cue_hrs = cue_secs / 60 / 60
    ax.eventplot(
        positions=cue_hrs,
        orientation="horizontal",
        lineoffsets=1 + cue_height / 2,
        linelengths=cue_height,
        linewidths=0.1,
        colors=cue_color,
        linestyles="solid",
        clip_on=False,
    )

    # Aesthetics.
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1/n_stages))
    ax.grid(which="minor")
    ax.set_ylim(0, 1)
    ax.margins(x=0)
    if ax.get_subplotspec().is_last_row():
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_xlabel("Time (hours)")
        ax.tick_params(axis="both", which="both", direction="out", top=False, right=False)
        ax.set_ylabel("Probability")
    else:
        ax.tick_params(axis="both", which="both", direction="out", bottom=False, top=False, right=False)

    ax.text(1, 1, participant_id, color=participant_color, ha="right", va="bottom", transform=ax.transAxes)

legend_ax = axes[0]
legend_handles = [
    plt.matplotlib.patches.Patch(
        label=stage_labels[s], facecolor=stage_palette[s], edgecolor="black", linewidth=0.5
    ) for s in legend_order
]
legend = legend_ax.legend(
    handles=legend_handles,
    title="Sleep Stage",
    loc="lower center",
    bbox_to_anchor=(0.65, 1.5),
    # handlelength=1, handleheight=.3,
    # handletextpad=,
    borderaxespad=0,
    labelspacing=0.01,
    # columnspacing=,
    ncol=n_stages,
    # fontsize=6,
)

legend_ax.text(0.0, 1.9, "TMR Relaxation cues", color=cue_palette["relax"], ha="left", va="bottom", transform=legend_ax.transAxes)
legend_ax.text(0.0, 1.6, "TMR Story cues", color=cue_palette["story"], ha="left", va="bottom", transform=legend_ax.transAxes)


fig.align_ylabels()


# Export.
utils.export_mpl(export_path)
