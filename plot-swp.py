"""
Will definitely re-organize later.
Don't want final plot to have individual subjects,
and all the arguments are just to remind me of how many options there are.
"""
import argparse
from pathlib import Path

from bids import BIDSLayout
import mne
import numpy as np
import pandas as pd
import yasa

import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns

import utils


utils.set_matplotlib_style("technical")


parser = argparse.ArgumentParser()
parser.add_argument("--channel", default="AFz", type=str, choices=["AFz", "Fz", "Fp1", "Fp2", "F3", "F4", "Cz", "C3", "C4"])
parser.add_argument("--window", default=30, type=int)
parser.add_argument("--allstages", action="store_true", help="Include non-SWS stages")
parser.add_argument("--allcycles", action="store_true", help="Include non-cued/late SWS cycles")
parser.add_argument("--allparams", action="store_true", help="Look at all SWS parameters.")
args = parser.parse_args()

channel = args.channel
window_size = args.window
include_all_stages = args.allstages
include_all_cycles = args.allcycles
include_all_params = args.allparams

bids_root = Path(utils.config["bids_root"])

if include_all_params:
    export_path = bids_root / "derivatives" / f"swp_channel-{channel}_allparams.png"
else:
    export_path = bids_root / "derivatives" / f"swp_channel-{channel}.png"

layout = BIDSLayout(bids_root, validate=False, derivatives=True)

bids_files = layout.get(task="sleep", suffix="swaves", extension="tsv")
# events_bids_files = layout.get(task="sleep", suffix="events", extension="tsv")
# hypno_bids_files = layout.get(task="sleep", suffix="hypno", extension="tsv")
# swp = pd.concat([ bf.get_df().assign(participant_id=bf.entities["subject"]) for bf in swp_bids_files ])
# events = pd.concat([ bf.get_df().assign(participant_id=bf.entities["subject"]) for bf in events_bids_files ])
# hypno = pd.concat([ bf.get_df().assign(participant_id=bf.entities["subject"]) for bf in hypno_bids_files ])

df_list = []
for bf in bids_files:
    participant = bf.entities["subject"]
    swp_ = bf.get_df().assign(participant_id=f"sub-{participant}")
    events_bfile = layout.get(subject=participant, task="sleep", suffix="events", extension="tsv")[0]
    events_ = events_bfile.get_df().assign(participant_id=f"sub-{participant}")

# s = swp.query("participant_id=='101'")
# e = events.query("participant_id=='101'")
# h = hypno.query("participant_id=='101'")
# # df.groupby("subject")[]

    events_ = events_.query("description.str.endswith('.wav')")
    swp_ = swp_.query(f"Channel.eq('{channel}')")
    if not include_all_stages:
        swp_ = swp_.query("Stage.eq(3)")
    if not include_all_cycles:
        last_cue_onset = events_["onset"].max()
        swp_ = swp_.loc[swp_["Start"].lt(last_cue_onset), :]
    swp_["segment"] = swp_["Start"].apply(
        lambda x: (events_["onset"].add(6).lt(x)
            & events_["onset"].add(6+window_size).gt(x)).any()
    ).replace({True: "post-cue", False: "outside cueing"})
    # swp_["placement"] = swp_["postcue"].map({True: "postcue", False: "outside cue"})
    # swp_["precue"] = swp_["Start"].apply(lambda x: (events_["onset"].sub(60).lt(x) & events_["onset"].gt(x)).any() )
    ## adjust for precues found that are also postcue
    # swp_["precue"] = swp_["precue"] & ~swp_["postcue"]
    # assert not swp_[["precue", "postcue"]].sum(axis=1).eq(2).any()
    # swp_["placement"] = pd.NA
    # swp_.loc[swp_["postcue"], "placement"] = "postcue"
    # swp_.loc[swp_["precue"], "placement"] = "precue"
    # swp_ = swp_.dropna(subset="placement")
    df_list.append(swp_)

df = pd.concat(df_list, ignore_index=True)

if include_all_params:
    sw_params = [
        "Frequency",
        "Duration",
        "Slope",
        "PTP",
        "ValNegPeak",
        "ValPosPeak",
        "PhaseAtSigmaPeak",
        "ndPAC",
    ]
    col_wrap = 3
else:
    sw_params = ["Duration", "PTP"]
    col_wrap = 2


if include_all_params:
    df_long = df.melt(id_vars=["participant_id", "segment"],
        value_vars=sw_params, var_name="sw_parameter", value_name="score")

    palette = {"outside cueing": "gainsboro", "post-cue": "mediumpurple"}
    g = sns.catplot(
        data=df_long,
        col="sw_parameter", col_wrap=col_wrap, col_order=sw_params,
        x="participant_id", y="score",
        hue="segment", hue_order=list(palette), palette=palette,
        kind="point", height=2, aspect=1, sharex=True, sharey=False,
        dodge=0.2, join=False,
    )

    g.set_axis_labels("Participant ID", "Score")
    g.set_titles("{col_name}")

else:

    n_axes = len(sw_params)
    figsize = (n_axes*2 + 0.5, 2)
    fig, axes = plt.subplots(ncols=n_axes, figsize=figsize, sharex=True, sharey=False)

    palette = {"outside cueing": "moccasin", "post-cue": "mediumpurple"}
    for ax, param in zip(axes, sw_params):
        desc = df.groupby(["participant_id", "segment"])[param].agg(["mean", "sem"])

        yvals = desc["mean"].to_numpy()
        yerr = desc["sem"].to_numpy()
        xvals = np.arange(yvals.size)
        ax.errorbar(xvals[::2], yvals[1::2], yerr=yerr[1::2], fmt="o", color=palette["post-cue"])
        ax.errorbar(xvals[1::2], yvals[0::2], yerr=yerr[0::2], fmt="o", color=palette["outside cueing"])

        # ax.set_xlabel("Participant ID")
        ax.set_ylabel(
            "Slow-wave Duration (s)" if param == "Duration"
            else "Slow-wave Amplitude (uV)"
        )
        # ax.set_xticks([.5, 3.5])
        # ax.set_xticklabels(desc.index.get_level_values("participant_id").unique().tolist())
        # ax.set_xbound(lower=-1, upper=5)
        ax.tick_params(axis="x", direction="out", top=False)

        if ax.get_subplotspec().is_last_col():
            labels = {
                "outside cueing": "Other periods of\nslow-wave sleep",
                "post-cue": f"Within {window_size} s\nafter relaxation cue"
            }
            handles = [ plt.matplotlib.lines.Line2D([0], [0], marker="o",
                    label=labels[l], color=c, linewidth=0)
                for l, c in palette.items() ]
            legend = ax.legend(
                # title="Location of slow-wave",
                handles=handles[::-1],
                loc="upper left", bbox_to_anchor=(1, 1),
                # handlelength=1, handleheight=0.3,
                # handletextpad=,
                borderaxespad=0,
                labelspacing=1)

fig.align_ylabels()


# Export.
# utils.export_mpl(export_path)
