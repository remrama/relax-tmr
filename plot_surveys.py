"""
This plots individual subject responses as horizontal bars.
Keeping for now but ultimately moving towards means per group.
"""
from pathlib import Path

from bids import BIDSLayout
import numpy as np
import pandas as pd
import colorcet as cc

import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns

import utils


# parser = argparse.ArgumentParser()
# parser.add_argument("--participant", type=int, default=101)
# parser.add_argument("--overwrite", action="store_true")
# args = parser.parse_args()

# dmlab.plotting.set_matplotlib_style("technical")
plt.rcParams["font.sans-serif"] = "Arial"


bids_root = Path(utils.config["bids_root"])

layout = BIDSLayout(bids_root, validate=False)
# stimuli_dir = bids_root / "stimuli"
# bids_file = layout.get(subject=participant, task="sleep", suffix="eeg", extension="fif")[0]
# pattern = "derivatives/sub-{subject}/sub-{subject}_task-{task}_hypno.tsv"
# export_path = Path(layout.build_path(bids_file.entities, pattern, validate=False))
bids_file1 = layout.get(suffix="screening", extension="tsv")[0]
bids_file2 = layout.get(suffix="initial", extension="tsv")[0]
bids_file3 = layout.get(suffix="debriefing", extension="tsv")[0]
df = bids_file.get_df()


# cmap = cc.cm.glasbey_dark
# df["color"] = df["SID"].map(cmap)
palette = utils.load_participant_palette()
df["SID"] = df["participant_id"].str.strip("-").str[-1].astype(int)
df["color"] = df["participant_id"].map(palette)
df["color"] = "mediumpurple"

df["SleepQuality"] = df["SleepQuality"].replace({1: 4, 2: 3, 3: 2, 4: 1})
df["SSS"] = df["SSS"].replace({1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1})

### Calculate aggregate survey scores

def imputed_sum(row):
    if row.isna().mean() > .5:
        # Return nan if more than half of responses are missing.
        return np.nan
    else:
        return row.fillna(row.mean()).sum()

def imputed_mean(row):
    return np.nan if row.isna().mean() > .5 else row.fillna(row.mean()).mean()


for survey in ["PANAS", "STAI", "TAS"]:
    columns = [ c for c in df if c.startswith(survey+"_") ]
    if survey == "PANAS":
        positive_probes = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]
        positive_columns = [ c for c in columns if int(c.split("_")[-1]) in positive_probes ]
        negative_columns = [ c for c in columns if c not in positive_columns ]
        df["PANAS_pos"] = df[positive_columns].apply(imputed_sum, axis=1)
        df["PANAS_neg"] = df[negative_columns].apply(imputed_sum, axis=1)
    elif survey in ["STAI", "TAS"]:
        df[survey] = df[columns].apply(imputed_sum, axis=1)
    else:
        raise ValueError(f"Not prepared for {survey} survey!")


########### Interesting variables

probes = {

    "alertness": [
        "Alertness_1", # How alert are you? 0-100
        "Alertness_2", # How refreshed do you feel? 0-100
        "Alertness_3", # How well are you able to concentrate? 0-100
        "SSS", # pick what best represents how you are feeling right now (7-1)
    ],

    "affect": [
        "Arousal_1", # rate your current level of Arousal (slider)
        "Pleasure_1", # rate your current level of Pleasure (slider)
        # "PANAS_pos",
        # "PANAS_neg",
        # "STAI",
    ],

    "sleep_quality": [
        "EasytoSleep", # How easy was it to fall asleep? 1-5
        "SleepOverall", # How was your sleep? 1-5
        "SleepQuality", # How would you rate your sleep quality overall from last night? 4-1
        "EasytoWake", # How easy was it to wake up? 1-5
        "SleepCalm", # Was your sleep calm? 1-5
        "WellRested", # Do you feel well-rested? 1-3
        "EnoughSleep", # Did you get enough sleep? 1-5
    ],

    "task_engagement": [
        "Difficult", # how difficult did you find the relaxation task? 1-7
        "Distracted", # how distracted were you while completing the relaxation task? 1-7
        "Enjoyable", # how enjoyable did you find the relaxation task? 1-7
        "Motivated", # how motivated were you to do well on the relaxation task? 1-7
        "Relaxed", # how relaxed were you while completing the relaxation task? 1-7
    ],

    "dreaming": [
        # "DreamRecall",
        "DreamArousal_1",
        "DreamPleasure_1",
        # "DreamReport",
    ],

    # "trait": [
    #     "TAS",
    #     # meditation probes
    # ],

}

##### interaction between difficulty to sleep and sleep quality
#### plot any differences pre-post (STAI? Alertness/SSS. arousal/pleasure. others?)
#### plot sleep quality from test relative to screening PSQI
#### maybe do venice questions, to test for ceiling at this point


for group_name, probe_list in probes.items():

    n_axes = len(probe_list)
    figsize = (8, 2*n_axes)
    fig, axes = plt.subplots(nrows=n_axes, figsize=figsize, sharex=False, sharey=False)

    for ax, probe in zip(axes, probe_list):

        ax.barh(data=df, y="SID", width=probe, color="color", height=0.8)

        probe_txt = bids_file.entities[probe]["Probe"]
        if "Arousal_" in probe or "Pleasure_" in probe:
            probe_txt = probe_txt.split(" - ")[0]
            
        if "Levels" in bids_file.entities[probe]:
            probe_lvls = bids_file.entities[probe]["Levels"]
            ticks, ticklabels = zip(*[ (int(x[0]), y) for x, y in probe_lvls.items() ])
            ticklabels = [ x.replace("Very Unrelaxed", "Not Relaxed") for x in ticklabels ]
            if probe == "SSS":
                ticklabels = [ x.split(";")[0].split(",")[0] for x in ticklabels ]
                ticklabels[1] = "Functioning at\nhigh level"
                ticklabels[-1] = "No longer\nfighting sleep"
            if probe in ["SSS", "SleepQuality"]:
                # ticks = ticks[::-1]
                ticklabels = ticklabels[::-1]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
        elif (probe.startswith("Alertness")
            or "Arousal_" in probe or "Pleasure_" in probe):
            ticks = [0, 50, 100]
            ax.set_xticks(ticks)

        ax.set_xbound(lower=min(ticks), upper=max(ticks))
        # ax.set_xlabel(probe)
        # ax.set_yticks([101, 102])
        # ax.set_yticklabels(["sub-101", "sub-102"])
        ax.set_title(probe_txt)
        # ax.set_ylabel("Participant ID")
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()
        # if probe in ["SSS", "SleepQuality"]:
        #     ax.invert_xaxis()
        # ax.grid(False)
        # ax.tick_params(top=False, right=False)
        # ax.tick_params(direction="in", axis="both")

    fig.suptitle(group_name.replace("_", " ").title())
    plt.tight_layout()

    export_path = bids_root / "derivatives" / f"debriefing-{group_name}__.png"
    utils.export_mpl(export_path)
