"""Plot interaction between pre/post measure and TMR condition."""
import argparse
from pathlib import Path

from bids import BIDSLayout
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns

import utils


utils.set_matplotlib_style()

cue_order = utils.config["cue_order"]
cue_palette = utils.config["cue_palette"]
cue_labels = utils.config["cue_labels"]
participant_palette = utils.load_participant_palette()


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--column", type=str, default="Alertness_1")
parser.add_argument("-c2", "--column2", type=str, default="TAS")
args = parser.parse_args()

column = args.column
trait_column = args.column2

root_dir = Path(utils.config["bids_root"])
import_path_pre = root_dir / "phenotype" / "initial.tsv"
import_path_post = root_dir / "phenotype" / "debriefing.tsv"
export_path = root_dir / "derivatives" / f"prepost-{column}.png"
export_path2 = root_dir / "derivatives" / f"prepost-{column}_corr-{trait_column}.png"

# layout = BIDSLayout(root_dir, validate=False)
# bids_file = layout.get(suffix="debriefing", extension="tsv")[0]
# df = bids_file.get_df()
pre = pd.read_csv(import_path_pre, index_col="participant_id", sep="\t")
post = pd.read_csv(import_path_post, index_col="participant_id", sep="\t")
pre_meta = utils.import_json(import_path_pre.with_suffix(".json"))
post_meta = utils.import_json(import_path_post.with_suffix(".json"))

# scrn_path = root_dir / "phenotype" / "screening.tsv"
# scrn = pd.read_csv(scrn_path, index_col="participant_id", sep="\t")
# scrn_meta = utils.import_json(scrn_path.with_suffix(".json"))

participants = utils.load_participants_file()
pre = pre.join(participants["tmr_condition"]).set_index("tmr_condition", append=True)
post = post.join(participants["tmr_condition"]).set_index("tmr_condition", append=True)

for survey in ["PANAS", "STAI", "TAS"]:
    if survey == "TAS":
        post = utils.agg_questionnaire_columns(post, survey)
    else:
        pre = utils.agg_questionnaire_columns(pre, survey)
        post = utils.agg_questionnaire_columns(post, survey)

pre = pre.assign(time="pre").set_index("time", append=True)
post = post.assign(time="post").set_index("time", append=True)

df = pd.concat([pre, post]).dropna(axis=1)
df = df.unstack("time")

df = df[column]

diff = (df["post"] - df["pre"]).to_frame(column)

df = df.melt(ignore_index=False, value_name=column)
df = df.reset_index()

g = sns.catplot(kind="point", data=df, y=column, x="time", hue="tmr_condition", order=["pre", "post"], dodge=True, palette=cue_palette)
# ax = sns.catplot(kind="swarm", data=df, y=column, hue="tmr_condition", x="time", order=["pre", "post"], dodge=True)

utils.export_mpl(export_path)

if trait_column in pre:
    diff = diff.join(pre[trait_column]).reset_index()
elif trait_column in post:
    diff = diff.join(post[trait_column]).reset_index()

g = sns.lmplot(data=diff, x=trait_column, y=column, hue="tmr_condition", palette=cue_palette)
g.set_ylabels(f"{column} change after sleep")

utils.export_mpl(export_path2)

