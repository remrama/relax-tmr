"""Plot summary sleep statistics across all participants."""
import argparse
from pathlib import Path

from bids import BIDSLayout
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import utils


bids_root = Path(utils.config["bids_root"])
export_path = bids_root / "derivatives" / "sleepstats.png"

layout = BIDSLayout(bids_root, validate=False, derivatives=True)

bids_files = layout.get(task="sleep", suffix="sstats", extension="tsv")

df = pd.concat(
    [bf.get_df().assign(participant_id="sub-" + str(bf.entities["subject"])) for bf in bids_files],
    ignore_index=True,
)

participants = utils.load_participants_file()
df = df.set_index("participant_id").join(participants["tmr_condition"]).reset_index()

cue_palette = dict(relax="mediumpurple", story="forestgreen")

palette = utils.load_participant_palette()

g = sns.catplot(
    kind="bar",
    data=df,
    col="sleep_statistic",
    hue="tmr_condition",
    x="sleep_statistic",
    y="value",
    palette=cue_palette,
    col_wrap=7,
    height=1,
    aspect=1,
    sharex=False, sharey=False,
)
g.set_titles("")
g.set_xlabels("")


# Export.
utils.export_mpl(export_path)
