"""Plot summary sleep statistics across all participants."""
import argparse
from pathlib import Path

from bids import BIDSLayout
import pandas as pd

import utils


bids_root = Path(utils.config["bids_root"])
export_path = bids_root / "derivatives" / "task-sleep_sstats.tsv"
export_path_plot = bids_root / "derivatives" / "task-sleep_sstats.png"

layout = BIDSLayout(bids_root, validate=False, derivatives=True)

bids_files = layout.get(task="sleep", suffix="sstats", extension="tsv")

df = pd.concat(
    [bf.get_df().assign(participant_id="sub-" + str(bf.entities["subject"])) for bf in bids_files],
    ignore_index=True,
)

table = df.pivot(index="participant_id", columns="sleep_statistic", values="value").rename_axis(None, axis=1)

# Export
utils.export_tsv(table, export_path)


#### PLOTTING

pp = utils.load_participants_file()
plot_df = df.set_index("participant_id").join(pp["tmr_condition"]).reset_index()

cue_palette = dict(relax="mediumpurple", story="forestgreen")

palette = utils.load_participant_palette()

g = sns.catplot(
    kind="bar",
    data=plot_df,
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
utils.export_mpl(export_path_plot)
