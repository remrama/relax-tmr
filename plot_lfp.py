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


root_dir = Path(utils.config["bids_root"])
layout = BIDSLayout(root_dir, derivatives=True, validate=False)
bids_files = layout.get(task="sleep", suffix="lfp", extension="tsv")

export_path = root_dir / "derivatives" / "task-sleep_lfp.png"

df = pd.concat(
    [bf.get_df().assign(participant_id="sub-" + str(bf.entities["subject"])) for bf in bids_files],
    ignore_index=True,
)
df = df.set_index("participant_id")
pp = utils.load_participants_file()
df = df.join(pp["tmr_condition"])


desc = df.groupby(["tmr_condition", "channel", "frequency"])["power"].agg(["mean", "sem"])
desc = desc.reset_index()

ax =  sns.lineplot(data=df.query("channel=='AFz'"), x="frequency", y="power", hue="tmr_condition")

utils.export_mpl(export_path)
