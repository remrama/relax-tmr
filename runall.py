"""
Run all analysis scripts in order.

* `config.json` has global configuration parameters
* `environment.yaml` has python packages and environment info
* `utils.py` has utility functions
"""
import subprocess
import sys

from tqdm import tqdm

import utils


def run_command(command):
    """Run shell command and exit upon failure."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        sys.exit()


participants = utils.participant_values()

surveys = [
    "Initial",
    "Debriefing",
]

participant_scripts = [
    "source2raw_pvt",
    "source2raw_eeg",
    "calc_hypno",
    "plot_hypno",
    "calc_sstats",
    "calc_waves",
    "calc_lfp",
]

group_scripts = [
    "plot_pvt",
    "plot_alert",
    "plot_sstats",
    "plot_venice",
    "plot_soundcheck",
    "plot_waves"
]


for s in tqdm(surveys, desc="surveys"):
    command = f"python source2raw_qualtrics.py --survey {s}"
    run_command(command)

for script in participant_scripts:
    for p in tqdm(participants, desc=script):
        run_command(f"python {script}.py --participant {p}")

for script in (pbar := tqdm(group_scripts)):
    pbar.set_description(script)
    command = f"python {script}.py"
    if script == "plot_waves":
        for metric in ["Duration", "PTP", "Slope", "Frequency"]:
            for channel in ["Fz", "AFz"]:
                run_command(f"{command} -m {metric} -c {channel}")
    else:
        run_command(command)
