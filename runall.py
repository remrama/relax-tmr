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
participant_scripts = [
    "source2raw_eeg",
    "calc_hypno",
    "plot_hypno",
    "calc_sstats",
    "calc_swaves",
]
surveys = ["Initial", "Debriefing"]

for s in tqdm(surveys, desc="surveys"):
    command = f"python source2raw_qualtrics.py --survey {s}"
    run_command(command)

for script in participant_scripts:
    for p in tqdm(participants, desc=script):
        run_command(f"python {script}.py --participant {p}")

# for participant in (pbar1 := tqdm(all_participants)):
#     pbar1.set_description(f"Processing {participant}")
#     for filename in (pbar2 := tqdm(all_participant_scripts, leave=False)):
#         pbar2.set_description(filename)
#         command = f"python {filename}.py --participant {participant}"
#         run_command(command)

# run_command("python venice_analysis.py")
