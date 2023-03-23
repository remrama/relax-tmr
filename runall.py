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

for y in [
    "Arousal_1", # rate your current level of Arousal (slider)
    "Pleasure_1", # rate your current level of Pleasure (slider)
    "Alertness_1", # How alert are you? 0-100
    "SSS", # pick what best represents how you are feeling right now (7-1)
    "PANAS_neg",
    "PANAS_pos",
    "STAI",
]:

    if y != "Alertness_1":
        # Test pre/post changes after sleep, by condition.
        run_command(f"python plot_prepost.py -c {y}")

    # Between-subjects post-sleep subjective responses.
    run_command(f"python plot_response.py -c {y}")

    for x in [
            "Difficult", # how difficult did you find the relaxation task? 1-7
            "Distracted", # how distracted were you while completing the relaxation task? 1-7
            "Enjoyable", # how enjoyable did you find the relaxation task? 1-7
            "Motivated", # how motivated were you to do well on the relaxation task? 1-7
            "Relaxed", # how relaxed were you while completing the relaxation task? 1-7
        ]:

        # Does engagement with Relaxation ask predict subjective responses.
        run_command(f"python plot_engagementXmood.py -x {x} -y {y}")

        if y in ["Arousal_1", "Pleasure_1", "PANAS_neg", "PANAS_pos", "SSS", "STAI"]:
            # Same thing but for change in response (controlling for baseline)
            run_command(f"python plot_engagementXmood.py -x {x} -y {y} -d")


