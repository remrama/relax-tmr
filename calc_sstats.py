"""Generate summary sleep statistics for one participant."""
import argparse
from pathlib import Path

import pandas as pd
import yasa

import utils


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--participant", type=int, required=True, choices=utils.participant_values()
)
args = parser.parse_args()


################################################################################
# SETUP
################################################################################

participant = args.participant
participant_id = f"sub-{participant:03d}"
task_id = "task-sleep"

bids_root = Path(utils.config["bids_root"])
import_path = bids_root / "derivatives" / participant_id / f"{participant_id}_{task_id}_hypno.tsv"
export_path = bids_root / "derivatives" / participant_id / f"{participant_id}_{task_id}_sstats.tsv"

hypno = pd.read_csv(import_path, sep="\t")
hypno_int = hypno["value"].to_numpy()
assert hypno["duration"].nunique() == 1
epoch_length = hypno["duration"].unique()[0]
sfreq = 1 / epoch_length

sstats_dict = yasa.sleep_statistics(hypno_int, sf_hyp=sfreq)
sstats = pd.Series(sstats_dict).rename_axis("sleep_statistic").rename("value")

sstats_sidecar = {
    "TIB": {
        "LongName": "Time in Bed",
        "Description": "total duration of the hypnogram"
    },
    "SPT": {
        "LongName": "Sleep Period Time",
        "Description": "duration from first to last period of sleep"
    },
    "WASO": {
        "LongName": "Wake After Sleep Onset",
        "Description": "duration of wake periods within SPT"
    },
    "TST": {
        "LongName": "Total Sleep Time",
        "Description": "total duration of N1 + N2 + N3 + REM sleep in SPT"
    },
    "SE": {
        "LongName": "Sleep Efficiency",
        "Description": "TST / TIB * 100 (%)"
    },
    "SME": {
        "LongName": "Sleep Maintenance Efficiency",
        "Description": "TST / SPT * 100 (%)"
    },
    "SOL": {
        "LongName": "Sleep Onset Latency",
        "Description": "Latency to first epoch of any sleep"
    }
}

# Export.
utils.export_tsv(sstats, export_path)
utils.export_json(sstats_sidecar, export_path.with_suffix(".json"))
