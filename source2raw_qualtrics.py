"""Convert a single qualtrics survey to tsv and companion json."""
import argparse
from datetime import timezone
from pathlib import Path

import numpy as np
import pyreadstat

import utils


parser = argparse.ArgumentParser()
parser.add_argument(
    "--survey", type=str, required=True, choices=["Screening", "Initial", "Debriefing"]
)
args = parser.parse_args()

survey_name = args.survey


################################################################################
# SETUP
################################################################################

survey_descriptions = {
    "Screening": "A survey of demographics, completed before arrival.",
    "Initial": "A survey of demographics and state measures, completed before tasks.",
    "Debriefing": "A survey of more demographics, state measures, and debriefing questions, completed after tasks",
}

# Identify location to import from and export to.
# Find the relevant file with a glob-search and confirming filename to Qualtrics convention.
root_dir = Path(utils.config["bids_root"])
import_dir = root_dir / "sourcedata" / "qualtrics"
potential_import_paths = list(import_dir.glob(f"Relaxation-TMR+{survey_name}*.sav"))
assert len(potential_import_paths) == 1, "Only the latest Qualtrics file should be present."
import_path = potential_import_paths[0]
export_path = root_dir / "phenotype" / f"{survey_name.lower()}.tsv"

# Load the Qualtrics survey data and metadata.
df, meta = pyreadstat.read_sav(import_path)


################################################################################
# PREPROCESSING
################################################################################


# Add timezone information to timestamps.
for datecol in ["RecordedDate", "StartDate", "EndDate"]:
    df[datecol] = df[datecol].dt.tz_localize("US/Mountain").dt.tz_convert(timezone.utc)

# Remove piloting/testing data and anyone who closed the survey out early.
df = (df
    .query("DistributionChannel == 'anonymous'")
    .query("Status == 0")
    .query("Finished == 1")
    .query("Progress == 100")
)

assert df["ResponseId"].is_unique, "Unexpectedly found non-unique Response IDs."
assert df["UserLanguage"].eq("EN").all(), "Unexpectedly found non-English responses."

# Remove default Qualtrics columns.
default_qualtrics_columns = [
    "StartDate",
    "EndDate",
    "RecordedDate",
    "Status",
    "DistributionChannel",
    "Progress",
    "Finished",
    "ResponseId",
    "UserLanguage",
    "Duration__in_seconds_",
]
df = df.drop(columns=default_qualtrics_columns)

# Remove time info
time_columns = [ c for c in df if (
    c.endswith("First_Click") or c.endswith("Last_Click") or
    c.endswith("Page_Submit") or c.endswith("Click_Count"))
]
df = df.drop(columns=time_columns)

# # Validate Likert scales.
# # Sometimes when the Qualtrics question is edited, the scale gets changed "unknowingly".
# # Here, check to make sure everything starts at 1 and increases by 1.
# for var in df:
#     if var in meta.variable_value_labels:
#         levels = meta.variable_value_labels[var]
#         values = list(levels.keys())
#         assert values[0] == 1, f"{var} doesn't start at 1, recode in Qualtrics."
#         assert values == sorted(values), f"{var} isn't increasing, recode in Qualtrics."
#         assert not np.any(np.diff(values) != 1), f"{var} isn't linear, recode in Qualtrics."

# Replace empty strings with NaNs.
df = df.replace("", np.nan)

# Rename participant ID columns to match rest of files.
# Make sure there are not random/test participants.
# Change participants values to full IDs.
df = df.dropna(subset="SID")
df = df.set_index("SID").rename_axis("participant_id")
df.index = df.index.astype(int)
participants = utils.participant_values()
df = df.loc[participants]
df.index = df.index.map("sub-{:03d}".format)


################################################################################
# GENERATING BIDS SIDECAR
################################################################################


# Generate BIDS sidecar with column metadata.
sidecar = {
    "MeasurementToolMetadata": {
        "Description": survey_descriptions[survey_name],
    }
}
for col in df:
    column_info = {}
    # Get probe string (if present).
    if col in meta.column_names_to_labels:
        column_info["Probe"] = meta.column_names_to_labels[col]
    # Get response option strings (if present).
    if col in meta.variable_value_labels:
        levels = meta.variable_value_labels[col]
        levels = { int(float(k)): v for k, v in levels.items() }
        column_info["Levels"] = levels
    if column_info:
        sidecar[col] = column_info


################################################################################
# EXPORTING
################################################################################


utils.export_tsv(df, export_path)
utils.export_json(sidecar, export_path.with_suffix(".json"))
