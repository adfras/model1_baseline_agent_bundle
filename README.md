# Model 1 Baseline Bundle

This repository is for a reproducible **Model 1** baseline on trial-level learner-response data. The current named dataset for this bundle is **DBE-KT22**.

## Dataset
- Name: `DBE-KT22`
- DOI: <https://doi.org/10.26193/6DZWOH>
- Paper: <https://arxiv.org/abs/2208.12651>

## How To Retrieve The Data
The repository includes a scripted fetch step that handles the ADA Dataverse access flow and downloads the published dataset files.

Run:

```powershell
py src/fetch_dbe_kt22.py
```

This downloads the dataset into:

```text
data/raw/DBE-KT22/
```

The script resolves and downloads these files from the Dataverse record:
- `1_DBE_KT22_file_descriptions_100102.xlsx`
- `1_Script_to_generate_sequences_100102_py.zip`
- `2_DBE_KT22_datafiles_100102_csv.zip`
- `2_DBE_KT22_Practice_Sequences_100102_json.zip`

To inspect the manifest without downloading the files:

```powershell
py src/fetch_dbe_kt22.py --manifest-only
```

## Non-Commit Rule
Downloaded dataset files are local working data and should not be committed. The repository ignore rules exclude the local `data/` tree so the retrieval process is documented, but the actual data stays out of version control.
