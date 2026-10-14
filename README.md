# SOW1 ISAC measurement console

Python desktop GUI for the workflow in `second_outdoor_VA_tests.ipynb`: inspect the channel in time domain, restore the calibrated segmented sweep, and acquire a manually positioned virtual array. The original notebooks remain available.

## Run

Use Python 3.12 and uv. The editable driver repositories must be beside this repository:

```text
parent/
  sow1-measurement-codes/
  mcg-lab-vna-control/
  mcg-lab-triaxial-positioner-control/
```

```powershell
uv sync
uv run python -m sow1_gui --demo
```

`--demo` automatically connects a synthetic instrument and opens live time-domain plots. It does not contact hardware. To launch the normal console:

```powershell
uv run python -m sow1_gui
```

Select **Keysight VNA**, enter a VISA resource or leave it empty for auto-detection, then connect. Real acquisition requires a working VISA installation and the laboratory VNA driver. The GUI currently targets the Keysight workflow used by the outdoor notebook.

## Campaign workflow

1. Set the output folder, setup and row indices, array geometry, segmented frequency table, IF bandwidth, powers, sweep count, and antenna metadata.
2. In **Calibration**, prepare the segmented configuration. Use the guided SOL controls if calibration is needed, or verify the existing active calset on the VNA. Record a verification note and confirm coverage for the required ports and segments.
3. Select **Restore · segmented**. Capture the S21/S31/S41 through references separately if needed.
4. Select **Inspect · time domain**, then **Start live view**. Adjust the physical scene. The **Time view** tab controls the sweep, transform window, and each trace's VNA X/Y axes.
5. Select **Restore · segmented**, then **Open row**. Manually place the antenna at index `i`, confirm the position, and acquire. The first three traces are recorded everywhere; the remaining traces are recorded only at horizontal endpoints.
6. Close the completed row, return to time-domain inspection, update the next setup/row, and repeat. Calibration confirmation is retained across mode switches when the segmented settings match.

The GUI does not move the positioner. A single worker serializes instrument access. **Stop after current sweep** preserves finished data and stops after the current driver call returns; it cannot interrupt a blocking VISA transfer.

## Data and recovery

Each run creates a unique folder under the selected output folder, explicitly labelled `SIMULATION` or `MEASUREMENT`. Each sweep updates an atomic `setup_N_nv_J.partial.npz` checkpoint. A fully acquired row becomes `setup_N_nv_J.npz`.

Rows retain the notebook's `row_sparams`, `conf_params`, `timestamp_list`, and BS/UE metadata keys. They also include the frequency vector, measured mask, JSON settings, calibration confirmation, and instrument identity. Use the mask: the zero-filled endpoint-only traces at interior positions are **not measurements**. Timestamps now include UTC dates and time zones.

Use **Resume checkpoint** after an interruption, then reconfirm the instrument calibration and restore segmented mode. Completed sweeps are skipped. Use **Saved results & evolution** to inspect a trace/repeat/position and its evolution across the row.

`data/`, `tmp/`, and `docs/` are ignored by Git. Local detailed documentation is in `docs/gui-user-guide.md`, with an implementation report in `docs/reports/gui-implementation.md`.

## Verification

```powershell
uv run python -m unittest discover -s tests -v
```

Tests use simulation and mocked driver calls, including a Qt workflow test. Real instrument behavior and calibration coverage must be validated on the laboratory VNA before a production campaign.
