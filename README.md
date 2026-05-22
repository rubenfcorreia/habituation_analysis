# Habituation Analysis GUI

PyQt5 + Matplotlib GUI for browsing habituation experiments under
`/data/common/habituation`, caching derived traces in
`/data/common/habituation/gui_output/`, and running day-wise statistics.

## Launch

Run the GUI from the repository root:

```bash
python run_gui.py
```

or, if you prefer the module entrypoint:

```bash
python -m habituation_analysis
```

or, if the console script is installed:

```bash
habituation-analysis
```

If you are using the Sci environment on this machine, the direct launcher is:

```bash
/home/rubencorreia/miniconda3/envs/sci/bin/python3 /home/rubencorreia/code/habituation_analysis/run_gui.py
```

## Quick Start

1. Launch the GUI with `python run_gui.py`, `python -m habituation_analysis`, or `habituation-analysis`.
2. Choose one animal from the `Animal` drop-down.
3. Choose an `expID` for a single session, or `Overall` for the selected animal.
4. In `Metrics`, review the pupil trace, locomotion trace, and right-eye video.
5. Adjust shared pupil percentile cutoffs if needed, then mark the session as `Pre-processed` once you are happy with it.
6. Switch to `Statistics` and run the analysis when you want the day-wise summary.

## What The GUI Does

- Loads all experiments found under `/data/common/habituation`.
- Caches analysis state, edits, and exported statistics under
  `/data/common/habituation/gui_output/`.
- Uses the right-eye video as the main video source.
- Derives locomotion from the frame-times CSV in
  `/data/Remote_Repository/[animalID]/[expID]/`.
- Keeps video and locomotion as separate timebases. If the right video is
  shorter than the CSV trace, the GUI keeps the full locomotion timeline and
  shows a warning in session view.
- Lets you review pupil dynamics, locomotion, not visible pupil intervals, and
  summary statistics.

## Typical Workflow

### 1. Open The GUI

Launch the app and wait for the loading window to finish. The main window will
restore the last animal, expID, and view you used. On first launch, it opens
the first available animal/session so the plots appear immediately.

### 2. Pick An Animal

Use the `Animal` drop-down at the top of the window to choose one animal.

- `All` gives you the cohort-wide browser.
- A specific animal lets you inspect that animal's sessions.

### 3. Pick A Session Or Overall View

Use the `ExpID` drop-down to choose:

- a specific expID to inspect one session
- `Overall` to see all sessions for the selected animal in one view

The arrow buttons next to the drop-down let you move through the animal's
sessions one by one.

## Metrics Tab

The `Metrics` tab is the main review view.

### Pupil Plot And Thresholds

- The pupil plot shows the z-scored pupil radius trace.
- The three threshold controls represent shared percentile cutoffs.
- The percentile values are global across all experiments.
- The absolute z-score values shown on the right are animal-specific, because
  each animal can map the same percentiles to different values.
- You can drag the threshold lines on the plot or edit the percentile values
  directly.

The small note under the threshold panel summarizes this behavior:

`Shared percentiles, animal-specific absolute values.`

### Locomotion Plot

- The locomotion trace is plotted underneath the pupil trace.
- It is calculated from the raw rotary encoder values in the frame-times CSV.
- The GUI uses the resampled BV2-style wheel speed signal.
- In some sessions, the right video ends before the locomotion CSV. The GUI
  keeps the full locomotion timebase, shows a warning when the mismatch is
  larger than 5 s, and marks the video coverage end on the plot.

### Video Review

- In session view, the right-eye video is shown beside the plots.
- Use the playback controls to move through the video.
- A vertical line moves along the plots as the video plays.
- The pupil fit overlay is shown on the video, matching the eye-view GUI.

### Mark Not Visible Pupil Intervals

When the pupil is hard to see, you can mark an interval directly from video
playback:

1. Play or scrub to the point where the not visible period starts.
2. Click `Set start`.
3. Move to the point where the not visible period ends.
4. Click `Set end / add`.
5. Repeat for more intervals if needed.
6. Click `Save intervals` to store the edit.

These manual masks are saved in the GUI output folder and are used by the
statistics tab.

### Pre-Processed Checklist

The `Pre-processed` checkbox is a manual checklist for each expID.

- Check it when you have reviewed the session and are happy with the current
  threshold settings.
- If any shared pupil percentile changes, the checkbox becomes unchecked
  automatically.
- This is session-specific bookkeeping, not an automated pipeline step.

### Important Metrics Tab Behavior

- `Overall` works for a specific animal.
- `All` is not available in the Metrics tab and shows an informational message.
- The current animal's statistics are marked dirty when thresholds or masks
  change, so you know the statistics should be rerun.

## Statistics Tab

The `Statistics` tab summarizes the selected scope by day.

### Running Statistics

When you open the tab, the GUI asks if you want to run the analysis.

The summary includes:

- locomotion fraction by day
- face motion fraction by day
- pupil state fractions by day
- lag to the first pupil state after the first minute of each session
- pupil state probability as a function of experiment progress from 0 to 100%

Only sessions longer than 30 minutes are included in the progress-based
probability analysis.

### Outputs

When statistics finish, the GUI saves:

- a JSON report
- an SVG figure
- a PNG figure

All outputs are written under `/data/common/habituation/gui_output/stats/`.

## Update Dataset

Use `Update dataset` if new experiments were added or source files changed.

- The GUI rescans `/data/common/habituation`.
- The cached dataset index is refreshed.
- Existing analysis is kept unless the source data changed.

## Files And Cache

- Source experiments: `/data/common/habituation`
- Cached GUI state: `/data/common/habituation/gui_output/`
- Locomotion CSVs: `/data/Remote_Repository/[animalID]/[expID]/`

## Practical Tips

- If the Metrics tab is blank, check whether you selected `All`.
- If a session was already pre-processed and you change the thresholds, the
  session will be marked as needing review again.
- The GUI restores the last browser position when it starts, so you can resume
  where you left off.
