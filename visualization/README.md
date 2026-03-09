# Deep Research Reports Factuality Visualization Tool

![Visualization UI Preview](docs/images/visualization-preview.png)

A lightweight Flask web app for reviewing sentence-level factuality annotations in long-form deep research reports.

It lets you:
- browse report JSON files from nested folders,
- inspect each sentence and model-provided evidence,
- add human verdicts and rationale,
- preview rationale in rendered Markdown,
- save updates back to the source JSON,
- and submit timestamped annotation snapshots.

## What This Tool Visualizes

Each report contains:
- a full report body (`response`),
- a `sentences_info` array with sentence-level metadata.

The UI shows each sentence as a card and supports:
- sentence index display toggle,
- an icon next to sentence index when `human_verdict` is annotated,
- visual highlight for unsupported sentences (`human_verdict = contradictory` or `inconclusive`),
- quick open of full original report with the selected sentence highlighted,
- read-only agent verdict and agent reason,
- editable human verdict and human verdict reason,
- Markdown preview modal for human verdict reason.

## Project Structure

```text
visualization/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── data/                # Source reports loaded by the app
└── data_uploads/        # Submitted timestamped snapshots
```

## Requirements

- Python 3.10+ recommended
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python app.py
```

Use a custom source data directory:

```bash
python app.py --data-dir /path/to/reports
```

The app runs at:

```text
http://localhost:20001
```

Note: `app.py` currently hardcodes port `20001`.

## Input Report Format

The app validates report JSON against a strict schema.

### Required top-level fields

- `response` (string): full report markdown/text
- `sentences_info` (array): sentence-level entries

### Optional top-level fields

- `report_id` (string or null)
- `topic` (string or null)
- `model` (string or null)

### Allowed sentence fields

- `sentence` (string, required)
- `relevance` (`"1" | "2" | "3" | "4" | "5" | null`)
- `relevance_reason` (string or null)
- `agent_verdict` (`"supported" | "contradictory" | "inconclusive" | null`)
- `agent_reason` (string or null)
- `human_verdict` (`"supported" | "contradictory" | "inconclusive" | null`)
- `human_verdict_reason` (string or null)

### Minimal example

```json
{
  "report_id": "example_report_001",
  "topic": "Hybrid LRSD + Deep Learning for Video Analysis",
  "model": "gpt-4.1",
  "response": "Full report content in markdown...",
  "sentences_info": [
    {
      "sentence": "This is a claim from the report.",
      "agent_verdict": "inconclusive",
      "agent_reason": "Insufficient evidence in cited sources.",
      "human_verdict": null,
      "human_verdict_reason": null
    }
  ]
}
```

Legacy note: `question` is still accepted for backward compatibility, but new files should use `topic`.

## Usage Workflow

1. Put report `.json` files under `data/` (subfolders supported).
2. Open the app in browser.
3. Select a report from the dropdown.
4. Click a sentence card on the left.
5. Review:
   - agent verdict (read-only),
   - agent reason (rendered markdown, read-only).
6. Annotate:
   - set `Human Verdict`,
   - write `Human Verdict Reason`.
7. Click `Preview` to view rendered markdown for human reason.
8. Click `Save Annotations` to write changes back to the same file under `data/`.
9. Click `Submit` to create a timestamped snapshot under `data_uploads/`.

## API Endpoints

- `GET /api/list_reports?dir=<subdir>`
  - Lists folders and report files under `data/`.
- `GET /api/load_report?report=<path_without_.json>`
  - Loads and validates one report.
- `POST /api/save_annotation?report=<path_without_.json>`
  - Validates and overwrites source report with updated content.
- `POST /api/upload`
  - Accepts multipart file or JSON body and stores a validated timestamped copy in `data_uploads/`.
- `GET /api/view_full_report?report=<path_without_.json>&sentence=<idx>`
  - Renders full report HTML and highlights selected sentence.

## Upload Output Layout

Submissions are saved as:

```text
data_uploads/<source_report_subdir>/submission_<username>/<basename>_timestamp_YYYYMMDD_HHMMSS.json
```

If username is missing, `anonymous` is used.

## Notes

- File path handling is constrained to app roots (`data/`, `data_uploads/`) for safety.
- Markdown rendering in the main app uses `marked` + `DOMPurify` in the browser.
- Full report rendering (`/api/view_full_report`) uses Python `markdown` with extensions.
