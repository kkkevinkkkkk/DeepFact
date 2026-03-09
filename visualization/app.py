from __future__ import annotations

import argparse
import errno
import json
import os
import re
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

DATA_ROOT = Path("data").resolve()
UPLOAD_ROOT = Path("data_uploads").resolve()
ALLOWED_EXTENSIONS = {"json"}

DATA_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

TOP_LEVEL_REQUIRED = {"response", "sentences_info"}
TOP_LEVEL_OPTIONAL = {"report_id", "topic", "model"}
TOP_LEVEL_ALLOWED = TOP_LEVEL_REQUIRED | TOP_LEVEL_OPTIONAL

SENTENCE_REQUIRED = {"sentence"}
SENTENCE_OPTIONAL = {
    "relevance",
    "relevance_reason",
    "agent_verdict",
    "agent_reason",
    "human_verdict",
    "human_verdict_reason",
}
SENTENCE_ALLOWED = SENTENCE_REQUIRED | SENTENCE_OPTIONAL

RELEVANCE_ALLOWED = {"1", "2", "3", "4", "5", None}
VERDICT_ALLOWED = {"supported", "contradictory", "inconclusive", None}


def set_data_root(data_dir: str) -> None:
    global DATA_ROOT
    DATA_ROOT = Path(data_dir).expanduser().resolve()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_report_arg(report_arg: str) -> str:
    value = (report_arg or "").strip().replace("\\", "/")
    value = value.lstrip("/")
    if value.endswith(".json"):
        value = value[:-5]
    return value


def ensure_within_root(path: Path, root: Path) -> None:
    if path != root and root not in path.parents:
        raise ValueError("Invalid path outside allowed directory")


def resolve_report_file(report_arg: str) -> tuple[Path, str]:
    report_rel = normalize_report_arg(report_arg)
    if not report_rel:
        raise ValueError("Missing report path")

    file_path = (DATA_ROOT / f"{report_rel}.json").resolve()
    ensure_within_root(file_path, DATA_ROOT)
    return file_path, report_rel


def resolve_data_directory(dir_arg: str) -> tuple[Path, str]:
    current_dir = (dir_arg or "").strip().replace("\\", "/").strip("/")
    directory = (DATA_ROOT / current_dir).resolve() if current_dir else DATA_ROOT
    ensure_within_root(directory, DATA_ROOT)
    return directory, current_dir


def resolve_upload_directory(current_report_path: str) -> Path:
    report_rel = normalize_report_arg(current_report_path)
    report_dir = os.path.dirname(report_rel) if report_rel else ""
    upload_dir = (UPLOAD_ROOT / report_dir).resolve() if report_dir else UPLOAD_ROOT
    ensure_within_root(upload_dir, UPLOAD_ROOT)
    return upload_dir


def validate_sentence(item: Any, index: int) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []

    if not isinstance(item, dict):
        return None, [f"sentences_info[{index}] must be an object"]

    extra = sorted(set(item.keys()) - SENTENCE_ALLOWED)
    if extra:
        errors.append(f"sentences_info[{index}] has unsupported keys: {', '.join(extra)}")

    sentence = item.get("sentence")
    if not isinstance(sentence, str):
        errors.append(f"sentences_info[{index}].sentence must be a string")

    relevance = item.get("relevance")
    if relevance not in RELEVANCE_ALLOWED:
        errors.append(f"sentences_info[{index}].relevance must be one of 1|2|3|4|null")

    agent_verdict = item.get("agent_verdict")
    if agent_verdict not in VERDICT_ALLOWED:
        errors.append(
            f"sentences_info[{index}].agent_verdict must be supported|contradictory|inconclusive|null"
        )

    human_verdict = item.get("human_verdict")
    if human_verdict not in VERDICT_ALLOWED:
        errors.append(
            f"sentences_info[{index}].human_verdict must be supported|contradictory|inconclusive|null"
        )

    for field in ("relevance_reason", "agent_reason", "human_verdict_reason"):
        value = item.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"sentences_info[{index}].{field} must be a string or null")

    if errors:
        return None, errors

    normalized_sentence = {
        "sentence": sentence,
        "relevance": relevance,
        "relevance_reason": item.get("relevance_reason"),
        "agent_verdict": agent_verdict,
        "agent_reason": item.get("agent_reason"),
        "human_verdict": human_verdict,
        "human_verdict_reason": item.get("human_verdict_reason"),
    }
    return normalized_sentence, []


def validate_report_schema(payload: Any) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []

    if not isinstance(payload, dict):
        return None, ["Top-level JSON must be an object"]

    extra_top = sorted(set(payload.keys()) - TOP_LEVEL_ALLOWED)
    if extra_top:
        errors.append(f"Unsupported top-level keys: {', '.join(extra_top)}")

    missing = sorted(TOP_LEVEL_REQUIRED - set(payload.keys()))
    if missing:
        errors.append(f"Missing required top-level keys: {', '.join(missing)}")

    for field in TOP_LEVEL_OPTIONAL:
        if field in payload and payload[field] is not None and not isinstance(payload[field], str):
            errors.append(f"{field} must be a string or null")

    response = payload.get("response")
    if "response" in payload and not isinstance(response, str):
        errors.append("response must be a string")

    sentences = payload.get("sentences_info")
    normalized_sentences: list[dict[str, Any]] = []
    if "sentences_info" in payload:
        if not isinstance(sentences, list):
            errors.append("sentences_info must be an array")
        else:
            for idx, sentence_item in enumerate(sentences):
                normalized_sentence, sentence_errors = validate_sentence(sentence_item, idx)
                if sentence_errors:
                    errors.extend(sentence_errors)
                elif normalized_sentence is not None:
                    normalized_sentences.append(normalized_sentence)

    if errors:
        return None, errors

    normalized: dict[str, Any] = {
        "response": response,
        "sentences_info": normalized_sentences,
    }

    for key in TOP_LEVEL_OPTIONAL:
        if key in payload:
            normalized[key] = payload.get(key)

    return normalized, []


def load_and_validate_report(file_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            raw_data = json.load(fh)
    except json.JSONDecodeError as exc:
        return None, [f"Invalid JSON: {exc}"]

    return validate_report_schema(raw_data)


def highlight_sentence_in_markdown(report_markdown: str, sentence: str) -> str:
    if not sentence:
        return report_markdown

    highlight_start = '<mark style="background-color: yellow; padding: 2px 4px; border-radius: 3px;">'
    highlight_end = "</mark>"

    def wrap_span(start_idx: int, end_idx: int) -> str:
        return (
            report_markdown[:start_idx]
            + highlight_start
            + report_markdown[start_idx:end_idx]
            + highlight_end
            + report_markdown[end_idx:]
        )

    # 1) Exact match first
    exact_start = report_markdown.find(sentence)
    if exact_start != -1:
        return wrap_span(exact_start, exact_start + len(sentence))

    cleaned_sentence = re.sub(r"[*_`~\[\]()]", "", sentence)

    # 2) Exact match on markdown-stripped sentence
    if cleaned_sentence:
        cleaned_start = report_markdown.find(cleaned_sentence)
        if cleaned_start != -1:
            return wrap_span(cleaned_start, cleaned_start + len(cleaned_sentence))

    # 3) Whitespace-tolerant regex match (handles line wraps/newlines)
    def regex_find(text: str) -> re.Match[str] | None:
        tokens = [re.escape(tok) for tok in re.findall(r"\S+", text)]
        if not tokens:
            return None
        pattern = re.compile(r"\s+".join(tokens), re.IGNORECASE | re.DOTALL)
        return pattern.search(report_markdown)

    for candidate in (sentence, cleaned_sentence):
        if not candidate:
            continue
        match = regex_find(candidate)
        if match:
            return wrap_span(match.start(), match.end())

    # 4) Fallback: prefix snippet of cleaned sentence
    if cleaned_sentence:
        words = cleaned_sentence.split()
        if len(words) >= 6:
            prefix = " ".join(words[: min(14, len(words))])
            match = regex_find(prefix)
            if match:
                return wrap_span(match.start(), match.end())

    return report_markdown


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/list_reports", methods=["GET"])
def list_reports():
    try:
        directory, current_dir = resolve_data_directory(request.args.get("dir", ""))

        if not directory.exists() or not directory.is_dir():
            return jsonify({"success": False, "error": "Directory not found"}), 404

        directories: list[str] = []
        reports: list[dict[str, str]] = []

        for item in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file() and item.suffix.lower() == ".json":
                report_rel = f"{current_dir}/{item.stem}" if current_dir else item.stem
                reports.append({"name": item.stem, "path": report_rel})

        return jsonify(
            {
                "success": True,
                "current_dir": current_dir,
                "directories": directories,
                "reports": reports,
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/load_report", methods=["GET"])
def load_report():
    try:
        file_path, report_rel = resolve_report_file(request.args.get("report", ""))

        if not file_path.exists():
            return jsonify({"success": False, "error": "Report file not found"}), 404

        normalized, errors = load_and_validate_report(file_path)
        if errors:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid report schema",
                        "details": errors,
                    }
                ),
                400,
            )

        return jsonify(
            {
                "success": True,
                "report_path": report_rel,
                "report": normalized,
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/save_annotation", methods=["POST"])
def save_annotation():
    try:
        file_path, _ = resolve_report_file(request.args.get("report", ""))
        payload = request.get_json(silent=True)

        normalized, errors = validate_report_schema(payload)
        if errors:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid report schema",
                        "details": errors,
                    }
                ),
                400,
            )

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(normalized, fh, indent=2, ensure_ascii=False)

        return jsonify({"success": True})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    try:
        payload: Any = None
        source_filename = "report.json"
        current_report_path = ""
        username = "anonymous"

        if request.files.get("file") is not None:
            upload = request.files["file"]
            if not upload.filename:
                return jsonify({"success": False, "error": "No selected file"}), 400
            if not allowed_file(upload.filename):
                return jsonify({"success": False, "error": "Invalid file type"}), 400

            source_filename = upload.filename
            current_report_path = request.form.get("current_report_path", "")
            username = request.form.get("username", "anonymous") or "anonymous"

            try:
                payload = json.loads(upload.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return jsonify({"success": False, "error": "Uploaded file is not valid JSON"}), 400
        else:
            body = request.get_json(silent=True)
            if not body:
                return jsonify({"success": False, "error": "Missing upload payload"}), 400
            payload = body.get("data", body)
            source_filename = body.get("filename", "report.json")
            current_report_path = body.get("current_report_path", "")
            username = body.get("username", "anonymous") or "anonymous"

        normalized, errors = validate_report_schema(payload)
        if errors:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid report schema",
                        "details": errors,
                    }
                ),
                400,
            )

        upload_dir = resolve_upload_directory(current_report_path)
        safe_username = secure_filename(username.strip()) or "anonymous"
        user_dir = (upload_dir / f"submission_{safe_username}").resolve()
        ensure_within_root(user_dir, UPLOAD_ROOT)
        user_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(secure_filename(source_filename))[0] or "report"
        output_name = f"{base}_timestamp_{timestamp}.json"
        output_path = user_dir / output_name

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(normalized, fh, indent=2, ensure_ascii=False)

        upload_id = str(output_path.relative_to(UPLOAD_ROOT)).replace("\\", "/")
        return jsonify({"success": True, "upload_id": upload_id, "filename": output_name})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/view_full_report", methods=["GET"])
def view_full_report():
    try:
        file_path, report_rel = resolve_report_file(request.args.get("report", ""))

        if not file_path.exists():
            return "<h1>Report file not found</h1>", 404

        normalized, errors = load_and_validate_report(file_path)
        if errors:
            errors_html = "<br>".join(errors)
            return f"<h1>Invalid report schema</h1><p>{errors_html}</p>", 400

        full_report = normalized.get("response", "")
        if not full_report:
            return "<h1>No full report content available</h1>", 404

        sentence_index_raw = request.args.get("sentence", "0")
        try:
            sentence_index = int(sentence_index_raw)
        except ValueError:
            sentence_index = 0

        target_sentence = ""
        sentences_info = normalized.get("sentences_info", [])
        if 0 <= sentence_index < len(sentences_info):
            target_sentence = sentences_info[sentence_index].get("sentence", "")

        highlighted_markdown = highlight_sentence_in_markdown(full_report, target_sentence)
        html_content = markdown.markdown(
            highlighted_markdown,
            extensions=["extra", "codehilite", "toc"],
        )

        html_page = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Full Report - {report_rel}</title>
    <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 1.8em;
            margin-bottom: 0.8em;
            color: #2c3e50;
        }}
        p {{ margin-bottom: 1.2em; }}
        code {{ background-color: #f8f9fa; padding: 0.2em 0.4em; border-radius: 3px; }}
        pre {{ background-color: #f8f9fa; padding: 1em; border-radius: 5px; overflow-x: auto; border: 1px solid #e9ecef; }}
        .highlight-info {{
            background-color: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
        }}
        mark {{
            background-color: yellow !important;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class=\"highlight-info\">
        <strong>Highlighted sentence:</strong> Sentence #{sentence_index + 1}
    </div>
    <div class=\"content\">{html_content}</div>
    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            const highlighted = document.querySelector('mark');
            if (highlighted) {{
                highlighted.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }});
    </script>
</body>
</html>
"""
        return html_page
    except ValueError as exc:
        return f"<h1>Invalid request</h1><p>{exc}</p>", 400
    except Exception as exc:  # pragma: no cover
        return f"<h1>Error loading report</h1><p>{exc}</p>", 500


def find_open_port(start: int = 5000, max_tries: int = 50) -> int:
    port = start
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                port += 1
    raise RuntimeError("No free port found in range")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep research report visualization tool")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for source report JSON files (default: %(default)s)",
    )
    args = parser.parse_args()
    set_data_root(args.data_dir)

    port = 20001
    app.run(debug=True, host="0.0.0.0", port=port)
