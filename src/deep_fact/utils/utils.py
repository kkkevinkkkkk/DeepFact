import json
import pandas as pd
import os
from werkzeug.utils import secure_filename

def read_jsonl(file_path, return_df=True):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    if return_df:
        return pd.DataFrame(data)
    return data


def write_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')

        for item in data:
            file.write(json.dumps(item) + '\n')
    print(f"saved to {file_path}")


def find_latest_submission(report_path=None, username="yukun", data_folder='~/visualization/data'):
    home_dir = os.path.expanduser('~')
    if "~" in data_folder:
        data_folder = data_folder.replace("~", home_dir)
    UPLOAD_FOLDER = data_folder.replace('/data', '/data_uploads')

    report_dir = ''
    if report_path:
        report_dir = os.path.dirname(report_path)
    with open(os.path.join(data_folder, report_path), 'r') as f:
        data = json.load(f)
    # Construct the search directory
    search_dir = os.path.join(UPLOAD_FOLDER, report_dir) if report_dir else UPLOAD_FOLDER
    user_dir = os.path.join(search_dir, f"submission_{username}")

    # Get the base filename without path and extension, using secure_filename like during upload
    report_basename = os.path.basename(report_path)
    base_filename = os.path.splitext(secure_filename(report_basename))[0]  # Add .json then remove extension to match upload process
    # Find all matching submissions for this report by this user
    submissions = []


    try:
        for filename in os.listdir(user_dir):
            if filename.startswith(base_filename + '_timestamp_') and filename.endswith('.json'):
                file_path = os.path.join(user_dir, filename)
                stat = os.stat(file_path)

                # Extract timestamp from filename
                timestamp_part = filename.replace(base_filename + '_timestamp_', '').replace('.json', '')

                submissions.append({
                    'filename': filename,
                    # 'path': os.path.relpath(file_path, UPLOAD_FOLDER),
                    'path': file_path,
                    'timestamp': timestamp_part,
                    'modification_time': stat.st_mtime,
                    'username': username
                })

    except PermissionError:
        return None

    if len(submissions) == 0:
        return None

    # Sort by modification time (newest first)
    submissions.sort(key=lambda x: x['modification_time'], reverse=True)
    latest_submission = submissions[0]
    print(f"Load from {latest_submission['path']}")
    with open(latest_submission['path'], 'r') as f:
        new_data = json.load(f)

    original_sentences_info = data["sentences_info"]
    data["sentences_info"] = new_data["sentences_info"]
    for i, sentence_info, ori_sentence_info in zip(range(len(data["sentences_info"])), data["sentences_info"], original_sentences_info):
        for key in ori_sentence_info:
            if key not in sentence_info:
                sentence_info[key] = ori_sentence_info[key]
    return data