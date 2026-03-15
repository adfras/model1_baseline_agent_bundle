from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import requests


DATASET_DOI = "10.26193/6DZWOH"
LANDING_URL = f"https://dataverse.ada.edu.au/citation?persistentId=doi:{DATASET_DOI}"
CHALLENGE_PATH = "https://dataverse.ada.edu.au/.within.website/x/cmd/anubis/api/pass-challenge"
DATASET_API_URL = (
    f"https://dataverse.ada.edu.au/api/datasets/:persistentId/?persistentId=doi:{DATASET_DOI}"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)


def extract_json_block(html: str, element_id: str) -> dict:
    pattern = rf'<script id="{re.escape(element_id)}" type="application/json">(.*?)</script>'
    match = re.search(pattern, html, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find JSON block for {element_id!r}")
    return json.loads(match.group(1))


def solve_pow(random_data: str, difficulty: int) -> tuple[str, int, int]:
    target_prefix = "0" * difficulty
    nonce = 0
    started = time.perf_counter()
    while True:
        digest = hashlib.sha256(f"{random_data}{nonce}".encode("utf-8")).hexdigest()
        if digest.startswith(target_prefix):
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return digest, nonce, elapsed_ms
        nonce += 1


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def complete_challenge(session: requests.Session) -> None:
    response = session.get(LANDING_URL, timeout=120)
    response.raise_for_status()
    if "anubis_challenge" not in response.text:
        return

    payload = extract_json_block(response.text, "anubis_challenge")
    challenge = payload["challenge"]
    rules = payload["rules"]
    digest, nonce, elapsed_ms = solve_pow(challenge["randomData"], int(rules["difficulty"]))

    params = {
        "id": challenge["id"],
        "response": digest,
        "nonce": nonce,
        "redir": LANDING_URL,
        "elapsedTime": elapsed_ms,
    }
    pass_response = session.get(CHALLENGE_PATH, params=params, timeout=120, allow_redirects=True)
    pass_response.raise_for_status()
    if "anubis_challenge" in pass_response.text:
        raise RuntimeError("Challenge was not cleared successfully")


def fetch_dataset_manifest(session: requests.Session) -> dict:
    response = session.get(DATASET_API_URL, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "OK":
        raise RuntimeError(f"Unexpected dataset API response: {payload}")
    return payload["data"]["latestVersion"]


def download_file(session: requests.Session, file_id: int, destination: Path) -> None:
    url = f"https://dataverse.ada.edu.au/api/access/datafile/{file_id}"
    with session.get(url, timeout=300, stream=True) as response:
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download the DBE-KT22 dataset from ADA Dataverse.")
    parser.add_argument(
        "--outdir",
        default=Path("data/raw/DBE-KT22"),
        type=Path,
        help="Directory to write the dataset files into.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Print the file manifest after authentication without downloading files.",
    )
    args = parser.parse_args(argv)

    session = build_session()
    complete_challenge(session)
    manifest = fetch_dataset_manifest(session)

    files = manifest.get("files", [])
    if not files:
        raise RuntimeError("No files were returned by the dataset manifest")

    print(f"Resolved {len(files)} files for doi:{DATASET_DOI}")
    for entry in files:
        data_file = entry["dataFile"]
        name = data_file["filename"]
        size = data_file.get("filesize")
        file_id = data_file["id"]
        print(f"- id={file_id} size={size} name={name}")

    if args.manifest_only:
        return 0

    for entry in files:
        data_file = entry["dataFile"]
        destination = args.outdir / data_file["filename"]
        print(f"Downloading {data_file['filename']} -> {destination}")
        download_file(session, int(data_file["id"]), destination)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
