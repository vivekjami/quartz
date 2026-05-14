import argparse
import gzip
import urllib.request
from pathlib import Path

BASE_URL = "https://data.commoncrawl.org"

def get_wet_paths(crawl: str, n_files: int) -> list[str]:
    """Fetch the WET paths file for a crawl and return the first n_files paths."""
    paths_url = f"{BASE_URL}/crawl-data/{crawl}/wet.paths.gz"
    print(f"Fetching paths list: {paths_url}")
    with urllib.request.urlopen(paths_url) as resp:
        with gzip.open(resp) as f:
            lines = f.read().decode().strip().splitlines()
    return lines[:n_files]

def download_wet_file(path: str, dest_dir: Path) -> Path:
    url = f"{BASE_URL}/{path}"
    filename = path.split("/")[-1]
    dest = dest_dir / filename
    if dest.exists():
        print(f"  Skipping (exists): {filename}")
        return dest
    print(f"  Downloading: {filename} ...")
    urllib.request.urlretrieve(url, dest)
    return dest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl", default="CC-MAIN-2025-13")
    parser.add_argument("--n-files", type=int, default=5)
    parser.add_argument("--dest", default="data/wet/")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    paths = get_wet_paths(args.crawl, args.n_files)
    print(f"Downloading {len(paths)} WET files from {args.crawl}")
    for path in paths:
        download_wet_file(path, dest)
    print("Done.")

if __name__ == "__main__":
    main()
