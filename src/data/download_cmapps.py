from pathlib import Path
from urllib.request import urlretrieve
import zipfile

CMAPSS_URL = "https://ti.arc.nasa.gov/c/6/"  # NASA prognostics data repository (C-MAPSS zip)
ZIP_NAME = "CMAPSSData.zip"

def download_cmapps(data_dir: str = "data") -> Path:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    zip_path = data_path / ZIP_NAME

    # Download if missing
    if not zip_path.exists():
        print(f"Downloading C-MAPSS dataset from: {CMAPSS_URL}")
        urlretrieve(CMAPSS_URL, zip_path)

    # Extract
    extracted_flag = data_path / ".extracted"
    if not extracted_flag.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_path)
        extracted_flag.write_text("ok")

    # Expected: data/train_FD001.txt etc.
    print("Dataset ready.")
    return data_path

if __name__ == "__main__":
    download_cmapps()
