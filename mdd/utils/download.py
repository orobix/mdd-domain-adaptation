import os
import shutil
import tarfile
from urllib.parse import parse_qs, urlparse

import requests
from tqdm import tqdm


def get_google_drive_stream(id):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    response = None
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    return response


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def download_with_resume(url, destination, dataset):
    # Check if the requested url is ok, i.e. 200 <= status_code < 400
    head = requests.head(url)
    if not head.ok:
        head.raise_for_status()

    # Don't download if the file exists
    if os.path.exists(os.path.expanduser(destination)):
        print(destination)
        print("File already exists, no need to download")
        return
    else:
        os.makedirs(os.path.dirname(destination), exist_ok=True)

    tmp_file = destination + ".part"
    if url.startswith("https://drive.google.com/"):
        first_byte = 0
    else:
        first_byte = (
            os.path.getsize(tmp_file) if os.path.exists(tmp_file) else 0
        )
    chunk_size = 1024 ** 2  # 1 MB
    file_mode = "ab" if first_byte else "wb"

    # Set headers to resume download from where we've left
    headers = {"Range": "bytes=%s-" % first_byte}

    # Since requests doesn't support local file reading
    # we check if protocol is file://
    if url.startswith("file://"):
        url_no_protocol = url.replace("file://", "", count=1)
        if os.path.exists(url_no_protocol):
            print("File already exists, no need to download")
            return
        else:
            raise Exception("File not found at %s" % url_no_protocol)
    elif url.startswith("https://drive.google.com/"):
        parsed_url = urlparse(url)
        id = parse_qs(parsed_url.query)["id"]
        r = get_google_drive_stream(id)
        if r is None:
            print("Something wrong getting Google Drive information")
            return
    else:
        r = requests.get(url, headers=headers, stream=True)

    file_size = int(r.headers.get("Content-Length", -1))
    if file_size >= 0:
        # Content-length set
        file_size += first_byte
        total = file_size
    else:
        # Content-length not set
        print("Cannot retrieve Content-length from server")
        total = None

    print("Download from " + url)
    print("Starting download at %.1fMB" % (first_byte / (10 ** 6)))
    print("File size is %.1fMB" % (file_size / (10 ** 6)))

    with tqdm(initial=first_byte, total=total, unit_scale=True) as pbar:
        with open(tmp_file, file_mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Rename the temp download file to the correct name if fully downloaded
    shutil.move(tmp_file, destination)

    # Extract files
    if dataset == "office-31":
        extract_to = os.path.join(os.path.dirname(destination), "office-31")
    elif dataset == "image-clef":
        extract_to = os.path.dirname(destination)
    tar = tarfile.open(destination, "r:gz")
    tar.extractall(path=extract_to)
    tar.close()


datasets = {
    "office-31": {
        "url": "https://drive.google.com/u/0/uc?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE&export=download",
        "dest": "./data/office.tar.gz",
    },
    "image-clef": {
        "url": "https://drive.google.com/u/0/uc?id=0B9kJH0-rJ2uRS3JILThaQXJhQlk&export=download",
        "dest": "./data/image-clef.tar.gz",
    },
}
