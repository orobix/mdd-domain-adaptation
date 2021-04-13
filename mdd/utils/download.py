import os
import shutil
import tarfile
import zipfile
from urllib.parse import parse_qs, urlparse

import requests
from mdd import DOWNLOAD_DIR
from tqdm import tqdm

datasets = {
    "office-31": {
        "url": "https://drive.google.com/u/1/uc?id=1KgFgIXL5yHeH9lwX6yasrC6czqTg3fcB&export=download",
        "dest": os.path.join(DOWNLOAD_DIR, "office-31.tar.gz"),
    },
    "image-clef": {
        "url": "https://drive.google.com/u/1/uc?id=1mZpsASvJLzZS50t3MxlXbvNB22s3QsZ7&export=download",
        "dest": os.path.join(DOWNLOAD_DIR, "image-clef.tar.gz"),
    },
    "office-home": {
        "url": "https://drive.google.com/u/1/uc?id=10yYOpN155ocIQ7LMtsFgzQoLz0IASPSZ&export=download",
        "dest": os.path.join(DOWNLOAD_DIR, "office-home.tar.gz"),
    },
}


def get_google_drive_stream(id, first_byte):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(
        URL,
        params={"id": id},
        stream=True,
        headers={"Range": "bytes=%s-" % first_byte},
    )
    token = get_confirm_token(response)

    response = None
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(
            URL,
            params=params,
            stream=True,
            headers={"Range": "bytes=%s-" % first_byte},
        )
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
        print("File already exists, no need to download")
    elif url.startswith("file://"):
        # Since requests doesn't support local file reading
        # we check if protocol is file://
        url_no_protocol = url.replace("file://", "", count=1)
        if os.path.exists(url_no_protocol):
            print("File already exists, no need to download")
        else:
            raise Exception("File not found at %s" % url_no_protocol)
    else:
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Create temp file to retrieve download if necessary
        tmp_file = destination + ".part"
        first_byte = os.path.getsize(tmp_file) if os.path.exists(tmp_file) else 0
        chunk_size = 1024 ** 2  # 1 MB
        file_mode = "ab" if first_byte else "wb"

        if url.startswith("https://drive.google.com/"):
            parsed_url = urlparse(url)
            id = parse_qs(parsed_url.query)["id"]
            r = get_google_drive_stream(id, first_byte)
            if r is None:
                print("Something wrong getting Google Drive information")
                return
            cr = r.headers.get("Content-Range", -1)
            file_size = int(cr.partition("/")[-1]) if cr != -1 else cr
        else:
            # Set headers to resume download from where we've left
            headers = {"Range": "bytes=%s-" % first_byte}
            r = requests.get(url, headers=headers, stream=True)
            file_size = int(r.headers.get("Content-Length", -1))

        if file_size < 0:
            # Content-length not set
            print("Cannot retrieve Content-length from server")
            file_size = None

        print("Download from " + url)
        print("Starting download at %.1fMB" % (first_byte / (10 ** 6)))
        print("File size is %.1fMB" % (file_size / (10 ** 6)))

        with tqdm(initial=first_byte, total=file_size, unit_scale=True) as pbar:
            with open(tmp_file, file_mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Rename the temp download file to the correct name if fully downloaded
        shutil.move(tmp_file, destination)

    print("Extracting {} to {}".format(destination, DOWNLOAD_DIR))
    file_ext = os.path.splitext(destination)[1]
    if file_ext == ".gz" or file_ext == ".tar":
        tar = tarfile.open(destination, "r|*")
        tar.extractall(path=DOWNLOAD_DIR)
        tar.close()
    else:
        with zipfile.ZipFile(destination, "r") as f:
            f.extractall(DOWNLOAD_DIR)

    print("Remove {}".format(destination))
    os.remove(destination)
