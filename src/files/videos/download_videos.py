import gdown, os

DOWNLOAD_FOLDER = 'files/videos'
IDS_FILE = 'files/videos/video_ids.txt'

with open(IDS_FILE, 'r') as f:
    model_ids = [line.strip() for line in f if line.strip()]

for model_id in model_ids:
    downloaded_file = gdown.download(id=model_id, quiet=False)
    os.replace(downloaded_file, os.path.join(DOWNLOAD_FOLDER, downloaded_file))