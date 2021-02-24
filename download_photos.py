import auth
from googleapiclient.discovery import build
import requests
from PIL import Image
import io
import get_faces


def download_file(url, destination_folder, file_name, mtcnn):
    """
    Saves individual cropped faces from a given photo url
    :param url: url of photo to be downloaded
    :param destination_folder: destination directory to save photos to
    :param file_name: name of photo to save
    :param mtcnn: mtcnn model from facenet-pytorch
    """
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Downloading file {file_name}")
        image = Image.open(io.BytesIO(response.content))
        get_faces.get_face(mtcnn, image, destination_folder, file_name)
    else:
        print(f"FAILED downloading {file_name}")


def download_photos(creds, photo_dir):
    """
    Pulls and downloads photos from google phtos API
    :param creds: authenticated credentials to google photos API
    :param photo_dir: destination directory to save photos to
    """
    google_photos = build('photoslibrary', 'v1', credentials=creds)
    next_page_token = None
    items = []
    while next_page_token != '':
        print(f"Number of items processed:{len(items)}")
        if len(items) > 50: break
        results = google_photos.mediaItems().search(
           body = {
                "filters": {
                    "contentFilter": {
                        "includedContentCategories": ["SELFIES", "PEOPLE"]
                    },
                    "mediaTypeFilter": {
                        "mediaTypes": ["PHOTO"]
                    },
                    "dateFilter": {
                        "ranges": [
                            {
                                "startDate": {
                                    "year": 2019,
                                    "month": 11,
                                    "day": 0
                                },
                                "endDate": {
                                    "year": 9999,
                                    "month": 12,
                                    "day": 0
                                }
                            },
                            {
                                "startDate": {
                                    "year": 2017,
                                    "month": 10,
                                    "day": 1
                                },
                                "endDate": {
                                    "year": 2017,
                                    "month": 11,
                                    "day": 1
                                }
                            }
                        ]
                    }
                },
                "pageSize":  100,
                "pageToken": next_page_token
            }
        ).execute()

        items += results.get("mediaItems", [])
        next_page_token = results.get('nextPageToken', '')

    mtcnn = get_faces.create_mtcnn()
    for photo in items:
        download_file(photo['baseUrl'] + '=d', photo_dir, photo["filename"], mtcnn)


if __name__ == "__main__":
    SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
    creds = auth.authenticate(SCOPES, "credentials.json")
    download_photos(creds, "photos/faces")

