import get_faces
import pickle
import os
import io
import requests
from PIL import Image
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


class GPhotos:

    def __init__(self, scopes, creds_file_name, photo_dir):
        self.scopes = scopes
        self.creds = self.authenticate(scopes, creds_file_name)
        self.photo_dir = photo_dir

    def authenticate(self, SCOPES, credentials_file):
        """
        Authenticates into google api.
        :param SCOPES: list of required scopes
        :param credentials_file: string name of credentials downloaded from google Oath consent screen
        :return: credentials object
        """
        creds = None

        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_file,
            scopes = SCOPES
        )
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        if creds.valid:
            print("Authentication Complete!")
        return creds

    def download_photos(self, dates=[((2017, 11, 0), (9999, 12, 0))], max_items=100):
        """
        Pulls and downloads photos from google photos API
        :param dates: list of date ranges to filter on in format [((yyyy,mm,dd), (yyyy,mm,dd)), ...], first is start,
            second is end
        :param max_items: max number of photos to download
        :return:
        """
        google_photos = build('photoslibrary', 'v1', credentials=self.creds)
        next_page_token = None
        items = []
        while next_page_token != '':
            print(f"Number of items processed:{len(items)}")
            if len(items) > max_items: break
            filters = {
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
                                "year": start[0],
                                "month": start[1],
                                "day": start[2]
                            },
                            "endDate": {
                                "year": end[0],
                                "month": end[1],
                                "day": end[2]
                            }
                        }
                        for start, end in dates
                    ]
                }
            }

            results = google_photos.mediaItems().search(
               body={
                    "filters": filters,
                    "pageSize":  100,
                    "pageToken": next_page_token
                }
            ).execute()

            items += results.get("mediaItems", [])
            next_page_token = results.get('nextPageToken', '')

        mtcnn = get_faces.create_mtcnn()
        for photo in items:
            self.download_file(photo['baseUrl'] + '=d', photo["filename"], mtcnn)

    def download_file(self, url, file_name, mtcnn):
        """
        Saves individual cropped faces from a given photo url
        :param url: url of photo to be downloaded
        :param file_name: name of photo to save
        :param mtcnn: mtcnn model from facenet-pytorch
        """
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Downloading file {file_name}")
            image = Image.open(io.BytesIO(response.content))
            get_faces.get_face(mtcnn, image, self.photo_dir, file_name)
        else:
            print(f"FAILED downloading {file_name}")


if __name__ == "__main__":
    SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
    gphotos = GPhotos(SCOPES, "credentials.json", "../photos/faces_dl")
    gphotos.download_photos()

