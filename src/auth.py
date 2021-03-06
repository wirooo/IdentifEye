import pickle
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


def authenticate(SCOPES, credentials_file):
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
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    if creds.valid:
        print("Authentication Complete!")
    return creds

