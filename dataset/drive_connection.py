import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive"]

def auth_drive():
  creds = None 
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)

  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    service = build("drive", "v3", credentials=creds)

    return service
  
  except HttpError as error:
    print(f"An error occurred: {error}")

def list_dataset(service):
    FOLDER_ID = "1nJm3E6XnYVVFz2itBBdC-qtSab6GZmLO"
    query = f"'{FOLDER_ID}' in parents"

    results = service.files().list(
       q=query, 
       pageSize=10, 
       fields="files(id, name)").execute()
    
    items = results.get("files", [])
    if not items:
        print("No files found.")
        return
    print("Files:")
    for item in items:
        print(f"{item['name']} ({item['id']})")

if __name__ == "__main__":
    service = auth_drive()
    list_dataset(service) 