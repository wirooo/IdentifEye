# Identifeye
Trains a model to perform facial recognition and labelling on your friends. 
Uses Google Photos API to download images.
Classification model developed using Pytorch using transfer learning on a pretrained CNN and facial recognition using facenet-pytorch.
Contains a GUI based tool developed with PySimpleGUI to help labelling images.

### Requirements
 - Working & compatible versions of Python, torch. Developed with `pip install torch==1.7.1+cu102 torchvision==0.8.2+cu102 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
 - Properly configured CUDA machine for use in Pytorch. Guide [here](https://medium.datadriveninvestor.com/installing-pytorch-and-tensorflow-with-cuda-enabled-gpu-f747e6924779). Developed with Cuda 10.2
 - `pip install -r requirements.txt`
 - A camera connected to your machine. Accessed by OpenCV as `VideoCapture(0)`
 - Google Developer enabled account: See [here](https://developers.google.com/)
 
### Usage
#### Google PhotosAPI
 1. Go to [Google API Console](https://console.developers.google.com/apis/library),  create a new project and enable "Google Photos Library API"
 2. Create OAuth 2 Credentials from 'APIs & Services' > 'Credentials' > 'OAuth client ID' > 'Desktop App'. Download the file as "credentials.json" and copy into `src/`
 3. Download photos from google photos with. Default folder will be `../photos/faces/`. Automatically filters for categories "selfies" and "people". 
 ```
 cd src
 gphotos.py [--folder [FOLDER]] [--creds [CREDS]]
 ```
#### Processing Faces From Photos
 1. If you have pre-downloaded photos to use, to retrieve all relevant faces from `photo_dir` and save to `save_dir`:
 ```
 get_faces.py [--photo_dir [PHOTO_DIR]] [--save_dir [SAVE_DIR]]
 ```
#### Labelling Faces
 1.  Launch a quick gui to label each face and save into correspondingly labelled folder. Names is a list of space separated labels for the faces you want to use:
 ```
 image_labeller.py  [--photo_dir [PHOTO_DIR]] [--label_dir [LABEL_DIR]] names [names ...]
 ```
#### Record New Data From Video
 1. Launch a camera recording and label as `name` into `folder` with:
 ```
 record_face.py [--frames FRAMES] [--fps FPS] folder name
 ```
#### Train and Run Facial Recognition
 1. By default, `image_labeller.py` puts labelled images into `../photos/labelled` correctly formatted by label. Run notebook `Training.ipynb` to train on the labelled data. Outputs created in `../model_save/`
 2. To launch application, run `face_detector.py`. Press `q` or `Ctrl+C` to quit. 
 
 
