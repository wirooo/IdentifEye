import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


class FaceDetector():

    def __init__(self, mtcnn, clf, class_names, device):
        self.mtcnn = mtcnn
        self.clf = clf.to(device)
        self.device = device
        self.class_names = class_names

    def draw(self, frame, boxes, probs):
        for box, prob in zip(boxes, probs):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
            cv2.putText(frame, f"face: {prob}", (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            return frame

    def detect_ROIs(self, boxes):
        return [[int(box[x]) for x in [1, 3, 0, 2]] for box in boxes]


    def classify(self, face):
        destRGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(destRGB.astype('uint8'), 'RGB')

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )

        img = transform(img)
        batch_t = torch.unsqueeze(img, 0).to(self.device)
        with torch.no_grad():
            out = self.clf(batch_t)
            _, pred = torch.max(out, 1)

        pred = np.array(pred[0].cpu())
        return pred, self.class_names[pred]

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            boxes, probs = self.mtcnn.detect(frame)


            ROIs = self.detect_ROIs(boxes)
            for y1, y2, x1, x2 in ROIs:
                face = frame[y1:y2, x1:x2]
                pred, name = self.classify(face)
                print(name)

            self.draw(frame, boxes, probs)


            cv2.imshow("Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    class_names = np.load("../model_save/class_names.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    clf = models.resnet18(pretrained=True)
    in_ftrs = clf.fc.in_features
    clf.fc = nn.Sequential(nn.Linear(in_ftrs, len(class_names)), nn.Sigmoid())
    clf.load_state_dict(torch.load("../model_save/model.pth"))
    clf.eval()

    mtcnn = MTCNN()
    face_detector = FaceDetector(mtcnn, clf, class_names, device)

    face_detector.run()
