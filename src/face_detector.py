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
        self.class_names = list(map(lambda str: str[0].upper() + str[1:].lower(), class_names))

    def draw(self, frame, boxes, face_probs, name_probs):
        for box, face_prob, name_prob in zip(boxes, face_probs, name_probs):
            name_preds = sorted([(self.class_names[i], name_prob[i]) for i in range(len(name_prob))], key=lambda x: x[1], reverse=True)
            if len(name_preds) > 3:
                name_preds = name_preds[:4]

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=1)
            cv2.putText(frame, f"face: {face_prob*100:.1f}%", (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)

            for i, (name, pred) in enumerate(name_preds):
                cv2.putText(frame, f"{name}: {pred:.3f}", (box[2], int(box[1] + i*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 - (0.1*i), (0, 0, 255), 1, cv2.LINE_AA)

        return frame

    def classify_face(self, face):
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
        out = np.array(out[0].cpu())
        return pred, out

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                boxes, face_probs = self.mtcnn.detect(frame)
                if boxes is not None and len(boxes) > 0:
                    name_probs = []
                    for box in boxes:
                        y1, y2, x1, x2 = int(box[1]), int(box[3]), int(box[0]), int(box[2])
                        face = frame[y1:y2, x1:x2]
                        pred, probs = self.classify_face(face)
                        name_probs.append(probs)

                    self.draw(frame, boxes, face_probs, name_probs)
                else:
                    cv2.putText(frame, "Couldn't Find Any Faces", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 1, cv2.LINE_AA)
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
