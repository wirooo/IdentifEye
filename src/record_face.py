import cv2
from facenet_pytorch import MTCNN
import argparse

class FaceRecorder():

    def __init__(self, mtcnn, img_folder, label):
        self.mtcnn = mtcnn
        self.img_folder = img_folder
        self.label = label
        print(img_folder, label)

    def draw(self, frame, box, prob):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=1)
        cv2.putText(frame, f"face: {prob*100:.1f}%", (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
        return frame

    def record(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                boxes, face_probs = mtcnn.detect(frame)
                if boxes is not None and len(boxes) > 0:
                    # for box in boxes:
                    #     y1, y2, x1, x2 = int(box[1]), int(box[3]), int(box[0]), int(box[2])
                    #     face = frame[y1:y2, x1:x2]
                    #     pred, probs = self.classify_face(face)
                    #     name_probs.append(probs)
                    self.draw(frame, boxes[0], face_probs[0])
                else:
                    cv2.putText(frame, "Couldn't Find Any Faces", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, nargs=1)
    parser.add_argument("name", type=str, nargs=1)
    args = parser.parse_args()
    mtcnn = MTCNN()
    face_recorder = FaceRecorder(mtcnn, args.folder, args.name)
    face_recorder.record()
