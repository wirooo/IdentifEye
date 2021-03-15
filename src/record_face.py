import cv2
from facenet_pytorch import MTCNN
import argparse
import os
import time


class FaceRecorder():

    def __init__(self, mtcnn, img_folder, label, max_frames, fps):
        self.mtcnn = mtcnn
        self.img_folder = img_folder
        self.label = label
        self.max_frames = max_frames
        self.fps = fps

    def draw(self, frame, box, prob):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=1)
        cv2.putText(frame, f"face: {prob*100:.1f}%", (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), 1, cv2.LINE_AA)
        return frame

    def record(self):
        cap = cv2.VideoCapture(0)

        prev_capture = 0
        frame_count = 0
        while frame_count < self.max_frames:
            ret, frame = cap.read()
            if time.time() - prev_capture > 1./self.fps:
                prev_capture = time.time()
                if ret:
                    boxes, face_probs = mtcnn.detect(frame)
                    if boxes is not None and len(boxes) > 0:
                        box = boxes[0]
                        y1, y2, x1, x2 = int(box[1]), int(box[3]), int(box[0]), int(box[2])
                        face = frame[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(self.img_folder, self.label, f"recorded_{frame_count}.png"), face)
                        cv2.putText(frame, f"Frame: {frame_count}/{self.max_frames}",
                                    (0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (0, 0, 255), 1, cv2.LINE_AA)
                        frame_count += 1
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
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    mtcnn = MTCNN()
    face_recorder = FaceRecorder(mtcnn, args.folder[0], args.name[0], args.frames, args.fps)
    face_recorder.record()

