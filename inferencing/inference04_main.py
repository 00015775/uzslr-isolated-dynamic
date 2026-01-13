import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
import time

from inference01_config import (
    MODEL_PATH, MAX_LEN, CHANNELS, NUM_CLASSES, MODEL_DIM,
    VIDEO_DEVICE, FRAME_WIDTH, FRAME_HEIGHT, MP_CONFIDENCE,
    BUFFER_SIZE, HANDS_REQUIRED, HAND_DISAPPEAR_TOLERANCE,
    DEFAULT_SIGNS, get_device
)
from inference02_preprocess import Preprocess
from inference03_model import SignLanguageModel


class SignRecognizer:
    def __init__(self):
        self.device = get_device()
        print(f"using device: {self.device}")
        
        # load model
        self.model = SignLanguageModel(
            max_len=MAX_LEN,
            channels=CHANNELS,
            num_classes=NUM_CLASSES,
            dim=MODEL_DIM
        )
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        print(f"model loaded from {MODEL_PATH}")
        
        # preprocessing
        self.preprocess = Preprocess(max_len=MAX_LEN).to(self.device)
        
        # mediapipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=MP_CONFIDENCE,
            min_tracking_confidence=MP_CONFIDENCE,
            static_image_mode=False
        )
        
        # frame buffer
        self.frame_buffer = deque(maxlen=BUFFER_SIZE)
        
        # hand tracking state
        self.both_hands_visible = False
        self.hand_disappear_counter = 0
        self.inference_active = False
        
        # prediction state
        self.current_prediction = None
        self.prediction_confidence = 0.0
        self.last_inference_time = 0
        
    def extract_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        # extract vector (1662,)
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468*3)
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21*3)
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21*3)
        
        vec = np.concatenate([face, pose, rh, lh])
        
        # check hand visibility
        has_both_hands = (results.right_hand_landmarks is not None and 
                         results.left_hand_landmarks is not None)
        
        return vec, has_both_hands, results
    
    def update_hand_state(self, has_both_hands):
        if has_both_hands:
            self.both_hands_visible = True
            self.hand_disappear_counter = 0
            if not self.inference_active:
                self.inference_active = True
                print("inference started - both hands detected")
        else:
            if self.both_hands_visible:
                self.hand_disappear_counter += 1
                if self.hand_disappear_counter > HAND_DISAPPEAR_TOLERANCE:
                    self.both_hands_visible = False
                    self.inference_active = False
                    self.frame_buffer.clear()
                    print("inference stopped - hands lost")
    
    def predict(self):
        if len(self.frame_buffer) < BUFFER_SIZE:
            return None, 0.0
        
        # stack frames
        frames = np.stack(list(self.frame_buffer)).astype(np.float32)  # (32, 1662)
        x = torch.from_numpy(frames).to(self.device)
        
        # preprocess
        x = self.preprocess(x)  # (32, 708)
        x = x.unsqueeze(0)  # (1, 32, 708)
        
        # create mask (all frames valid)
        mask = torch.ones(1, MAX_LEN, dtype=torch.bool, device=self.device)
        
        # inference
        with torch.no_grad():
            logits = self.model(x, mask)  # (1, 50)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=-1)
        
        pred_sign = DEFAULT_SIGNS[pred_idx.item()]
        confidence_val = confidence.item()
        
        return pred_sign, confidence_val
    
    def draw_info(self, frame, results):
        h, w = frame.shape[:2]
        
        # draw prediction at top
        if self.current_prediction:
            text = f"{self.current_prediction} ({self.prediction_confidence:.2f})"
            cv2.putText(frame, text, (w//2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # status info
        status = "active" if self.inference_active else "waiting for both hands"
        color = (0, 255, 0) if self.inference_active else (0, 0, 255)
        cv2.putText(frame, f"status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # buffer info
        buffer_text = f"buffer: {len(self.frame_buffer)}/{BUFFER_SIZE}"
        cv2.putText(frame, buffer_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # draw landmarks
        mp_drawing = mp.solutions.drawing_utils
        
        # draw face landmarks (eyes, lips, nose)
        if results.face_landmarks:
            # draw all face landmarks in light color
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, 
                self.mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1)
            )
            
            # draw contours (eyes, lips) in more visible color
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
        
        # draw hands
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(220,0,0), thickness=2)
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,220), thickness=2)
            )
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(VIDEO_DEVICE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("cannot open camera")
            return
        
        print("starting camera... press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # extract landmarks
            vec, has_both_hands, results = self.extract_landmarks(frame)
            
            # update hand tracking state
            self.update_hand_state(has_both_hands)
            
            # add to buffer if active
            if self.inference_active:
                self.frame_buffer.append(vec)
                
                # predict every 0.5 seconds
                current_time = time.time()
                if current_time - self.last_inference_time > 0.5:
                    pred, conf = self.predict()
                    if pred:
                        self.current_prediction = pred
                        self.prediction_confidence = conf
                        self.last_inference_time = current_time
            
            # draw info
            frame = self.draw_info(frame, results)
            
            cv2.imshow('sign recognition', frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyWindow('sign recognition')  
        cv2.waitKey(1)  
        cv2.destroyAllWindows()
        cv2.waitKey(1)  
        self.holistic.close()


def main():
    recognizer = SignRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()