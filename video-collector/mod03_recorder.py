import cv2
import numpy as np
import mediapipe as mp
from mod01_config import (
    VIDEO_DEVICE, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    FRAMES_PER_REP, COUNTDOWN_SECONDS,
    MP_CONFIDENCE, POSE_REMOVE_IDX, POSE_KEEP_CONNECTIONS
)
from mod02_storage import ensure_folders, path_videos, path_landmarks


# MediaPipe initializations
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=MP_CONFIDENCE,
    min_tracking_confidence=MP_CONFIDENCE,
    static_image_mode=False
)

# Landmark detection (no drawing just yet)
def detect_landmarks(frame, black_bg=False):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = holistic.process(frame_rgb)
    frame_rgb.flags.writeable = True
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # zero visibility of unwanted pose points (but will still be saved into .npy)
    if results.pose_landmarks:
        for i in POSE_REMOVE_IDX:
            results.pose_landmarks.landmark[i].visibility = 0.0

    if black_bg:
        h, w = frame.shape[:2]
        frame = np.zeros((h, w, 3), dtype=np.uint8)

    return frame, results


# Draw everything (face, pose, right and left hands)
def draw_landmarks(frame, results):
    # Face tessellation
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(0,244,0), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=1)
        )
    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=8, circle_radius=2),
            mp_drawing.DrawingSpec(color=(220,0,0), thickness=8)
        )
    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,0,220), thickness=8, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=8)
        )
    # Pose (only upper-body)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, POSE_KEEP_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=17, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=17)
        )
    return frame


# Extract flattened vector 
def extract_vector(results):
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(468*3)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21*3)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, pose, rh, lh])


# ONE REPETITION 
def record_one_repetition(cap, signer_id: str, sign: str, rep_idx: int):
    ensure_folders(signer_id, sign)

    rep_dir = path_videos(signer_id, sign) / f"rep-{rep_idx}"
    lm_dir  = path_landmarks(signer_id, sign) / f"rep-{rep_idx}"
    rep_dir.mkdir(exist_ok=True)
    lm_dir.mkdir(exist_ok=True)

    video_path = rep_dir / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    # SMOOTH COUNTDOWN 
    import time
    start_time = time.time()
    elapsed = 0
    while elapsed < COUNTDOWN_SECONDS:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera failed during countdown")
        frame = cv2.flip(frame, 1)
        secs_left = int(COUNTDOWN_SECONDS - elapsed) + 1
        cv2.putText(frame, f"Start in {secs_left}", (400, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 6)
        cv2.putText(frame, f"Signer: {signer_id} | Sign: {sign} | Rep: {rep_idx+1}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Recorder", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("User abort during countdown")
        elapsed = time.time() - start_time


    # record 32 frames 
    for f_idx in range(FRAMES_PER_REP):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera lost during recording")

        # raw frame -> video file
        out.write(frame)

        # landmarks
        frame_vis, results = detect_landmarks(frame, black_bg=False)
        frame_vis = draw_landmarks(frame_vis, results)
        frame_vis = cv2.flip(frame_vis, 1)

        # on-screen info
        cv2.putText(frame_vis, f"Rep {rep_idx+1} Frame {f_idx+1}/{FRAMES_PER_REP}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Recorder", frame_vis)

        # save .npy
        vec = extract_vector(results)
        np.save(lm_dir / f"frame-{f_idx:02d}.npy", vec)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            out.release()
            raise KeyboardInterrupt("User abort")

    out.release()
    cv2.destroyWindow("Recorder")
    return rep_idx + 1

