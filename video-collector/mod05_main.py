
import cv2
from mod01_config import VIDEO_DEVICE, FRAME_WIDTH, FRAME_HEIGHT, FPS
from mod04_ui import select_signer, select_sign, after_recording_menu
from mod03_recorder import record_one_repetition
from mod02_storage import count_repetitions


def main():
    # CAMERA
    cap = cv2.VideoCapture(VIDEO_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("Cannot open camera. Check VIDEO_DEVICE in config.py")
        return

    # STAGE 1: Signer 
    signer_id = select_signer()

    # MAIN LOOP
    while True:
        # STAGE 2: Sign word 
        chosen_sign = select_sign(signer_id)
        if chosen_sign is None:               # user pressed "back"
            signer_id = select_signer()       # start over with a (current or new) signer
            continue

        # STAGE 3: Recording
        rep_idx = count_repetitions(signer_id, chosen_sign)
        while True:
            print(f"\nPress **s** in the camera window to start repetition {rep_idx+1} of **{chosen_sign}**")
            print("(you can quit the whole program with Ctrl+C)")

            # wait for 's'
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "Press 's' to start", (300, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
                cv2.putText(frame, f"Signer: {signer_id} | Sign: {chosen_sign} | Next rep: {rep_idx+1}",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Recorder – press s", frame)
                k = cv2.waitKey(10) & 0xFF
                if k == ord('s'):
                    cv2.destroyWindow("Recorder – press s")
                    break
                if k == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # record one repetition
            try:
                rep_idx = record_one_repetition(cap, signer_id, chosen_sign, rep_idx)
            except KeyboardInterrupt:
                print("\nRecording aborted by user.")
                break
            except Exception as e:
                print(f"\nError during recording: {e}")
                break

            # after-recording menu 
            action = after_recording_menu(signer_id, chosen_sign, rep_idx-1)
            if action == "done":
                break          # back to sign list
            # else "again" -> loop continues, rep_idx already incremented

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


