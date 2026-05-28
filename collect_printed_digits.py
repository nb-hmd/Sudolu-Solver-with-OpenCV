import argparse
import os
import time

import cv2
import numpy as np

import image_processor as ip
from digit_recognizer import DigitRecognizer


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _key_to_digit(key):
    if ord("0") <= key <= ord("9"):
        return int(chr(key))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out-dir", type=str, default=os.path.join("data", "printed_digits"))
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    for d in range(10):
        _ensure_dir(os.path.join(args.out_dir, str(d)))

    recognizer = DigitRecognizer()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Could not open webcam (index {args.camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Point webcam at a Sudoku puzzle.")
    print("Press 'c' to capture the grid and label digits.")
    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        display = frame.copy()
        processed = ip.pre_process_image(frame)
        biggest, max_area = ip.find_contours(processed, display)
        if biggest.size != 0:
            biggest = ip.reorder(biggest)
            pts = np.array(biggest, dtype=np.int32).reshape(4, 2)
            draw_pts = np.array([pts[0], pts[1], pts[3], pts[2]], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [draw_pts], True, (0, 255, 0), 2)

        cv2.imshow("Collector", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key != ord("c"):
            continue

        if biggest is None or biggest.size == 0:
            print("No grid detected. Try again.")
            continue

        warped = ip.get_warp(frame, biggest, 450, 450)
        warped_bin = ip.pre_process_grid_for_ocr(warped)
        cells = ip.split_boxes(warped_bin)

        candidates = []
        for idx, cell in enumerate(cells):
            ink = float(np.count_nonzero(cell)) / float(cell.size)
            if ink > 0.01:
                candidates.append(idx)

        if not candidates:
            print("No digit candidates found. Try better lighting/angle.")
            continue

        ts = time.strftime("%Y%m%d_%H%M%S")
        print(f"Captured. Labeling {len(candidates)} cells (press 0-9, 0=skip, q=stop labeling).")

        for idx in candidates:
            cell = cells[idx]
            view = cv2.resize(cell, (224, 224), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Cell", view)

            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord("q"):
                    break

                digit = _key_to_digit(k)
                if digit is None:
                    continue
                if digit == 0:
                    break

                processed_cell = recognizer.pre_process_cell(cell)
                img28 = (processed_cell[0, :, :, 0] * 255.0).astype(np.uint8)
                out_path = os.path.join(args.out_dir, str(digit), f"{ts}_cell{idx:02d}.png")
                cv2.imwrite(out_path, img28)
                break

            if k == ord("q"):
                break

        cv2.destroyWindow("Cell")
        print("Done labeling this capture.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

