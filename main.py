import cv2
import numpy as np
import image_processor as ip
import sudoku_solver as solver
from digit_recognizer import DigitRecognizer
import os
import argparse
import time
from collections import deque

def _solve_from_frame(img, recognizer, biggest=None):
    if biggest is None:
        img_processed = ip.pre_process_image(img)
        contours, _ = ip.find_contours(img_processed, img)
        if contours.size == 0:
            return None
        biggest = ip.reorder(contours)

    img_warped = ip.get_warp(img, biggest, 450, 450)
    img_warped_clean = ip.pre_process_grid_for_ocr(img_warped)

    boxes = ip.split_boxes(img_warped_clean)
    digits, confidences = recognizer.predict_cells_with_confidence(boxes)

    numbers = np.array(digits)
    board = numbers.reshape(9, 9).tolist()
    original_board = [row[:] for row in board]

    if not solver.solve_sudoku(board):
        return {
            "biggest": biggest,
            "board": board,
            "original_board": original_board,
            "digits": digits,
            "confidences": confidences,
            "solved": False,
            "img_warped": img_warped,
            "img_warped_clean": img_warped_clean,
        }

    solved_numbers = [item for sublist in board for item in sublist]
    flat_original = [item for sublist in original_board for item in sublist]
    display_sol = []
    for i in range(81):
        if flat_original[i] == 0:
            display_sol.append(solved_numbers[i])
        else:
            display_sol.append(0)

    return {
        "biggest": biggest,
        "board": board,
        "original_board": original_board,
        "digits": digits,
        "confidences": confidences,
        "solved": True,
        "solution_overlay": display_sol,
        "img_warped": img_warped,
        "img_warped_clean": img_warped_clean,
    }


def _overlay_solution(img, biggest, solved_numbers):
    img_solved = np.zeros((450, 450, 3), np.uint8)
    img_solved = ip.display_numbers(img_solved, solved_numbers)

    pts1 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    pts2 = np.float32(biggest)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_inv_warp = cv2.warpPerspective(img_solved, matrix, (img.shape[1], img.shape[0]))

    img_inv_warp_gray = cv2.cvtColor(img_inv_warp, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_inv_warp_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    img_fg = cv2.bitwise_and(img_inv_warp, img_inv_warp, mask=mask)
    return cv2.add(img_bg, img_fg)

def _draw_grid_outline(img, biggest, color=(0, 255, 0), thickness=2):
    pts = np.array(biggest, dtype=np.int32).reshape(4, 2)
    draw_pts = np.array([pts[0], pts[1], pts[3], pts[2]], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [draw_pts], True, color, thickness)

def _confidence_ok(digits, confidences, min_digit_conf):
    for d, c in zip(digits, confidences):
        if d != 0 and c < min_digit_conf:
            return False
    return True


def _count_conflicts(board):
    conflicts = 0
    for r in range(9):
        seen = {}
        for c in range(9):
            v = board[r][c]
            if v == 0:
                continue
            if v in seen:
                conflicts += 1
            else:
                seen[v] = c
    for c in range(9):
        seen = {}
        for r in range(9):
            v = board[r][c]
            if v == 0:
                continue
            if v in seen:
                conflicts += 1
            else:
                seen[v] = r
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            seen = set()
            for r in range(br, br + 3):
                for c in range(bc, bc + 3):
                    v = board[r][c]
                    if v == 0:
                        continue
                    if v in seen:
                        conflicts += 1
                    else:
                        seen.add(v)
    return conflicts


def _temporal_filter(history, min_digit_conf):
    if not history:
        return [0] * 81, [0.0] * 81

    window = len(history)
    required_votes = (window // 2) + 1

    out_digits = []
    out_confs = []
    for idx in range(81):
        votes = {}
        weight = {}
        for digits, confs in history:
            d = digits[idx]
            if d == 0:
                continue
            votes[d] = votes.get(d, 0) + 1
            weight[d] = weight.get(d, 0.0) + float(confs[idx])

        if not votes:
            out_digits.append(0)
            out_confs.append(0.0)
            continue

        best_digit = max(weight.items(), key=lambda kv: kv[1])[0]
        best_votes = votes[best_digit]
        avg_conf = weight[best_digit] / max(1, best_votes)

        if best_votes >= required_votes and avg_conf >= min_digit_conf:
            out_digits.append(int(best_digit))
            out_confs.append(float(avg_conf))
        else:
            out_digits.append(0)
            out_confs.append(float(avg_conf))

    return out_digits, out_confs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lock-frames", type=int, default=10)
    parser.add_argument("--solve-every", type=int, default=5)
    parser.add_argument("--stable-reads", type=int, default=2)
    parser.add_argument("--min-digit-conf", type=float, default=0.80)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--no-auto", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    auto_mode = not args.no_auto

    # Check if model exists
    if not os.path.exists('model.h5'):
        print("Model not found. Please run 'python train_model.py' first to generate the digit recognition model.")
        # We can continue but OCR won't work well
    
    recognizer = DigitRecognizer()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read image: {args.image}")
            return

        result = _solve_from_frame(img, recognizer)
        if result is None:
            print("No Sudoku grid detected.")
            return

        cv2.drawContours(img, [result["biggest"]], -1, (0, 255, 0), 2)
        if args.debug:
            dbg = cv2.cvtColor(result["img_warped_clean"], cv2.COLOR_GRAY2BGR)
            ip.display_numbers_with_confidence(dbg, result["digits"], result["confidences"])
            cv2.imshow("Warped Grid (Debug)", dbg)
        else:
            cv2.imshow("Warped Grid", result["img_warped_clean"])

        if result["solved"] and _confidence_ok(result["digits"], result["confidences"], args.min_digit_conf):
            img = _overlay_solution(img, result["biggest"], result["solution_overlay"])
            cv2.imshow("Sudoku Solver", img)
            if args.save:
                cv2.imwrite(args.save, img)
        else:
            print("Could not solve. Check if digits were recognized correctly.")
            cv2.imshow("Sudoku Solver", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Could not open webcam (index {args.camera_index}). Try closing other camera apps or changing the camera index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    solved_state = False
    solved_numbers = []
    locked_biggest = None
    candidate_biggest = None
    candidate_area = 0.0
    stable_frames = 0
    last_detected = 0.0
    frame_i = 0
    history = deque(maxlen=max(1, args.temporal_window))
    last_digits = None
    stable_ocr = 0
    solved_source_digits = None
    last_status = ""
    
    print("Press 's' to capture and solve the Sudoku.")
    print("Press 'q' to quit.")
    print("Press 'r' to reset solution.")
    print("Press 'a' to toggle auto-solve.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
        frame_i += 1
        frame = img.copy()

        img_processed = ip.pre_process_image(frame)
        contours, max_area = ip.find_contours(img_processed, frame)
        biggest = None
        
        if contours.size != 0:
            biggest = ip.reorder(contours)
            last_detected = time.time()

            if locked_biggest is None:
                if candidate_biggest is None:
                    candidate_biggest = biggest
                    candidate_area = float(max_area)
                    stable_frames = 1
                else:
                    area = float(max_area)
                    ratio = area / candidate_area if candidate_area > 0 else 0
                    if 0.85 <= ratio <= 1.15:
                        stable_frames += 1
                    else:
                        candidate_biggest = biggest
                        candidate_area = area
                        stable_frames = 1

                if stable_frames >= max(1, args.lock_frames):
                    locked_biggest = candidate_biggest
            else:
                if not solved_state:
                    locked_biggest = biggest
                    candidate_area = float(max_area)
            
            # Draw the detected grid contour for feedback
            _draw_grid_outline(img, locked_biggest if locked_biggest is not None else biggest)
            
            # If we have a solution, overlay it
            if solved_state:
                use_biggest = locked_biggest if locked_biggest is not None else biggest
                img = _overlay_solution(img, use_biggest, solved_numbers)

        if auto_mode and (frame_i % max(1, args.solve_every) == 0):
            use_biggest = locked_biggest if locked_biggest is not None else biggest
            if use_biggest is not None:
                result = _solve_from_frame(frame, recognizer, biggest=use_biggest)
                if result is not None:
                    history.append((result["digits"], result["confidences"]))
                    digits, confidences = _temporal_filter(history, args.min_digit_conf)

                    if last_digits is not None and digits == last_digits:
                        stable_ocr += 1
                    else:
                        last_digits = digits
                        stable_ocr = 1

                    if solved_source_digits is not None and digits != solved_source_digits:
                        solved_state = False
                        solved_numbers = []
                        solved_source_digits = None

                    original_board = np.array(digits).reshape(9, 9).tolist()
                    conflicts = _count_conflicts(original_board)
                    conf_ok = _confidence_ok(digits, confidences, args.min_digit_conf)
                    nonzero_conf = [c for d, c in zip(digits, confidences) if d != 0]
                    min_conf = min(nonzero_conf) if nonzero_conf else 0.0
                    filled = sum(1 for d in digits if d != 0)
                    if args.debug:
                        status = f"stable={stable_ocr}/{args.stable_reads} conf_ok={int(conf_ok)} min_conf={min_conf:.2f} filled={filled} conflicts={conflicts}"
                        if status != last_status:
                            print(status)
                            if conflicts > 0:
                                print("OCR board is invalid (duplicate digits).")
                                print(np.array(original_board))
                            elif not conf_ok:
                                print("OCR confidence too low. Try --min-digit-conf 0.70 to be less strict.")
                            last_status = status
                            dbg = cv2.cvtColor(result["img_warped_clean"], cv2.COLOR_GRAY2BGR)
                            ip.display_numbers_with_confidence(dbg, digits, confidences)
                            cv2.imshow("Warped Grid (Debug)", dbg)

                    if stable_ocr >= max(1, args.stable_reads) and conf_ok and conflicts == 0:
                        if args.debug:
                            dbg = cv2.cvtColor(result["img_warped_clean"], cv2.COLOR_GRAY2BGR)
                            ip.display_numbers_with_confidence(dbg, digits, confidences)
                            cv2.imshow("Warped Grid (Debug)", dbg)
                        else:
                            cv2.imshow("Warped Grid", result["img_warped_clean"])

                        board_to_solve = np.array(digits).reshape(9, 9).tolist()
                        original_copy = [row[:] for row in board_to_solve]
                        if solver.solve_sudoku(board_to_solve):
                            solved_flat = [item for sublist in board_to_solve for item in sublist]
                            original_flat = [item for sublist in original_copy for item in sublist]
                            display_sol = []
                            for i in range(81):
                                display_sol.append(solved_flat[i] if original_flat[i] == 0 else 0)

                            solved_numbers = display_sol
                            solved_state = True
                            solved_source_digits = digits

        status = "AUTO" if auto_mode else "MANUAL"
        cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Sudoku Solver", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            use_biggest = locked_biggest if locked_biggest is not None else biggest
            if use_biggest is None:
                print("No Sudoku grid detected in frame.")
                solved_state = False
                continue

            print("Solving...")
            result = _solve_from_frame(frame, recognizer, biggest=use_biggest)
            if result is None:
                print("No Sudoku grid detected in frame.")
                solved_state = False
                continue

            if args.debug:
                dbg = cv2.cvtColor(result["img_warped_clean"], cv2.COLOR_GRAY2BGR)
                ip.display_numbers_with_confidence(dbg, result["digits"], result["confidences"])
                cv2.imshow("Warped Grid (Debug)", dbg)
            else:
                cv2.imshow("Warped Grid", result["img_warped_clean"])

            print("\nDetected numbers:")
            print(np.array(result["digits"]).reshape(9, 9))

            if result["solved"] and _confidence_ok(result["digits"], result["confidences"], args.min_digit_conf):
                print("Solved!")
                solved_numbers = result["solution_overlay"]
                solved_state = True
                locked_biggest = result["biggest"]
                solved_source_digits = result["digits"]
            else:
                print("Could not solve. Check if digits were recognized correctly.")
                solved_state = False

        if key == ord('r'):
            solved_state = False
            solved_numbers = []
            solved_source_digits = None

        if key == ord('a'):
            auto_mode = not auto_mode
            solved_state = False
            solved_numbers = []
            solved_source_digits = None
            history.clear()
            last_digits = None
            stable_ocr = 0

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
