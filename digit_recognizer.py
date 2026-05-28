import numpy as np
import cv2
from tensorflow.keras.models import load_model

class DigitRecognizer:
    def __init__(self, model_path='model.h5'):
        try:
            self.model = load_model(model_path)
            print("Model loaded successfully")
        except:
            print(f"Warning: Could not load model from {model_path}. Please run train_model.py first.")
            self.model = None

    def pre_process_cell(self, img):
        """
        Process a single cell image for prediction to match MNIST format:
        1. Threshold
        2. Remove borders
        3. Find digit bounding box
        4. Resize digit to 20x20 maintaining aspect ratio
        5. Center in 28x28 image
        """
        # 1. Threshold
        # Check if image is already grayscale
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(np.mean(thresh)) > 127.0:
            thresh = cv2.bitwise_not(thresh)

        # 2. Remove borders (clear a few pixels around the edge)
        h, w = thresh.shape
        margin = max(6, int(min(h, w) * 0.14))
        thresh[0:margin, :] = 0
        thresh[h-margin:h, :] = 0
        thresh[:, 0:margin] = 0
        thresh[:, w-margin:w] = 0
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        if num_labels <= 1:
            return np.zeros((1, 28, 28, 1))

        cell_area = float(thresh.shape[0] * thresh.shape[1])
        min_area = max(40.0, cell_area * 0.01)
        max_area = cell_area * 0.35

        best = None
        best_area = 0.0
        for label in range(1, num_labels):
            x, y, bw, bh, area = stats[label]
            cx, cy = centroids[label]

            if area < min_area or area > max_area:
                continue
            if x <= margin or y <= margin or (x + bw) >= (thresh.shape[1] - margin) or (y + bh) >= (thresh.shape[0] - margin):
                continue

            if not (thresh.shape[1] * 0.12 <= cx <= thresh.shape[1] * 0.88):
                continue
            if not (thresh.shape[0] * 0.12 <= cy <= thresh.shape[0] * 0.88):
                continue

            if area > best_area:
                best = (x, y, bw, bh, label)
                best_area = float(area)

        if best is None:
            return np.zeros((1, 28, 28, 1))

        x, y, bw, bh, label = best
        component_mask = (labels == label).astype(np.uint8) * 255
        digit = cv2.bitwise_and(thresh, component_mask)[y : y + bh, x : x + bw]
        
        # 4. Resize to fit in 20x20 box (MNIST standard)
        # Calculate scale to fit 20px
        if bh > bw:
            scale = 20.0 / bh
            new_h = 20
            new_w = int(bw * scale)
        else:
            scale = 20.0 / bw
            new_w = 20
            new_h = int(bh * scale)
            
        if new_w <= 0 or new_h <= 0:
             return np.zeros((1, 28, 28, 1))
             
        resized_digit = cv2.resize(digit, (new_w, new_h))
        
        # 5. Center in 28x28 image
        final_img = np.zeros((28, 28), dtype=np.uint8)
        
        # Calculate offsets to center
        y_off = (28 - new_h) // 2
        x_off = (28 - new_w) // 2
        
        final_img[y_off:y_off+new_h, x_off:x_off+new_w] = resized_digit
        
        # Normalize
        final_img = final_img / 255.0
        final_img = final_img.reshape(1, 28, 28, 1)
        
        return final_img

    def predict(self, img):
        digit, _ = self.predict_with_confidence(img)
        return digit

    def predict_with_confidence(self, img):
        if self.model is None:
            return 0, 0.0

        processed = self.pre_process_cell(img)
        if np.sum(processed) == 0:
            return 0, 0.0

        prediction = self.model.predict(processed, verbose=0)
        class_index = int(np.argmax(prediction, axis=-1)[0])
        probability = float(np.amax(prediction))

        if probability > 0.7:
            return class_index, probability
        return 0, probability

    def predict_cells_with_confidence(self, cells):
        if self.model is None:
            return [0] * len(cells), [0.0] * len(cells)

        processed_cells = []
        empty_mask = []
        for cell in cells:
            processed = self.pre_process_cell(cell)
            if np.sum(processed) == 0:
                empty_mask.append(True)
                processed_cells.append(processed)
            else:
                empty_mask.append(False)
                processed_cells.append(processed)

        batch = np.vstack(processed_cells)
        predictions = self.model.predict(batch, verbose=0)

        digits = []
        confidences = []
        for i in range(predictions.shape[0]):
            class_index = int(np.argmax(predictions[i]))
            probability = float(np.max(predictions[i]))
            if empty_mask[i]:
                digits.append(0)
                confidences.append(0.0)
            elif probability > 0.7:
                digits.append(class_index)
                confidences.append(probability)
            else:
                digits.append(0)
                confidences.append(probability)

        return digits, confidences
