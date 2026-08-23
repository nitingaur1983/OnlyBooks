from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from pytesseract import Output


@dataclass
class SpineCandidate:
    index: int
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float


class ShelfVision:
    """POC vision pipeline.

    This intentionally uses classical CV + Tesseract so there are no API keys or
    hosted-model dependencies. The interface can later be backed by YOLO/DETR/etc.
    """

    def detect_and_read(self, image_path: str | Path) -> list[SpineCandidate]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Unable to read uploaded image")

        boxes = self._detect_spines(image)
        results: list[SpineCandidate] = []

        for idx, (x, y, w, h) in enumerate(boxes):
            crop = image[y:y+h, x:x+w]
            text, conf = self._ocr_best_orientation(crop)
            if text.strip():
                results.append(
                    SpineCandidate(
                        index=idx,
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        text=text.strip(),
                        confidence=conf,
                    )
                )

        # Fallback: if contour segmentation fails, OCR the whole shelf image once.
        if not results:
            text, conf = self._ocr_best_orientation(image)
            h, w = image.shape[:2]
            if text.strip():
                results.append(
                    SpineCandidate(0, 0, 0, w, h, text.strip(), conf)
                )

        return results

    def _detect_spines(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Vertical edges tend to outline book spines.
        sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(15, h // 30)))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int]] = []
        min_h = int(h * 0.28)
        min_w = max(12, int(w * 0.012))
        max_w = int(w * 0.22)

        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = ch / max(cw, 1)
            if ch >= min_h and min_w <= cw <= max_w and aspect >= 1.8:
                pad_x = max(2, int(cw * 0.08))
                pad_y = max(2, int(ch * 0.02))
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(w, x + cw + pad_x)
                y1 = min(h, y + ch + pad_y)
                candidates.append((x0, y0, x1 - x0, y1 - y0))

        # Remove highly overlapping boxes, then sort left-to-right.
        candidates = self._dedupe(candidates)
        candidates.sort(key=lambda b: (b[1] // max(h // 4, 1), b[0]))
        return candidates[:80]

    @staticmethod
    def _iou(a, b) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    def _dedupe(self, boxes):
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        kept = []
        for box in boxes:
            if all(self._iou(box, existing) < 0.45 for existing in kept):
                kept.append(box)
        return kept

    def _ocr_best_orientation(self, crop: np.ndarray) -> tuple[str, float]:
        attempts = []
        for angle in (0, 90, 270):
            rotated = self._rotate(crop, angle)
            prepared = self._prepare_for_ocr(rotated)
            data = pytesseract.image_to_data(
                prepared,
                output_type=Output.DICT,
                config="--oem 3 --psm 6",
            )
            words, confs = [], []
            for text, conf in zip(data["text"], data["conf"]):
                text = text.strip()
                try:
                    conf_val = float(conf)
                except (TypeError, ValueError):
                    conf_val = -1
                if text and conf_val >= 0:
                    words.append(text)
                    confs.append(conf_val)
            joined = " ".join(words)
            avg = float(sum(confs) / len(confs)) if confs else 0.0
            attempts.append((joined, avg))

        # Prefer a result with both text volume and confidence.
        return max(attempts, key=lambda t: (t[1] * max(len(t[0]), 1), t[1]))

    @staticmethod
    def _prepare_for_ocr(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Upscaling materially helps text on narrow spines.
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        return gray

    @staticmethod
    def _rotate(image: np.ndarray, angle: int) -> np.ndarray:
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
