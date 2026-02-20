import cv2
import numpy as np
import os


class DocumentScanner:
    def __init__(self):
        pass

    def load_image(self, path):
        img = cv2.imread(path)
        return img

    def to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def gaussian_filter(self, gray, ks=5, sigma=1.0):
        kerX = cv2.getGaussianKernel(ks, sigma)
        kerY = cv2.transpose(kerX)
        ker_gau = kerX * kerY
        blurred = cv2.filter2D(gray, cv2.CV_8U, ker_gau)
        return blurred

    def suppress_texture(self, gray, ksize=(25, 25), iterations=1):
        kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_big, iterations=iterations)
        return closed

    def canny(self, img, low=50, high=150):
        return cv2.Canny(img, low, high)

    def strengthen_edges(self, edges, ksize=(3, 3), dilate_iter=1, close_iter=1):
        kernel = np.ones(ksize, np.uint8)
        edges2 = cv2.dilate(edges, kernel, iterations=dilate_iter)
        edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        return edges2

    def find_document_contour(self, edges2):
        cnts, _ = cv2.findContours(edges2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        doc = None
        area = 0
        for c in cnts[:15]:  # check top 15 contours
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc = approx
                area = cv2.contourArea(approx)
                break

        return doc, area

    def warp(self, img, doc):
        pts = doc.reshape(4, 2).astype("float32")

        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left

        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return warped

    def normalize_light(self, warped_gray, blur_ksize=(55, 55)):
        bg = cv2.GaussianBlur(warped_gray, blur_ksize, 0)
        normalized = cv2.divide(warped_gray, bg, scale=255)
        return normalized

    def to_binary(self, normalized, block_size=25, C=10):
        if block_size % 2 == 0:
            block_size += 1

        binary = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )
        return binary

    def scan(self, path, save=True, out_path=None):
        img = self.load_image(path)
        if img is None:
            return None

        gray = self.to_gray(img)

        # -------- Method 1: Gaussian (for normal documents) --------
        blurred1 = self.gaussian_filter(gray, ks=5, sigma=1.0)
        edges1 = self.canny(blurred1, 50, 150)
        edges1_strong = self.strengthen_edges(edges1, ksize=(3, 3), dilate_iter=1, close_iter=1)
        doc1, area1 = self.find_document_contour(edges1_strong)

        # -------- Method 2: Morphology (for grid/texture documents) --------
        blurred2 = self.gaussian_filter(gray, ks=5, sigma=1.0)
        closed = self.suppress_texture(blurred2, ksize=(25, 25), iterations=1)
        edges2 = self.canny(closed, 50, 150)
        edges2_strong = self.strengthen_edges(edges2, ksize=(3, 3), dilate_iter=1, close_iter=1)
        doc2, area2 = self.find_document_contour(edges2_strong)

        # -------- Select the best method (largest area) --------
        if area1 > area2:
            doc = doc1
        else:
            doc = doc2

        if doc is None:
            return None

        warped = self.warp(img, doc)
        warped_gray = self.to_gray(warped)
        normalized = self.normalize_light(warped_gray, blur_ksize=(55, 55))
        binary = self.to_binary(normalized, block_size=25, C=10)

        if not save:
            return binary

        # if caller doesn't give out_path, save next to input
        if out_path is None:
            folder = os.path.dirname(path)
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(folder, f"{base}_scanned.png")

        ok = cv2.imwrite(out_path, binary)
        if not ok:
            return None

        # return the saved file path (so UI can load it by QPixmap(path))
        return out_path