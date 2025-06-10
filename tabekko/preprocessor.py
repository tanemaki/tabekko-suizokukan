import cv2
import numpy as np


class ImagePreProcessor:
    def __init__(self, image: np.ndarray, edge_length=28):
        self.edge_length = edge_length
        self._resized_original_image = self.resize_image(image, 620, 480)
        self._edges = self.detect_edges(self._resized_original_image)
        self._contours = self.find_contours(self._edges)
        self._squares = self.find_squares(self._contours)
        self._annotated_image = self.annotate_image_with_squares(
            self._resized_original_image, self._squares
        )

        if len(self._squares) == 0:
            self._sorted_square = None
            self._square_image = None
            self._image_binarlizer = None
            self._standardized_image = None
        else:
            # n.b., self._squares[0] implicitly assume the first entry is special
            self._sorted_square = self.sort_points(self._squares[0])
            self._square_image = self.get_image_within_square(
                image=self._resized_original_image, square=self._sorted_square
            )

            self._image_binarlizer = ImageBinalizer(self._square_image)

            self._standardized_image = self.resize_image(
                self.noise_removed_image,
                width=self.edge_length,
                height=self.edge_length,
            )

    @property
    def binary_image(self):
        return self._image_binarlizer.binary_image

    @property
    def noise_removed_image(self):
        return self._image_binarlizer.noise_removed_image

    @property
    def standardized_image(self):
        return self._standardized_image

    @staticmethod
    def resize_image(image, width, height):
        return cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )

    @property
    def resized_original_image(self):
        return self._resized_original_image

    @staticmethod
    def detect_edges(image) -> np.ndarray:
        # 画像をグレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # エッジ検出
        edges = cv2.Canny(gray, 100, 250)

        # 検出された線を太くする
        kernel = np.ones((7, 7), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return edges

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @staticmethod
    def find_contours(edges):
        # Find contours in the edges
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return contours

    @staticmethod
    def find_squares(contours):
        # Filter contours to find squares
        squares = []
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # Check if the approximated contour has 4 points and is convex
            # also check if the area is large enough
            area = cv2.contourArea(approx)

            if len(approx) == 4 and cv2.isContourConvex(approx) and (area > 1000):
                squares.append(approx)

        return squares

    @property
    def num_squares(self):
        return len(self._squares)

    @staticmethod
    def annotate_image_with_squares(image, squares):
        # Draw the detected squares on the image
        annotated_image = image.copy()
        for square in squares:
            cv2.polylines(annotated_image, [square], True, (0, 255, 0), 2)

        return annotated_image

    @property
    def annotated_image(self):
        return self._annotated_image

    @staticmethod
    def sort_points(square):
        # 4点の座標の大小関係を比べ、左上、右上、右下、左下と時計回りに並ぶように並べ替える
        square = square.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and difference of points to determine their positions
        s = square.sum(axis=1)
        diff = np.diff(square, axis=1)

        rect[0] = square[np.argmin(s)]  # Top-left
        rect[2] = square[np.argmax(s)]  # Bottom-right
        rect[1] = square[np.argmin(diff)]  # Top-right
        rect[3] = square[np.argmax(diff)]  # Bottom-left

        return rect

    @staticmethod
    def get_image_within_square(image, square, edge_pixels=400):
        pts1 = square.reshape(4, 2).astype("float32")
        pts2 = np.array(
            [[0, 0], [edge_pixels, 0], [edge_pixels, edge_pixels], [0, edge_pixels]],
            dtype="float32",
        )

        # Compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # Apply the perspective transformation
        transformed_image = cv2.warpPerspective(image, M, (edge_pixels, edge_pixels))

        return transformed_image

    @property
    def square_image(self):
        return self._square_image


class ImageBinalizer:
    def __init__(self, image):
        self._binary_image = self.binarize_image(image)
        self._noise_removed_image = self.remove_noise(self._binary_image)

    @property
    def binary_image(self) -> np.ndarray:
        return self._binary_image

    @property
    def noise_removed_image(self) -> np.ndarray:
        return self._noise_removed_image

    @staticmethod
    def binarize_image(image):
        # transformed_imageの輪郭を検出し、二値化する。中央にある物体を白、背景を黒とする。
        # Convert the transformed image to grimage_preprocessor.square_imageayscale
        gray_transformed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to isolate the central object
        _, binary_image = cv2.threshold(
            gray_transformed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Invert the binary image to make the central object white and background black
        inverted_image = cv2.bitwise_not(binary_image)

        return inverted_image

    @staticmethod
    def remove_noise(bianry_image):
        # ノイズを除去するために、モルフォロジー処理を行う
        kernel = np.ones((9, 9), np.uint8)

        # Repeat erosion followed by dilation several times（周りの線を消す処理）
        for _ in range(5):  # Repeat the process 3 times
            bianry_image = cv2.erode(bianry_image, kernel, iterations=1)
            bianry_image = cv2.dilate(bianry_image, kernel, iterations=1)

        kernel = np.ones((5, 5), np.uint8)

        # Remove small holes in the object
        bianry_image = cv2.morphologyEx(
            bianry_image, cv2.MORPH_CLOSE, kernel, iterations=5
        )
        return bianry_image
