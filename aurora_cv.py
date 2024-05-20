import cv2
import numpy as np
import string


def convert_to_grayscale(image_data: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)


def apply_threshold(
        image_data: np.ndarray, threshold_value: int = 100, max_value: int = 150
) -> np.ndarray:
    _, binary_image = cv2.threshold(
        image_data, threshold_value, max_value, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary_image


def apply_gaussian_blur(image_data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(image_data, (kernel_size, kernel_size), 0)


def apply_morphological_operation(
        image_data: np.ndarray, operation: int = cv2.MORPH_CLOSE, kernel_size: int = 5
) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image_data, operation, kernel)


def apply_unsharp_mask(image_data: np.ndarray) -> np.ndarray:
    blurred_image = cv2.GaussianBlur(image_data, (5, 5), 0)
    return cv2.addWeighted(image_data, 1.5, blurred_image, -0.5, 0)


def apply_bilateral_filtering(image_data: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image_data, 5, 50, 50)


def apply_adaptive_threshold(image_data: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )


def apply_non_local_means_denoising(image_data: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(image_data, None, 10, 7, 21)


def find_contours(binary_image: np.ndarray) -> list:
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def find_largest_contour(contour_list: list) -> np.ndarray:
    return max(contour_list, key=cv2.contourArea)


def approximate_contour(contour: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon * perimeter, True)


def find_table_corners(image_data: np.ndarray) -> np.ndarray:
    grayscale_image = convert_to_grayscale(image_data)
    blurred_image = apply_gaussian_blur(grayscale_image)
    binary_image = apply_threshold(blurred_image)
    binary_image = apply_morphological_operation(binary_image, cv2.MORPH_CLOSE, 5)
    contour_list = find_contours(binary_image)
    largest_contour = find_largest_contour(contour_list)
    corner_points = approximate_contour(largest_contour)
    return corner_points


def fix_perspective(image_data: np.ndarray, corner_points: np.ndarray) -> np.ndarray:
    corner_points = corner_points.reshape(-1, 2)
    center = np.mean(corner_points, axis=0)
    angles = np.arctan2(corner_points[:, 1] - center[1], corner_points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    corner_points = corner_points[sorted_indices]

    if corner_points.shape[0] < 4:
        raise ValueError("At least 4 corner points are required")

    source_points = np.float32(corner_points[:4])
    width = max(
        np.linalg.norm(source_points[0] - source_points[1]),
        np.linalg.norm(source_points[2] - source_points[3]),
    )
    height = max(
        np.linalg.norm(source_points[0] - source_points[3]),
        np.linalg.norm(source_points[1] - source_points[2]),
    )
    destination_points = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    perspective_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    corrected_image = cv2.warpPerspective(
        image_data, perspective_matrix, (int(width), int(height))
    )
    return corrected_image


def deskew_image(image_data: np.ndarray) -> np.ndarray:
    grayscale_image = convert_to_grayscale(image_data)
    binary_image = apply_threshold(grayscale_image)
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (x, y), (w, h), angle = rect
    if angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
    (h, w) = image_data.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected_image = cv2.warpAffine(image_data, M, (w, h))
    return corrected_image


def detect_trapezoidal_table(image_data: np.ndarray) -> bool:
    grayscale_image = convert_to_grayscale(image_data)
    binary_image = apply_threshold(grayscale_image)
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if aspect_ratio > 1.5:
        return True
    return False


def process_image(image_data: np.ndarray) -> np.ndarray:
    image_data = deskew_image(image_data)
    corner_points = find_table_corners(image_data)
    corrected_image = fix_perspective(image_data, corner_points)
    if detect_trapezoidal_table(corrected_image):
        corrected_image = apply_unsharp_mask(corrected_image)
        corrected_image = apply_bilateral_filtering(corrected_image)
        corrected_image = apply_adaptive_threshold(corrected_image)
        corrected_image = apply_non_local_means_denoising(corrected_image)
    return corrected_image


def apply_filter(image: np.ndarray, filter_type: str) -> np.ndarray:
    if filter_type == "gaussian_blur":
        filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "median_blur":
        filtered_image = cv2.medianBlur(image, 5)
    elif filter_type == "bilateral_filter":
        filtered_image = cv2.bilateralFilter(image, 5, 50, 50)
    elif filter_type == "canny_edge_detection":
        filtered_image = cv2.Canny(image, 100, 200)
    else:
        filtered_image = image
    return filtered_image


def scan_and_mark_cells(image: np.ndarray) -> tuple:
    filtered_image = apply_filter(image, "bilateral_filter")

    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        gray_image, 140, 240, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    other_contours = [
        contour
        for contour in contours
        if cv2.boundingRect(contour)[0] > 5 and cv2.boundingRect(contour)[1] > 5
    ]
    first_column_contours = [
        contour for contour in contours if cv2.boundingRect(contour)[0] < 5
    ]
    first_row_contours = [
        contour for contour in contours if cv2.boundingRect(contour)[1] < 5
    ]

    marked_image = image.copy()

    for contour in other_contours:
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(marked_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return other_contours, first_column_contours, first_row_contours


def enhance_contrast_and_sharpness(image: np.ndarray) -> np.ndarray:
    enhanced_image = cv2.convertScaleAbs(
        image, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA
    )
    sharpened_image = cv2.filter2D(enhanced_image, -1, SHARPENING_KERNEL)
    return sharpened_image


def detect_empty_contours(image: np.ndarray, contours: list) -> list:
    marked_img = image.copy()
    marked_img = enhance_contrast_and_sharpness(marked_img)
    marked_cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = marked_img[y: y + h, x: x + w]
        mean_intensity = np.mean(roi)
        if mean_intensity < THRESHOLD_VALUE:
            marked_cells.append(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return marked_cells


def remove_small_contours(contours: list, min_area_threshold: int) -> list:
    return [
        contour for contour in contours if cv2.contourArea(contour) > min_area_threshold
    ]


def remove_similar_contours(contours: list, threshold_distance: int) -> list:
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        close_to_existing = False
        for existing_contour in filtered_contours:
            existing_x, existing_y, _, _ = cv2.boundingRect(existing_contour)
            if abs(y - existing_y) < threshold_distance:
                close_to_existing = True
                break
        if not close_to_existing:
            filtered_contours.append(contour)
    return filtered_contours


def find_contour_centers(contours: list) -> list:
    centers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        centers.append((center_x, center_y))
    return centers


def find_answer(
        marked_contours: list,
        first_column_contours_dict: dict,
        first_row_contours_dict: dict,
        correct_answers: list
) -> dict:
    answer = set()
    for point in marked_contours:
        x_point, y_point = point
        for key_row, value_row in first_column_contours_dict.items():
            x_row, y_row, w_row, h_row = cv2.boundingRect(value_row)
            if y_row <= y_point <= y_row + h_row:
                for key_col, value_col in first_row_contours_dict.items():
                    x_col, y_col, w_col, h_col = cv2.boundingRect(value_col)
                    if x_col <= x_point <= x_col + w_col:
                        answer.add((key_row, key_col))
    print(answer)
    result = {"answer": [], "total-correct-answers": 0, "total-incorrect-answers": 0}
    for row, col in answer:
        found = False
        for correct_answer in correct_answers:
            if row == correct_answer["question"]:
                correct_answer_value = correct_answer["correct_answer"]
                result["answer"].append({
                    "question": row,
                    "answer": col,
                    "correct-answer": correct_answer_value
                })
                if col == correct_answer_value:
                    result["total-correct-answers"] += 1
                else:
                    result["total-incorrect-answers"] += 1
                found = True
                break
        if not found:
            result["answer"].append({
                "question": row,
                "answer": "",
                "correct-answer": ""
            })
            result["total-incorrect-answers"] += 1
    return result


IMAGE_PATH = "photos/photo.jpeg"
CONTRAST_ALPHA = 0.8
CONTRAST_BETA = 5
SHARPENING_KERNEL = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]])
THRESHOLD_VALUE = 240

CORRECT_ANSWER_TEST = [
    {'question': '1', 'correct_answer': 'A'},
    {'question': '7', 'correct_answer': 'A'},
    {'question': '3', 'correct_answer': 'C'},
    {'question': '2', 'correct_answer': 'B'},
    {'question': '5', 'correct_answer': 'C'},
    {'question': '4', 'correct_answer': 'D'},
    {'question': '6', 'correct_answer': 'B'}
]


def get_answer(path: str, correct_answers: list):
    MIN_AREA_THRESHOLD = 120
    THRESHOLD_DISTANCE = 5

    image_data = cv2.imread(path)

    try:
        processed_image = process_image(image_data)
    except Exception as e:
        return {"error": f"Ошибка обработки изображения: {e}"}

    other_contours, first_column_contours, first_row_contours = scan_and_mark_cells(
        processed_image
    )

    try:
        min_x_contour = min(
            first_row_contours, key=lambda contour: cv2.boundingRect(contour)[0]
        )
        first_row_contours_without_min_x = [
            contour for contour in first_row_contours if cv2.boundingRect(contour)[0] >= 10
        ]

        min_y_contour = min(
            first_column_contours, key=lambda contour: cv2.boundingRect(contour)[1]
        )
        first_column_contours_without_min_x = [
            contour for contour in first_column_contours if cv2.boundingRect(contour)[1] >= 10
        ]
    except Exception as e:
        raise ValueError(f"Error processing contours: {e}")

    marked_cells = detect_empty_contours(processed_image, other_contours)

    sorted_first_row_contours_without_min_x = sorted(
        first_row_contours_without_min_x, key=lambda contour: cv2.boundingRect(contour)[0]
    )
    sorted_first_column_contours_without_min_x = sorted(
        first_column_contours_without_min_x,
        key=lambda contour: cv2.boundingRect(contour)[1],
    )

    contours_without_small = remove_small_contours(
        sorted_first_column_contours_without_min_x, MIN_AREA_THRESHOLD
    )
    filtered_contours = remove_similar_contours(contours_without_small, THRESHOLD_DISTANCE)

    sorted_first_column_contours_without_min_x = filtered_contours

    first_row_contours_dict = {
        letter: contour
        for letter, contour in zip(
            string.ascii_uppercase, sorted_first_row_contours_without_min_x
        )
    }

    first_column_contours_dict = {
        str(i + 1): contour
        for i, contour in enumerate(sorted_first_column_contours_without_min_x)
    }

    marked_points = find_contour_centers(marked_cells)

    answer = find_answer(marked_points, first_column_contours_dict, first_row_contours_dict, correct_answers)

    return answer
