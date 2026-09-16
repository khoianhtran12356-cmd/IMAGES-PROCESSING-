import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

# ------------------------------------------------------------
# ĐƯỜNG DẪN
# ------------------------------------------------------------

TEMPLATE_DIR = r"D:\CharacterDataset\templates"
INPUT_DIR = r"D:\CharacterDataset\input"
OUTPUT_DIR = r"D:\CharacterDataset\output"


# ------------------------------------------------------------
# CLASS
# ------------------------------------------------------------
# Nếu bạn có 1-9 + A-Z = 35 class.
#
# Nếu có thêm số 0 thì sửa thành:
# CLASS_NAMES = list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#
# Hoặc nếu bộ template của bạn có class khác,
# chỉ cần sửa danh sách này.
# ------------------------------------------------------------

CLASS_NAMES = list("123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ------------------------------------------------------------
# TEMPLATE MATCHING
# ------------------------------------------------------------

# Score cuối:
#
# score =
#       INTENSITY_WEIGHT * intensity_score
#     + GRADIENT_WEIGHT  * gradient_score
#
# Vì ký tự màu đen tương phản thấp nên ưu tiên gradient.

INTENSITY_WEIGHT = 0.35
GRADIENT_WEIGHT = 0.65


# Threshold ban đầu.
#
# Sau khi chạy thử nên xem score thực tế rồi điều chỉnh.
MATCH_THRESHOLD = 0.55


# ------------------------------------------------------------
# PEAK DETECTION
# ------------------------------------------------------------

# Khoảng cách tối thiểu giữa hai candidate.
#
# Thường nên gần kích thước ký tự.
PEAK_DISTANCE = 20


# ------------------------------------------------------------
# NMS
# ------------------------------------------------------------

NMS_IOU_THRESHOLD = 0.20


# ------------------------------------------------------------
# CONFIDENCE REVIEW
# ------------------------------------------------------------

# >= AUTO_THRESHOLD:
#     tự động chấp nhận
#
# REVIEW_THRESHOLD <= score < AUTO_THRESHOLD:
#     vẫn ghi label nhưng debug màu khác
#
# < REVIEW_THRESHOLD:
#     không ghi label

AUTO_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.55


# ------------------------------------------------------------
# SCORE MARGIN
# ------------------------------------------------------------
# Nếu 2 class có score quá gần nhau thì đánh dấu REVIEW.
#
# Ví dụ:
#
# 3 = 0.72
# B = 0.70
#
# margin = 0.02 -> không chắc chắn.

MIN_SCORE_MARGIN = 0.08


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_gray(img):
    """
    Grayscale
    + CLAHE
    + Gaussian blur nhẹ
    """

    if img.ndim == 3:
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = img.copy()

    # Tăng tương phản cục bộ
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # Giảm nhiễu nhẹ
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray


def make_gradient(gray):
    """
    Tạo ảnh gradient magnitude.
    """

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    # Chuẩn hóa
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude.astype(np.uint8)


# ============================================================
# LOAD TEMPLATE
# ============================================================

def load_templates():

    template_dir = Path(TEMPLATE_DIR)

    templates = {}

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    print("=" * 70)
    print("LOADING TEMPLATES")
    print("=" * 70)

    for class_name in CLASS_NAMES:

        # ----------------------------------------------------
        # Tìm file template
        # ----------------------------------------------------

        candidates = []

        for ext in valid_extensions:

            p = template_dir / f"{class_name}{ext}"

            if p.exists():
                candidates.append(p)

        if len(candidates) == 0:

            print(
                f"[ERROR] Missing template: "
                f"{class_name}"
            )

            continue

        path = candidates[0]

        # ----------------------------------------------------
        # Read
        # ----------------------------------------------------

        img = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:

            print(
                f"[ERROR] Cannot read: {path}"
            )

            continue

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        gray = preprocess_gray(img)

        gradient = make_gradient(gray)

        h, w = gray.shape

        templates[class_name] = {

            "gray": gray,

            "gradient": gradient,

            "width": w,

            "height": h
        }

        print(
            f"[OK] {class_name:>2} "
            f"-> {w} x {h}"
        )

    print()
    print(
        f"Loaded: {len(templates)} / "
        f"{len(CLASS_NAMES)} templates"
    )

    return templates


# ============================================================
# LOCAL MAXIMUM / PEAK EXTRACTION
# ============================================================

def find_peaks(
    score_map,
    threshold,
    min_distance
):
    """
    Lấy local maxima từ score map.

    Không lấy tất cả pixel > threshold vì một vùng match
    tốt có thể tạo ra hàng trăm pixel liên tiếp.

    Ta chỉ lấy peak.
    """

    # Kernel xác định vùng lân cận
    k = max(
        3,
        int(min_distance)
    )

    if k % 2 == 0:
        k += 1

    kernel = np.ones(
        (k, k),
        np.uint8
    )

    # Maximum trong vùng lân cận
    local_max = cv2.dilate(
        score_map,
        kernel
    )

    # Pixel nào vừa là maximum vừa vượt threshold
    mask = (
        (score_map >= threshold) &
        (score_map >= local_max - 1e-6)
    )

    ys, xs = np.where(mask)

    candidates = []

    for x, y in zip(xs, ys):

        candidates.append(
            (
                int(x),
                int(y),
                float(score_map[y, x])
            )
        )

    # Sort score giảm dần
    candidates.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return candidates


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    ix1 = max(x1, a1)
    iy1 = max(y1, b1)

    ix2 = min(x2, a2)
    iy2 = min(y2, b2)

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = iw * ih

    area1 = max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )

    area2 = max(
        0,
        a2 - a1
    ) * max(
        0,
        b2 - b1
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# GLOBAL NMS
# ============================================================

def global_nms(
    detections,
    iou_threshold
):
    """
    NMS trên tất cả class.

    Điều này rất quan trọng.

    Ví dụ cùng một vị trí:

        3 = 0.91
        B = 0.63

    => giữ 3.
    """

    detections = sorted(
        detections,
        key=lambda d: d["score"],
        reverse=True
    )

    selected = []

    for det in detections:

        keep = True

        for selected_det in selected:

            iou = calculate_iou(
                det["box"],
                selected_det["box"]
            )

            if iou > iou_threshold:

                keep = False
                break

        if keep:
            selected.append(det)

    return selected


# ============================================================
# MATCH ONE TEMPLATE
# ============================================================

def match_one_template(
    image_gray,
    image_gradient,
    template,
    class_name
):

    template_gray = template["gray"]
    template_gradient = template["gradient"]

    th, tw = template_gray.shape

    H, W = image_gray.shape

    # Template lớn hơn image
    if th > H or tw > W:
        return []

    # --------------------------------------------------------
    # Intensity matching
    # --------------------------------------------------------

    score_intensity = cv2.matchTemplate(
        image_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    # --------------------------------------------------------
    # Gradient matching
    # --------------------------------------------------------

    score_gradient = cv2.matchTemplate(
        image_gradient,
        template_gradient,
        cv2.TM_CCOEFF_NORMED
    )

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    score = (
        INTENSITY_WEIGHT * score_intensity
        +
        GRADIENT_WEIGHT * score_gradient
    )

    # --------------------------------------------------------
    # Find peaks
    # --------------------------------------------------------

    peaks = find_peaks(
        score,
        MATCH_THRESHOLD,
        PEAK_DISTANCE
    )

    detections = []

    for x, y, s in peaks:

        box = (
            x,
            y,
            x + tw,
            y + th
        )

        detections.append({

            "class": class_name,

            "score": float(s),

            "box": box,

            "x": x,

            "y": y,

            "width": tw,

            "height": th
        })

    return detections


# ============================================================
# FIND SECOND BEST CLASS
# ============================================================

def calculate_margin(
    detection,
    all_detections
):

    box = detection["box"]

    current_class = detection["class"]

    current_score = detection["score"]

    second_best = -1.0

    for other in all_detections:

        if other is detection:
            continue

        # Chỉ xét những detection nằm gần cùng vị trí
        iou = calculate_iou(
            box,
            other["box"]
        )

        if iou > 0.20:

            if other["class"] != current_class:

                if other["score"] > second_best:

                    second_best = other["score"]

    if second_best < 0:

        return 1.0

    return current_score - second_best


# ============================================================
# DRAW DETECTION
# ============================================================

def draw_detection(
    image,
    detection,
    status
):

    x1, y1, x2, y2 = detection["box"]

    label = detection["class"]

    score = detection["score"]

    # --------------------------------------------------------
    # Màu
    # --------------------------------------------------------
    #
    # AUTO   = xanh lá
    # REVIEW = vàng
    # REJECT = đỏ
    #

    if status == "AUTO":

        color = (
            0,
            255,
            0
        )

    elif status == "REVIEW":

        color = (
            0,
            255,
            255
        )

    else:

        color = (
            0,
            0,
            255
        )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = (
        f"{label} "
        f"{score:.2f}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65

    thickness = 2

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    text_y = max(
        y1,
        th + baseline + 2
    )

    cv2.rectangle(
        image,
        (
            x1,
            text_y - th - baseline - 4
        ),
        (
            x1 + tw + 5,
            text_y
        ),
        color,
        -1
    )

    cv2.putText(
        image,
        text,
        (
            x1 + 2,
            text_y - baseline - 2
        ),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# YOLO CONVERSION
# ============================================================

def box_to_yolo(
    box,
    image_width,
    image_height
):

    x1, y1, x2, y2 = box

    cx = (
        (x1 + x2) / 2
    ) / image_width

    cy = (
        (y1 + y2) / 2
    ) / image_height

    w = (
        x2 - x1
    ) / image_width

    h = (
        y2 - y1
    ) / image_height

    return cx, cy, w, h


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    templates,
    debug_dir,
    label_dir
):

    print()
    print("=" * 70)
    print(
        f"IMAGE: {image_path.name}"
    )
    print("=" * 70)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "[ERROR] Cannot read image"
        )

        return

    H, W = image.shape[:2]

    # --------------------------------------------------------
    # Preprocess input
    # --------------------------------------------------------

    gray = preprocess_gray(
        image
    )

    gradient = make_gradient(
        gray
    )

    # --------------------------------------------------------
    # Matching
    # --------------------------------------------------------

    all_detections = []

    for class_name, template in templates.items():

        detections = match_one_template(
            gray,
            gradient,
            template,
            class_name
        )

        all_detections.extend(
            detections
        )

    print(
        f"Raw candidates: "
        f"{len(all_detections)}"
    )

    # --------------------------------------------------------
    # Global NMS
    # --------------------------------------------------------

    detections = global_nms(
        all_detections,
        NMS_IOU_THRESHOLD
    )

    print(
        f"After NMS: "
        f"{len(detections)}"
    )

    # --------------------------------------------------------
    # Calculate margin
    # --------------------------------------------------------

    final_detections = []

    for detection in detections:

        margin = calculate_margin(
            detection,
            all_detections
        )

        detection["margin"] = margin

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if detection["score"] < REVIEW_THRESHOLD:

            status = "REJECT"

        elif (
            detection["score"] >= AUTO_THRESHOLD
            and
            margin >= MIN_SCORE_MARGIN
        ):

            status = "AUTO"

        else:

            status = "REVIEW"

        detection["status"] = status

        final_detections.append(
            detection
        )

    # --------------------------------------------------------
    # Draw debug
    # --------------------------------------------------------

    debug_image = image.copy()

    for detection in final_detections:

        draw_detection(
            debug_image,
            detection,
            detection["status"]
        )

    # --------------------------------------------------------
    # Save debug image
    # --------------------------------------------------------

    debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    debug_path = (
        debug_dir /
        image_path.name
    )

    cv2.imwrite(
        str(debug_path),
        debug_image
    )

    # --------------------------------------------------------
    # Save YOLO labels
    # --------------------------------------------------------

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_path = (
        label_dir /
        f"{image_path.stem}.txt"
    )

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as f:

        for detection in final_detections:

            # Không ghi reject
            if detection["status"] == "REJECT":
                continue

            class_name = detection["class"]

            # class ID
            class_id = CLASS_NAMES.index(
                class_name
            )

            cx, cy, w, h = box_to_yolo(
                detection["box"],
                W,
                H
            )

            f.write(
                f"{class_id} "
                f"{cx:.6f} "
                f"{cy:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()

    for i, detection in enumerate(
        final_detections,
        start=1
    ):

        x1, y1, x2, y2 = (
            detection["box"]
        )

        print(
            f"{i:02d}. "
            f"{detection['class']:>2} | "
            f"score={detection['score']:.3f} | "
            f"margin={detection['margin']:.3f} | "
            f"{detection['status']:>6} | "
            f"box=({x1},{y1},{x2},{y2})"
        )

    print()
    print(
        f"Debug : {debug_path}"
    )

    print(
        f"Label : {label_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    template_dir = Path(
        TEMPLATE_DIR
    )

    input_dir = Path(
        INPUT_DIR
    )

    output_dir = Path(
        OUTPUT_DIR
    )

    debug_dir = (
        output_dir /
        "debug"
    )

    label_dir = (
        output_dir /
        "labels"
    )

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not template_dir.exists():

        print(
            f"[ERROR] Template folder not found:\n"
            f"{template_dir}"
        )

        return

    if not input_dir.exists():

        print(
            f"[ERROR] Input folder not found:\n"
            f"{input_dir}"
        )

        return

    # --------------------------------------------------------
    # Load templates
    # --------------------------------------------------------

    templates = load_templates()

    if len(templates) == 0:

        print(
            "[ERROR] No templates!"
        )

        return

    y--------------------------------------------------------
    # Input images
    # --------------------------------------------------------

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    image_paths = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.suffix.lower()
            in extensions
        ]
    )

    print()
    print("=" * 70)
    print(
        f"INPUT IMAGES: "
        f"{len(image_paths)}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(image_paths)}]"
        )

        process_image(
            image_path,
            templates,
            debug_dir,
            label_dir
        )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL DONE")
    print("=" * 70)

    print(
        f"Debug images:\n"
        f"{debug_dir}"
    )

    print(
        f"\nYOLO labels:\n"
        f"{label_dir}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
