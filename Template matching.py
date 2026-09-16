import cv2
import numpy as np
from pathlib import Path
import csv


# ============================================================
#                         CONFIG
# ============================================================

# ------------------------------------------------------------
# PATH
# ------------------------------------------------------------

TEMPLATE_DIR = r"D:\CharacterDataset\template"
INPUT_DIR = r"D:\CharacterDataset\input"
OUTPUT_DIR = r"D:\CharacterDataset\output"


# ------------------------------------------------------------
# IMAGE EXTENSIONS
# ------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
#                 TEMPLATE MATCHING SETTINGS
# ============================================================

# Trọng số giữa grayscale và gradient.
#
# Vì ký tự của bạn màu đen, tương phản thấp:
#
# intensity  -> thông tin mức xám
# gradient   -> hình dạng / biên ký tự
#
INTENSITY_WEIGHT = 0.40
GRADIENT_WEIGHT = 0.60


# ------------------------------------------------------------
# Raw template threshold
# ------------------------------------------------------------
#
# Đây là threshold khi lấy candidate từ từng template.
#
# Nên để tương đối thấp vì sau đó còn có bước:
#
# Multi-template aggregation
# + Global NMS
# + Margin
#
RAW_MATCH_THRESHOLD = 0.48


# ------------------------------------------------------------
# Khoảng cách tối thiểu giữa local maxima
# ------------------------------------------------------------

PEAK_DISTANCE = 20


# ============================================================
#             MULTI-TEMPLATE AGGREGATION
# ============================================================

# Số template tốt nhất được sử dụng để tính score class.
#
# Ví dụ class 3 có 50 template:
#
# 0.91
# 0.89
# 0.87
# 0.83
# 0.82
# ...
#
# TOP_K = 5
#
# => score dựa trên 5 template tốt nhất.
#
TOP_K = 5


# Trọng số giữa template tốt nhất và trung bình Top-K.
#
# final score =
#
# MAX_WEIGHT * max_score
# +
# TOPK_WEIGHT * mean_topk
#
MAX_WEIGHT = 0.35
TOPK_WEIGHT = 0.65


# ============================================================
#                     FINAL THRESHOLD
# ============================================================

# Score cuối >= AUTO_THRESHOLD
# và margin đủ lớn
# => AUTO
#
AUTO_THRESHOLD = 0.68


# Score cuối >= REVIEW_THRESHOLD
# => REVIEW
#
REVIEW_THRESHOLD = 0.55


# Nếu score class tốt nhất và class thứ 2 quá gần nhau
# => REVIEW
#
MIN_SCORE_MARGIN = 0.07


# ============================================================
#                         NMS
# ============================================================

# NMS giữa các class.
#
# Ví dụ cùng một vị trí:
#
# 3 = 0.88
# B = 0.71
#
# => giữ 3
#
GLOBAL_NMS_IOU = 0.25


# ============================================================
#             SAME CLASS CLUSTERING
# ============================================================

# Các template cùng một class nếu overlap lớn
# sẽ được xem là cùng một ký tự.
#
SAME_CLASS_IOU = 0.25


# ============================================================
#                PREPROCESSING SETTINGS
# ============================================================

USE_CLAHE = True

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

BLUR_KERNEL = 3


# ============================================================
#                       OUTPUT
# ============================================================

DEBUG_DIR_NAME = "debug"
LABEL_DIR_NAME = "labels"

# Lưu ảnh debug cho cả AUTO và REVIEW
SAVE_REVIEW_DEBUG = True


# ============================================================
#                 CLASS ORDER / CLASS ID
# ============================================================

def natural_class_sort(class_names):
    """
    Sắp xếp:
        0,1,2,...9,A,B,...Z

    thay vì:
        1,10,2,...
    """

    def key_func(x):

        x_upper = x.upper()

        if x_upper.isdigit():

            return (
                0,
                int(x_upper)
            )

        return (
            1,
            x_upper
        )

    return sorted(
        class_names,
        key=key_func
    )


# ============================================================
#                 PREPROCESS GRAYSCALE
# ============================================================

def preprocess_gray(img):
    """
    Tiền xử lý ảnh grayscale.

    1. Grayscale
    2. CLAHE
    3. Gaussian blur nhẹ
    """

    if img.ndim == 3:

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = img.copy()

    gray = gray.astype(
        np.uint8
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    if USE_CLAHE:

        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP,
            tileGridSize=CLAHE_GRID
        )

        gray = clahe.apply(gray)

    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    if BLUR_KERNEL >= 3:

        k = BLUR_KERNEL

        if k % 2 == 0:
            k += 1

        gray = cv2.GaussianBlur(
            gray,
            (k, k),
            0
        )

    return gray


# ============================================================
#                    GRADIENT IMAGE
# ============================================================

def make_gradient(gray):

    gray_f = gray.astype(
        np.float32
    )

    gx = cv2.Sobel(
        gray_f,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray_f,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    # Normalize về 0-255
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude.astype(
        np.uint8
    )


# ============================================================
#                     LOAD TEMPLATES
# ============================================================

def load_template_bank():

    template_root = Path(
        TEMPLATE_DIR
    )

    if not template_root.exists():

        raise FileNotFoundError(
            f"Template folder not found:\n"
            f"{template_root}"
        )

    # --------------------------------------------------------
    # Class folders
    # --------------------------------------------------------

    class_dirs = [
        p
        for p in template_root.iterdir()
        if p.is_dir()
    ]

    class_names = natural_class_sort(
        [
            p.name
            for p in class_dirs
        ]
    )

    if len(class_names) == 0:

        raise RuntimeError(
            "No class folders found!"
        )

    # --------------------------------------------------------
    # Class ID
    # --------------------------------------------------------

    class_to_id = {
        name: idx
        for idx, name in enumerate(
            class_names
        )
    }

    # --------------------------------------------------------
    # Template bank
    # --------------------------------------------------------

    template_bank = {}

    print()
    print("=" * 80)
    print("LOADING MULTI-TEMPLATE BANK")
    print("=" * 80)

    print()

    for class_name in class_names:

        class_dir = (
            template_root /
            class_name
        )

        image_paths = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.suffix.lower()
                in IMAGE_EXTENSIONS
            ]
        )

        template_list = []

        for template_path in image_paths:

            img = cv2.imread(
                str(template_path),
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:

                print(
                    f"[WARNING] Cannot read: "
                    f"{template_path}"
                )

                continue

            gray = preprocess_gray(
                img
            )

            gradient = make_gradient(
                gray
            )

            h, w = gray.shape

            template_list.append({

                "name":
                    template_path.name,

                "gray":
                    gray,

                "gradient":
                    gradient,

                "width":
                    w,

                "height":
                    h
            })

        template_bank[class_name] = (
            template_list
        )

        print(
            f"Class "
            f"{class_name:>3} "
            f"| ID = "
            f"{class_to_id[class_name]:>2} "
            f"| templates = "
            f"{len(template_list)}"
        )

    print()
    print(
        f"Total classes   : "
        f"{len(class_names)}"
    )

    print(
        f"Total templates : "
        f"{sum(len(v) for v in template_bank.values())}"
    )

    return (
        template_bank,
        class_names,
        class_to_id
    )


# ============================================================
#                       FIND PEAKS
# ============================================================

def find_peaks(
    score_map,
    threshold,
    min_distance
):

    size = max(
        3,
        int(min_distance)
    )

    if size % 2 == 0:
        size += 1

    kernel = np.ones(
        (size, size),
        np.uint8
    )

    local_max = cv2.dilate(
        score_map,
        kernel
    )

    mask = (
        (score_map >= threshold)
        &
        (score_map >= local_max - 1e-6)
    )

    ys, xs = np.where(
        mask
    )

    peaks = []

    for x, y in zip(xs, ys):

        peaks.append(
            (
                int(x),
                int(y),
                float(score_map[y, x])
            )
        )

    peaks.sort(
        key=lambda a: a[2],
        reverse=True
    )

    return peaks


# ============================================================
#                          IOU
# ============================================================

def box_iou(box1, box2):

    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    ix1 = max(
        x1,
        a1
    )

    iy1 = max(
        y1,
        b1
    )

    ix2 = min(
        x2,
        a2
    )

    iy2 = min(
        y2,
        b2
    )

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = (
        iw * ih
    )

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

    union = (
        area1
        +
        area2
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection /
        union
    )


# ============================================================
#                 MATCH ONE TEMPLATE
# ============================================================

def match_template(
    image_gray,
    image_gradient,
    template,
    class_name
):

    template_gray = (
        template["gray"]
    )

    template_gradient = (
        template["gradient"]
    )

    th, tw = (
        template_gray.shape
    )

    H, W = (
        image_gray.shape
    )

    if th > H or tw > W:

        return []

    # --------------------------------------------------------
    # Grayscale matching
    # --------------------------------------------------------

    score_gray = cv2.matchTemplate(
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
    # Combined
    # --------------------------------------------------------

    score_map = (
        INTENSITY_WEIGHT *
        score_gray
        +
        GRADIENT_WEIGHT *
        score_gradient
    )

    # --------------------------------------------------------
    # Peaks
    # --------------------------------------------------------

    peaks = find_peaks(
        score_map,
        RAW_MATCH_THRESHOLD,
        PEAK_DISTANCE
    )

    detections = []

    for x, y, score in peaks:

        box = (
            x,
            y,
            x + tw,
            y + th
        )

        detections.append({

            "class":
                class_name,

            "score":
                score,

            "box":
                box,

            "template":
                template["name"]
        })

    return detections


# ============================================================
#           CLUSTER DETECTIONS OF SAME CLASS
# ============================================================

def cluster_same_class(
    detections,
    iou_threshold
):
    """
    Gom các detection cùng class.

    Ví dụ class 3 có:

        3_template_1 -> 0.91
        3_template_2 -> 0.88
        3_template_7 -> 0.85
        3_template_9 -> 0.47

    Nếu chúng cùng một vị trí,
    chúng sẽ trở thành một cluster.
    """

    if len(detections) == 0:

        return []

    detections = sorted(
        detections,
        key=lambda d: d["score"],
        reverse=True
    )

    clusters = []

    for detection in detections:

        assigned = False

        for cluster in clusters:

            representative = (
                cluster["representative"]
            )

            iou = box_iou(
                detection["box"],
                representative["box"]
            )

            if iou >= iou_threshold:

                cluster["detections"].append(
                    detection
                )

                assigned = True

                break

        if not assigned:

            clusters.append({

                "representative":
                    detection,

                "detections":
                    [detection]
            })

    return clusters


# ============================================================
#             AGGREGATE MULTI-TEMPLATE SCORE
# ============================================================

def aggregate_cluster(
    cluster,
    top_k
):

    detections = sorted(
        cluster["detections"],
        key=lambda d: d["score"],
        reverse=True
    )

    scores = np.array(
        [
            d["score"]
            for d in detections
        ],
        dtype=np.float32
    )

    k = min(
        top_k,
        len(scores)
    )

    top_scores = scores[:k]

    max_score = float(
        top_scores[0]
    )

    topk_mean = float(
        np.mean(top_scores)
    )

    final_score = (
        MAX_WEIGHT * max_score
        +
        TOPK_WEIGHT * topk_mean
    )

    representative = detections[0]

    return {

        "class":
            representative["class"],

        "score":
            float(final_score),

        "max_score":
            max_score,

        "topk_mean":
            topk_mean,

        "num_support":
            len(detections),

        "top_scores":
            top_scores.tolist(),

        "box":
            representative["box"],

        "template":
            representative["template"]
    }


# ============================================================
#                PROCESS ONE CLASS
# ============================================================

def process_class(
    image_gray,
    image_gradient,
    class_name,
    templates
):

    raw_detections = []

    # --------------------------------------------------------
    # Match every template of this class
    # --------------------------------------------------------

    for template in templates:

        detections = match_template(
            image_gray,
            image_gradient,
            template,
            class_name
        )

        raw_detections.extend(
            detections
        )

    # --------------------------------------------------------
    # Cluster overlapping detections
    # --------------------------------------------------------

    clusters = cluster_same_class(
        raw_detections,
        SAME_CLASS_IOU
    )

    # --------------------------------------------------------
    # Aggregate each cluster
    # --------------------------------------------------------

    aggregated = []

    for cluster in clusters:

        result = aggregate_cluster(
            cluster,
            TOP_K
        )

        aggregated.append(
            result
        )

    return aggregated


# ============================================================
#                 GLOBAL NMS
# ============================================================

def global_nms(
    detections,
    iou_threshold
):

    detections = sorted(
        detections,
        key=lambda d: d["score"],
        reverse=True
    )

    selected = []

    for detection in detections:

        keep = True

        for selected_det in selected:

            iou = box_iou(
                detection["box"],
                selected_det["box"]
            )

            if iou >= iou_threshold:

                keep = False
                break

        if keep:

            selected.append(
                detection
            )

    return selected


# ============================================================
#             FIND SECOND BEST CLASS
# ============================================================

def calculate_margin(
    detection,
    detections
):

    best_score = detection[
        "score"
    ]

    best_box = detection[
        "box"
    ]

    second_score = -1.0

    for other in detections:

        if other is detection:
            continue

        if other["class"] == detection["class"]:
            continue

        iou = box_iou(
            best_box,
            other["box"]
        )

        # Chỉ coi là đối thủ nếu cùng vị trí
        if iou >= 0.20:

            if other["score"] > second_score:

                second_score = (
                    other["score"]
                )

    # Không có class cạnh tranh
    if second_score < 0:

        return 1.0

    return (
        best_score -
        second_score
    )


# ============================================================
#                     STATUS
# ============================================================

def get_status(
    score,
    margin
):

    if score < REVIEW_THRESHOLD:

        return "REJECT"

    if (
        score >= AUTO_THRESHOLD
        and
        margin >= MIN_SCORE_MARGIN
    ):

        return "AUTO"

    return "REVIEW"


# ============================================================
#                 DRAW DEBUG BOX
# ============================================================

def draw_detection(
    image,
    detection
):

    x1, y1, x2, y2 = (
        detection["box"]
    )

    status = detection[
        "status"
    ]

    class_name = detection[
        "class"
    ]

    score = detection[
        "score"
    ]

    margin = detection[
        "margin"
    ]

    support = detection[
        "num_support"
    ]
# --------------------------------------------------------
    # Color
    # --------------------------------------------------------

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
    # Box
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
        f"{class_name} "
        f"{score:.2f} "
        f"M:{margin:.2f} "
        f"N:{support}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.55

    thickness = 2

    (tw, th), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )
    )

    text_y = max(
        y1,
        th + baseline + 4
    )

    cv2.rectangle(
        image,
        (
            x1,
            text_y - th - baseline - 4
        ),
        (
            x1 + tw + 4,
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
#                    YOLO CONVERSION
# ============================================================

def box_to_yolo(
    box,
    image_width,
    image_height
):

    x1, y1, x2, y2 = box

    cx = (
        (x1 + x2) / 2.0
    ) / image_width

    cy = (
        (y1 + y2) / 2.0
    ) / image_height

    w = (
        x2 - x1
    ) / image_width

    h = (
        y2 - y1
    ) / image_height

    return (
        cx,
        cy,
        w,
        h
    )


# ============================================================
#                    SAVE YOLO LABEL
# ============================================================

def save_yolo_label(
    label_path,
    detections,
    class_to_id,
    image_width,
    image_height
):
    """
    Luôn tạo .txt.

    Nếu không có detection hợp lệ:
        file .txt vẫn được tạo nhưng rỗng.
    """

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as f:

        for detection in detections:

            # Không ghi REJECT
            if detection["status"] == "REJECT":
                continue

            class_name = detection[
                "class"
            ]

            class_id = class_to_id[
                class_name
            ]

            cx, cy, w, h = box_to_yolo(
                detection["box"],
                image_width,
                image_height
            )

            # ------------------------------------------------
            # Clamp tránh giá trị vượt [0,1]
            # ------------------------------------------------

            cx = np.clip(
                cx,
                0.0,
                1.0
            )

            cy = np.clip(
                cy,
                0.0,
                1.0
            )

            w = np.clip(
                w,
                0.0,
                1.0
            )

            h = np.clip(
                h,
                0.0,
                1.0
            )

            f.write(
                f"{class_id} "
                f"{cx:.6f} "
                f"{cy:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


# ============================================================
#                  SAVE CLASSES.TXT
# ============================================================

def save_classes(
    output_dir,
    class_names
):

    path = (
        output_dir /
        "classes.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for class_id, name in enumerate(
            class_names
        ):

            f.write(
                f"{class_id}: {name}\n"
            )

    print(
        f"\nClass mapping saved:\n"
        f"{path}"
    )


# ============================================================
#                    PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    template_bank,
    class_names,
    class_to_id,
    debug_dir,
    label_dir
):

    print()
    print("-" * 80)
    print(
        f"IMAGE: {image_path.name}"
    )
    print("-" * 80)

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "[ERROR] Cannot read image."
        )

        return {
            "image": image_path.name,
            "status": "READ_ERROR",
            "detections": 0,
            "auto": 0,
            "review": 0,
            "reject": 0
        }

    H, W = image.shape[:2]

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    gray = preprocess_gray(
        image
    )

    gradient = make_gradient(
        gray
    )

    # --------------------------------------------------------
    # Process every class
    # --------------------------------------------------------

    all_class_detections = []

    for class_name in class_names:

        templates = (
            template_bank[class_name]
        )

        if len(templates) == 0:
            continue

        detections = process_class(
            gray,
            gradient,
            class_name,
            templates
        )

        all_class_detections.extend(
            detections
        )

    print(
        f"Aggregated candidates: "
        f"{len(all_class_detections)}"
    )

    # --------------------------------------------------------
    # Global NMS
    # --------------------------------------------------------

    final_detections = global_nms(
        all_class_detections,
        GLOBAL_NMS_IOU
    )

    # --------------------------------------------------------
    # Calculate margin
    # --------------------------------------------------------

    for detection in final_detections:

        margin = calculate_margin(
            detection,
            all_class_detections
        )

        detection["margin"] = (
            float(margin)
        )

        detection["status"] = get_status(
            detection["score"],
            detection["margin"]
        )

    # --------------------------------------------------------
    # Sort spatially
    #
    # Trái -> phải
    # Trên -> dưới
    # --------------------------------------------------------

    final_detections.sort(
        key=lambda d: (
            d["box"][1],
            d["box"][0]
        )
    )

    # --------------------------------------------------------
    # Debug image
    # --------------------------------------------------------

    debug_image = image.copy()

    for detection in final_detections:

        if (
            detection["status"] == "REVIEW"
            and
            not SAVE_REVIEW_DEBUG
        ):
            continue

        draw_detection(
            debug_image,
            detection
        )

    # --------------------------------------------------------
    # Save debug
    # --------------------------------------------------------

    debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    debug_path = (
        debug_dir /
        image_path.name
    )

    success = cv2.imwrite(
        str(debug_path),
        debug_image
    )

    if not success:

        print(
            "[ERROR] Cannot save debug image:"
        )

        print(
            debug_path
        )

    # --------------------------------------------------------
    # Save YOLO TXT
    # --------------------------------------------------------

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_path = (
        label_dir /
        f"{image_path.stem}.txt"
    )

    save_yolo_label(
        label_path,
        final_detections,
        class_to_id,
        W,
        H
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    auto_count = sum(
        d["status"] == "AUTO"
        for d in final_detections
    )

    review_count = sum(
        d["status"] == "REVIEW"
        for d in final_detections
    )

    reject_count = sum(
        d["status"] == "REJECT"
        for d in final_detections
    )

    # --------------------------------------------------------
    # Print
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
            f"{detection['class']:>3} | "
            f"score={detection['score']:.3f} | "
            f"margin={detection['margin']:.3f} | "
            f"support={detection['num_support']:>3} | "
            f"{detection['status']:>6} | "
            f"box=({x1},{y1},{x2},{y2})"
        )

    print()

    print(
        f"Final detections : "
        f"{len(final_detections)}"
    )

    print(
        f"AUTO              : "
        f"{auto_count}"
    )

    print(
        f"REVIEW             : "
        f"{review_count}"
    )

    print(
        f"REJECT             : "
        f"{reject_count}"
    )

    print(
        f"Debug -> "
        f"{debug_path}"
    )

    print(
        f"Label -> "
        f"{label_path}"
    )

    return {

        "image":
            image_path.name,

        "status":
            "OK",

        "detections":
            len(final_detections),

        "auto":
            auto_count,

        "review":
            review_count,

        "reject":
            reject_count
    }


# ============================================================
#                       SAVE CSV
# ============================================================

def save_results_csv(
    output_dir,
    results
):

    csv_path = (
        output_dir /
        "results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "status",
                "detections",
                "auto",
                "review",
                "reject"
            ]
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f"\nResults CSV saved:\n"
        f"{csv_path}"
    )


# ============================================================
#                           MAIN
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
        DEBUG_DIR_NAME
    )

    label_dir = (
        output_dir /
        LABEL_DIR_NAME
    )

    # --------------------------------------------------------
    # Check folders
    # --------------------------------------------------------

    if not template_dir.exists():

        raise FileNotFoundError(
            f"Template folder does not exist:\n"
            f"{template_dir}"
        )

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input folder does not exist:\n"
            f"{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load template bank
    # --------------------------------------------------------

    (
        template_bank,
        class_names,
        class_to_id
    ) = load_template_bank()

    # --------------------------------------------------------
    # Save class mapping
    # --------------------------------------------------------

    save_classes(
        output_dir,
        class_names
    )

    # --------------------------------------------------------
    # Input images
    # --------------------------------------------------------

    image_paths = sorted(
        [
            p
            for p in input_dir.iterdir()
            if (
                p.is_file()
                and
                p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    print()
    print("=" * 80)
    print(
        f"INPUT IMAGES: "
        f"{len(image_paths)}"
    )
    print("=" * 80)

    if len(image_paths) == 0:

        print(
            "No images found."
        )

        return

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    results = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(image_paths)}]"
        )

        result = process_image(
            image_path,
            template_bank,
            class_names,
            class_to_id,
            debug_dir,
            label_dir
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    save_results_csv(
        output_dir,
        results
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_images = len(
        results
    )

    total_detection = sum(
        r["detections"]
        for r in results
    )

    total_auto = sum(
        r["auto"]
        for r in results
    )

    total_review = sum(
        r["review"]
        for r in results
    )

    total_reject = sum(
        r["reject"]
        for r in results
    )

    print()
    print("=" * 80)
    print("                    COMPLETE")
    print("=" * 80)

    print(
        f"Images processed : "
        f"{total_images}"
    )

    print(
        f"Detections       : "
        f"{total_detection}"
    )

    print(
        f"AUTO             : "
        f"{total_auto}"
    )

    print(
        f"REVIEW           : "
        f"{total_review}"
    )

    print(
        f"REJECT           : "
        f"{total_reject}"
    )

    print()
    print(
        f"Debug folder:\n"
        f"{debug_dir}"
    )

    print()
    print(
        f"YOLO labels:\n"
        f"{label_dir}"
    )

    print()
    print(
        f"Class mapping:\n"
        f"{output_dir / 'classes.txt'}"
    )


# ============================================================
#                           RUN
# ============================================================

if __name__ == "__main__":

    main()
