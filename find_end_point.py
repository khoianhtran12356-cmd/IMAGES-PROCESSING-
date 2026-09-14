# -*- coding: utf-8 -*-
"""
B1. Tìm điểm END (X) - góc dưới-phải vùng nhựa đen - dựa trên cạnh FG
    của khung đồng (copper) bên trái ảnh AOI.
B2. Dựng hệ toạ độ cục bộ "GFX": gốc F, trục Fy hướng về phía G,
    trục Fx hướng về phía X (END) - hai trục này vuông góc với nhau
    (đã đảm bảo từ bước tính END).
B3. Với điểm P cho trước bằng toạ độ cục bộ (Fx, Fy) trong hệ GFX,
    ánh xạ P về hệ toạ độ ảnh gốc, sau đó cắt / debug vùng ROI vuông
    224x224 (axis-aligned trên ảnh gốc) quanh P.

Ý tưởng tìm END (giữ nguyên, đã verify đúng):
  1. Ngưỡng ảnh để tách vùng đồng (sáng) khỏi nền/nhựa (tối).
  2. Cắt ROI bên trái ảnh (mặc định 1500px, full chiều cao) vì
     toàn bộ đa giác ABCDEFGH nằm gọn trong vùng này.
  3. Tìm contour lớn nhất trong ROI (khung đồng), approxPolyDP để
     lấy các đỉnh đa giác.
  4. F, G là 2 đỉnh có x nhỏ nhất (cạnh ngoài cùng bên trái).
     - F: y lớn hơn (điểm dưới)
     - G: y nhỏ hơn (điểm trên)
  5. Vector FG -> vector vuông góc (chọn hướng +x, tức là hướng
     vào trong ảnh / về phía vùng đen).
  6. END (X) = F + fg_to_end_dist * vector_vuông_góc_đơn_vị.
"""

from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# CẤU HÌNH — chỉnh trực tiếp các giá trị dưới đây rồi chạy file, KHÔNG cần
# truyền tham số dòng lệnh.
# =============================================================================

# Thư mục chứa ảnh input (đổi thành đường dẫn thật của bạn)
INPUT_DIR = r"C:\path\to\anh_input"

# Thư mục sẽ lưu ảnh debug ROI của điểm P + file results.csv
DEBUG_DIR = r"C:\path\to\anh_debug"

# Bề rộng vùng ROI bên trái để tìm khung đồng ABCDEFGH (mặc định 1500px)
ROI_WIDTH = 1500

# Khoảng cách từ F đến END theo phương vuông góc với FG (px)
FG_TO_END_DIST = 4454.0

# Ngưỡng sáng cố định để tách vùng đồng, để None nếu muốn tự động (Otsu)
BRIGHT_THRESH = None

# Toạ độ điểm P trong hệ toạ độ cục bộ GFX: (Fx, Fy)
#   - Fx: khoảng cách từ F theo trục hướng về phía X (END)
#   - Fy: khoảng cách từ F theo trục hướng về phía G
P_LOCAL = (4244.0, 370.0)

# Kích thước cạnh vùng ROI vuông cần cắt quanh P (px)
ROI_SIZE = 224
# =============================================================================


def find_end_point(
    image_path: str,
    roi_width: int = 1500,
    fg_to_end_dist: float = 4454.0,
    bright_thresh: int | None = None,
):
    """
    Trả về dict chứa toạ độ F, G, END (trên hệ toạ độ ảnh GỐC) cùng hai
    vector đơn vị của hệ toạ độ cục bộ GFX:
      - e_y (fg_unit): hướng từ F về G
      - e_x (perp)   : hướng từ F về END (X)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- 1 & 2. Cắt ROI trái và ngưỡng ảnh ----
    x_offset, y_offset = 0, 0
    roi_width = min(roi_width, w)
    roi = gray[:, x_offset : x_offset + roi_width]

    # Cân bằng sáng nhẹ trước khi threshold, giúp ổn định hơn với
    # ảnh AOI chiếu sáng không đều
    roi_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)

    if bright_thresh is None:
        _, mask = cv2.threshold(
            roi_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, mask = cv2.threshold(roi_eq, bright_thresh, 255, cv2.THRESH_BINARY)

    # Dọn nhiễu
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # ---- 3. Tìm contour khung đồng ----
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("Không tìm thấy vùng đồng (copper) trong ROI trái.")

    cnt = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(cnt, True)
    approx = None
    # thử nhiều epsilon để cố gắng ra đúng ~8 đỉnh (A..H)
    for eps_factor in (0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03):
        eps = eps_factor * peri
        candidate = cv2.approxPolyDP(cnt, eps, True)
        if 6 <= len(candidate) <= 10:
            approx = candidate
            break
    if approx is None:
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

    pts = approx.reshape(-1, 2).astype(np.float64)

    # ---- 4. Xác định F, G: 2 đỉnh có x nhỏ nhất ----
    order_by_x = pts[np.argsort(pts[:, 0])]
    left_two = order_by_x[:2]
    if left_two[0][1] > left_two[1][1]:
        F, G = left_two[0], left_two[1]
    else:
        F, G = left_two[1], left_two[0]

    # ---- 5. Vector FG (trục Fy) và vector vuông góc hướng vào ảnh (trục Fx) ----
    fg_vec = G - F
    fg_len = np.linalg.norm(fg_vec)
    if fg_len < 1e-6:
        raise RuntimeError("Cạnh FG suy biến (F trùng G), kiểm tra lại threshold.")
    fg_unit = fg_vec / fg_len  # e_y: hướng F -> G

    perp1 = np.array([-fg_unit[1], fg_unit[0]])
    perp2 = np.array([fg_unit[1], -fg_unit[0]])
    perp = perp1 if perp1[0] > perp2[0] else perp2  # e_x: hướng F -> X (END)

    # ---- 6. Tính điểm END (X) ----
    end_point = F + perp * fg_to_end_dist

    # ---- 7. Map lại toạ độ ảnh gốc (ROI chỉ cắt theo x, offset=0) ----
    F_full = F + np.array([x_offset, y_offset])
    G_full = G + np.array([x_offset, y_offset])
    end_full = end_point + np.array([x_offset, y_offset])
    polygon_full = pts + np.array([x_offset, y_offset])

    return {
        "F": F_full,
        "G": G_full,
        "END": end_full,
        "e_x": perp,      # hướng F -> X trong ảnh gốc
        "e_y": fg_unit,   # hướng F -> G trong ảnh gốc
        "polygon": polygon_full,
        "mask": mask,
    }


def local_to_image_coords(F, e_x, e_y, p_local):
    """
    Ánh xạ điểm P từ hệ toạ độ cục bộ GFX (gốc F, trục Fx theo e_x,
    trục Fy theo e_y) về hệ toạ độ ảnh gốc.
    """
    px, py = p_local
    F = np.asarray(F, dtype=np.float64)
    e_x = np.asarray(e_x, dtype=np.float64)
    e_y = np.asarray(e_y, dtype=np.float64)
    return F + px * e_x + py * e_y


def get_square_roi_box(center, size, img_w, img_h):
    """
    Trả về (x1, y1, x2, y2) của vùng vuông axis-aligned kích thước
    size x size, tâm tại `center`, đã clip trong biên ảnh. Nếu tâm
    quá gần biên, box sẽ được dịch vào trong để vẫn giữ đủ size x size
    (miễn là ảnh đủ lớn); trả kèm cờ `truncated` nếu ảnh nhỏ hơn size.
    """
    cx, cy = center
    half = size / 2.0

    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    # dịch box vào trong biên ảnh nếu tràn ra ngoài (vẫn giữ đúng size)
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    truncated = x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, img_w), min(y2, img_h)

    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), truncated


def process_folder(
    input_dir: str,
    debug_dir: str,
    roi_width: int = 1500,
    fg_to_end_dist: float = 4454.0,
    bright_thresh: int | None = None,
    p_local=(4244.0, 370.0),
    roi_size: int = 224,
    extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
):
    """
    Với mỗi ảnh trong input_dir:
      1. Tìm F, G, END (X) và dựng hệ toạ độ cục bộ GFX.
      2. Ánh xạ điểm P (cho theo toạ độ cục bộ p_local) về ảnh gốc.
      3. Cắt vùng ROI vuông roi_size x roi_size quanh P trên ảnh gốc,
         lưu riêng thành "<ten_anh>_roi.jpg".
      4. Vẽ khung ROI + tâm P lên toàn bộ ảnh gốc để debug, lưu thành
         "<ten_anh>_roi_debug.jpg".
    Ghi log toạ độ vào debug_dir/results.csv.
    """
    in_dir = Path(input_dir)
    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in in_dir.iterdir() if p.suffix.lower() in extensions
    )
    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong: {in_dir}")
        return []

    results = []
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "file,status,F_x,F_y,G_x,G_y,END_x,END_y,"
            "P_x,P_y,roi_x1,roi_y1,roi_x2,roi_y2,truncated,error\n"
        )

        for img_path in image_paths:
            roi_debug_path = out_dir / f"{img_path.stem}_roi_debug{img_path.suffix}"
            roi_crop_path = out_dir / f"{img_path.stem}_roi{img_path.suffix}"
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
                h, w = img.shape[:2]

                res = find_end_point(
                    str(img_path),
                    roi_width=roi_width,
                    fg_to_end_dist=fg_to_end_dist,
                    bright_thresh=bright_thresh,
                )
                F, e_x, e_y = res["F"], res["e_x"], res["e_y"]

                # ---- Ánh xạ P từ hệ toạ độ cục bộ GFX về ảnh gốc ----
                P_img = local_to_image_coords(F, e_x, e_y, p_local)

                # ---- Cắt ROI vuông quanh P ----
                x1, y1, x2, y2, truncated = get_square_roi_box(
                    P_img, roi_size, w, h
                )
                roi_crop = img[y1:y2, x1:x2]
                cv2.imwrite(str(roi_crop_path), roi_crop)

                # ---- Vẽ debug ROI lên ảnh gốc ----
                vis = img.copy()
                Pi = tuple(int(round(v)) for v in P_img)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.drawMarker(
                    vis, Pi, (0, 0, 255), markerType=cv2.MARKER_CROSS,
                    markerSize=30, thickness=4,
                )
                cv2.imwrite(str(roi_debug_path), vis)

                f.write(
                    f"{img_path.name},OK,"
                    f"{F[0]:.1f},{F[1]:.1f},"
                    f"{res['G'][0]:.1f},{res['G'][1]:.1f},"
                    f"{res['END'][0]:.1f},{res['END'][1]:.1f},"
                    f"{P_img[0]:.1f},{P_img[1]:.1f},"
                    f"{x1},{y1},{x2},{y2},{truncated},\n"
                )
                print(
                    f"[OK]  {img_path.name}  ->  P(ảnh gốc)="
                    f"{P_img[0]:.1f},{P_img[1]:.1f}  ROI=({x1},{y1})-({x2},{y2})"
                    f"{'  [TRUNCATED]' if truncated else ''}"
                )
                results.append(
                    {
                        "file": img_path.name,
                        "status": "OK",
                        "P": tuple(P_img),
                        "roi_box": (x1, y1, x2, y2),
                        **res,
                    }
                )
            except Exception as e:
                f.write(f"{img_path.name},FAIL,,,,,,,,,,,,,,{e}\n")
                print(f"[FAIL] {img_path.name}  ->  {e}")
                results.append(
                    {"file": img_path.name, "status": "FAIL", "error": str(e)}
                )

    print(f"\nĐã xử lý {len(image_paths)} ảnh.")
    print(f"Ảnh debug ROI + ảnh crop ROI: {out_dir}")
    print(f"Bảng kết quả: {csv_path}")
    return results


if __name__ == "__main__":
    # Chạy trực tiếp theo cấu hình khai báo ở đầu file (INPUT_DIR, DEBUG_DIR, ...)
    process_folder(
        input_dir=INPUT_DIR,
        debug_dir=DEBUG_DIR,
        roi_width=ROI_WIDTH,
        fg_to_end_dist=FG_TO_END_DIST,
        bright_thresh=BRIGHT_THRESH,
        p_local=P_LOCAL,
        roi_size=ROI_SIZE,
    )
  
