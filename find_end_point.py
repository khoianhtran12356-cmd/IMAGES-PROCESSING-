# -*- coding: utf-8 -*-
"""
Tìm điểm END (góc dưới-phải của vùng nhựa đen) dựa trên cạnh FG
của khung đồng (copper) bên trái ảnh AOI.

Ý tưởng:
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
  6. END = F + fg_to_end_dist * vector_vuông_góc_đơn_vị.
  7. Vì ROI chỉ cắt theo trục x bắt đầu từ 0 (x_offset=0, y_offset=0)
     nên tọa độ tính được đã nằm trên hệ tọa độ ảnh gốc. Code vẫn
     giữ offset để tổng quát hoá khi bạn đổi vùng crop sau này.
"""

import cv2
import numpy as np


def find_end_point(
    image_path: str,
    roi_width: int = 1500,
    fg_to_end_dist: float = 4454.0,
    bright_thresh: int | None = None,
    debug_save_path: str | None = None,
):
    """
    Trả về dict chứa toạ độ F, G, END (trên hệ toạ độ ảnh GỐC),
    danh sách đỉnh đa giác đã detect, và mask nhị phân dùng để debug.
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

    # ---- 5. Vector FG và vector vuông góc hướng vào ảnh (+x) ----
    fg_vec = G - F
    fg_len = np.linalg.norm(fg_vec)
    if fg_len < 1e-6:
        raise RuntimeError("Cạnh FG suy biến (F trùng G), kiểm tra lại threshold.")
    fg_unit = fg_vec / fg_len

    perp1 = np.array([-fg_unit[1], fg_unit[0]])
    perp2 = np.array([fg_unit[1], -fg_unit[0]])
    perp = perp1 if perp1[0] > perp2[0] else perp2  # chọn hướng +x

    # ---- 6. Tính điểm END ----
    end_point = F + perp * fg_to_end_dist

    # ---- 7. Map lại toạ độ ảnh gốc (ROI chỉ cắt theo x, offset=0) ----
    F_full = (F[0] + x_offset, F[1] + y_offset)
    G_full = (G[0] + x_offset, G[1] + y_offset)
    end_full = (end_point[0] + x_offset, end_point[1] + y_offset)
    polygon_full = pts + np.array([x_offset, y_offset])

    result = {
        "F": F_full,
        "G": G_full,
        "END": end_full,
        "polygon": polygon_full,
        "mask": mask,
    }

    if debug_save_path:
        vis = img.copy()
        Fi = tuple(map(int, F_full))
        Gi = tuple(map(int, G_full))
        Ei = tuple(map(int, end_full))
        for p in polygon_full.astype(int):
            cv2.circle(vis, tuple(p), 10, (255, 200, 0), -1)
        cv2.circle(vis, Fi, 18, (255, 0, 0), -1)      # F: xanh dương
        cv2.circle(vis, Gi, 18, (0, 255, 0), -1)      # G: xanh lá
        cv2.circle(vis, Ei, 22, (0, 0, 255), -1)      # END: đỏ
        cv2.line(vis, Fi, Gi, (255, 255, 0), 4)
        cv2.line(vis, Fi, Ei, (0, 255, 255), 4)
        cv2.imwrite(debug_save_path, vis)

    return result


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    result = find_end_point(image_path, debug_save_path="output_end_point.jpg")

    print("F  :", result["F"])
    print("G  :", result["G"])
    print("END:", result["END"])
    print("Số đỉnh đa giác detect được:", len(result["polygon"]))
    print("Đã lưu ảnh debug: output_end_point.jpg")

# -*- coding: utf-8 -*-
"""
Tìm điểm END (góc dưới-phải của vùng nhựa đen) dựa trên cạnh FG
của khung đồng (copper) bên trái ảnh AOI.

Ý tưởng:
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
  6. END = F + fg_to_end_dist * vector_vuông_góc_đơn_vị.
  7. Vì ROI chỉ cắt theo trục x bắt đầu từ 0 (x_offset=0, y_offset=0)
     nên tọa độ tính được đã nằm trên hệ tọa độ ảnh gốc. Code vẫn
     giữ offset để tổng quát hoá khi bạn đổi vùng crop sau này.
"""

from pathlib import Path

import cv2
import numpy as np


def find_end_point(
    image_path: str,
    roi_width: int = 1500,
    fg_to_end_dist: float = 4454.0,
    bright_thresh: int | None = None,
    debug_save_path: str | None = None,
):
    """
    Trả về dict chứa toạ độ F, G, END (trên hệ toạ độ ảnh GỐC),
    danh sách đỉnh đa giác đã detect, và mask nhị phân dùng để debug.
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

    # ---- 5. Vector FG và vector vuông góc hướng vào ảnh (+x) ----
    fg_vec = G - F
    fg_len = np.linalg.norm(fg_vec)
    if fg_len < 1e-6:
        raise RuntimeError("Cạnh FG suy biến (F trùng G), kiểm tra lại threshold.")
    fg_unit = fg_vec / fg_len

    perp1 = np.array([-fg_unit[1], fg_unit[0]])
    perp2 = np.array([fg_unit[1], -fg_unit[0]])
    perp = perp1 if perp1[0] > perp2[0] else perp2  # chọn hướng +x

    # ---- 6. Tính điểm END ----
    end_point = F + perp * fg_to_end_dist

    # ---- 7. Map lại toạ độ ảnh gốc (ROI chỉ cắt theo x, offset=0) ----
    F_full = (F[0] + x_offset, F[1] + y_offset)
    G_full = (G[0] + x_offset, G[1] + y_offset)
    end_full = (end_point[0] + x_offset, end_point[1] + y_offset)
    polygon_full = pts + np.array([x_offset, y_offset])

    result = {
        "F": F_full,
        "G": G_full,
        "END": end_full,
        "polygon": polygon_full,
        "mask": mask,
    }

    if debug_save_path:
        vis = img.copy()
        Fi = tuple(map(int, F_full))
        Gi = tuple(map(int, G_full))
        Ei = tuple(map(int, end_full))
        for p in polygon_full.astype(int):
            cv2.circle(vis, tuple(p), 10, (255, 200, 0), -1)
        cv2.circle(vis, Fi, 18, (255, 0, 0), -1)      # F: xanh dương
        cv2.circle(vis, Gi, 18, (0, 255, 0), -1)      # G: xanh lá
        cv2.circle(vis, Ei, 22, (0, 0, 255), -1)      # END: đỏ
        cv2.line(vis, Fi, Gi, (255, 255, 0), 4)
        cv2.line(vis, Fi, Ei, (0, 255, 255), 4)
        cv2.imwrite(debug_save_path, vis)

    return result


def process_folder(
    input_dir: str,
    debug_dir: str,
    roi_width: int = 1500,
    fg_to_end_dist: float = 4454.0,
    bright_thresh: int | None = None,
    extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
):
    """
    Chạy find_end_point cho toàn bộ ảnh trong input_dir, lưu ảnh debug
    (chấm đỏ điểm END trên ảnh gốc) vào debug_dir, và trả về danh sách
    kết quả (kèm lỗi nếu có) cho từng ảnh. Đồng thời ghi log CSV
    "results.csv" trong debug_dir để tiện đối chiếu hàng loạt.
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
        f.write("file,status,F_x,F_y,G_x,G_y,END_x,END_y,num_vertices,error\n")

        for img_path in image_paths:
            debug_path = out_dir / f"{img_path.stem}_debug{img_path.suffix}"
            try:
                res = find_end_point(
                    str(img_path),
                    roi_width=roi_width,
                    fg_to_end_dist=fg_to_end_dist,
                    bright_thresh=bright_thresh,
                    debug_save_path=str(debug_path),
                )
                Fx, Fy = res["F"]
                Gx, Gy = res["G"]
                Ex, Ey = res["END"]
                n_pts = len(res["polygon"])
                f.write(
                    f"{img_path.name},OK,{Fx:.1f},{Fy:.1f},"
                    f"{Gx:.1f},{Gy:.1f},{Ex:.1f},{Ey:.1f},{n_pts},\n"
                )
                print(f"[OK]  {img_path.name}  ->  END={Ex:.1f},{Ey:.1f}")
                results.append({"file": img_path.name, "status": "OK", **res})
            except Exception as e:
                f.write(f"{img_path.name},FAIL,,,,,,,,{e}\n")
                print(f"[FAIL] {img_path.name}  ->  {e}")
                results.append(
                    {"file": img_path.name, "status": "FAIL", "error": str(e)}
                )

    print(f"\nĐã xử lý {len(image_paths)} ảnh.")
    print(f"Ảnh debug (chấm đỏ điểm END): {out_dir}")
    print(f"Bảng kết quả: {csv_path}")
    return results


if __name__ == "__main__":
    import sys

    # Cách dùng:
    #   1 ảnh:    python3 find_end_point.py duong_dan_anh.jpg
    #   thư mục:  python3 find_end_point.py duong_dan_thu_muc_input duong_dan_thu_muc_debug
    if len(sys.argv) >= 3:
        input_dir, debug_dir = sys.argv[1], sys.argv[2]
        process_folder(input_dir, debug_dir)
    else:
        image_path = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
        result = find_end_point(image_path, debug_save_path="output_end_point.jpg")

        print("F  :", result["F"])
        print("G  :", result["G"])
        print("END:", result["END"])
        print("Số đỉnh đa giác detect được:", len(result["polygon"]))
        print("Đã lưu ảnh debug: output_end_point.jpg")
      
