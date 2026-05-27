import cv2
import numpy as np


def find_template_multiscale(
        frame,
        template_base,
        threshold=0.8,
        roi_pct=None,  # (x1, y1, x2, y2), 0..1
        scales=None,
        min_height_pct=0.03,
        max_height_pct=0.08,
        scale_steps=12,
        method=cv2.TM_CCOEFF_NORMED,
        nms_overlap=0.3,
        mask_base=None,
        mae_threshold=28,
        quality_threshold=0.78
):
    if roi_pct is not None:
        rx, ry, rw, rh = percent_roi_to_pixels(frame.shape, roi_pct)
        search_img = frame[ry:ry + rh, rx:rx + rw]
    else:
        rx, ry = 0, 0
        search_img = frame

    if scales is None:
        scales = make_scales_for_frame(
            frame.shape,
            template_base.shape,
            min_height_pct=min_height_pct,
            max_height_pct=max_height_pct,
            steps=scale_steps,
        )

    candidates = []

    for scale in scales:
        template = cv2.resize(template_base, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        th, tw = template.shape[:2]

        if th < 3 or tw < 3:
            continue

        if th > search_img.shape[0] or tw > search_img.shape[1]:
            continue

        mask = None

        if mask_base is not None:
            mask = cv2.resize(
                mask_base,
                (tw, th),
                interpolation=cv2.INTER_NEAREST,
            )

            visible = cv2.countNonZero(mask)

            # require enough visible pixels AND enough visible area
            if visible < 10:
                continue

            if visible / float(tw * th) < 0.15:
                continue

        result = cv2.matchTemplate(search_img, template, method, mask=mask)

        # ys, xs = np.where(result >= threshold)

        # local maxima only
        kernel = np.ones((3, 3), np.uint8)
        local_max = result == cv2.dilate(result, kernel)

        ys, xs = np.where((result >= threshold) & local_max)

        for x, y in zip(xs, ys):
            full_x = x + rx
            full_y = y + ry

            patch = search_img[y:y + th, x:x + tw]

            mae = 0
            if mask_base is not None:
                # mae = masked_mae(patch, template, mask)
                mae = edge_ncc(patch, template, mask)

                # print(mae, float(result[y, x]), (th, tw))

                if mae < mae_threshold:  # NCC
                    continue
                # print('allow---')

            candidates.append({
                "score": float(result[y, x]),
                "mae": float(mae),
                "loc": (full_x, full_y),
                "scale": float(scale),
                "shape": (th, tw),
                "box": (full_x, full_y, tw, th),
            })

    # print(f'{mae_threshold=}')
    # for c in sorted(candidates, key=lambda k: k["mae"], reverse=True):
    #     # if c['mae'] > mae_threshold:
    #     #     continue
    #     print(c)

    return non_max_suppression(candidates, nms_overlap)


def masked_mae(patch, template, mask):
    m = mask > 0

    diff = np.abs(
        patch.astype(np.int16) - template.astype(np.int16)
    )

    # if 3-channel image and 2D mask
    return diff[m].mean()


from skimage.metrics import structural_similarity as ssim


def masked_ssim_gray(patch, template, mask, size=(24, 24)):
    p = normalized_gray(patch, size)
    t = normalized_gray(template, size)

    m = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0
    # m = mask

    if not np.any(m):
        return -1.0

    score_map = ssim(
        p,
        t,
        data_range=255,
        full=True,
    )[1]

    return float(score_map[m].mean())


# def normalized_gray(img, size=(24, 24)):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#
#     # normalize brightness/contrast
#     gray = cv2.equalizeHist(gray)
#
#     return gray

# def normalized_gray(img, size=(24, 24)):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#
#     gray = gray.astype(np.float32)
#
#     mean = gray.mean()
#     std = gray.std()
#
#     if std < 1e-6:
#         return gray * 0
#
#     gray = (gray - mean) / std
#     return gray


def edge_ncc(patch, template, mask=None):
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    ksize = 3
    p_edge = cv2.Sobel(patch, cv2.CV_32F, 1, 1, ksize=ksize)
    t_edge = cv2.Sobel(template, cv2.CV_32F, 1, 1, ksize=ksize)

    if mask is not None:
        p_edge = p_edge * (mask / 255.0)
        t_edge = t_edge * (mask / 255.0)

    # NCC
    num = np.sum(p_edge * t_edge)
    denom = np.sqrt(np.sum(p_edge ** 2) * np.sum(t_edge ** 2)) + 1e-6
    return num / denom


def normalized_gray(img, size=(24, 24)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)

    mean = mag.mean()
    std = mag.std()

    if std < 1e-6:
        return mag * 0

    return (mag - mean) / std


def masked_mae_gray(patch, template, mask, size=(24, 24)):
    p = normalized_gray(patch, size)
    t = normalized_gray(template, size)

    m = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0

    if not np.any(m):
        return float("inf")

    diff = np.abs(p.astype(np.int16) - t.astype(np.int16))
    return float(diff[m].mean())


def percent_roi_to_pixels(frame_shape, roi_pct):
    """
    roi_pct = (x1, y1, x2, y2), values from 0.0 to 1.0
    Example: (0.1, 0.2, 0.9, 0.8)
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi_pct

    x1 = int(round(x1 * w))
    y1 = int(round(y1 * h))
    x2 = int(round(x2 * w))
    y2 = int(round(y2 * h))

    return x1, y1, x2 - x1, y2 - y1


def make_scales_for_frame(
        frame_shape,
        template_shape,
        min_height_pct=0.03,
        max_height_pct=0.08,
        steps=12,
):
    frame_h = frame_shape[0]
    template_h = template_shape[0]

    min_target_h = frame_h * min_height_pct
    max_target_h = frame_h * max_height_pct

    min_scale = min_target_h / template_h
    max_scale = max_target_h / template_h

    return np.linspace(min_scale, max_scale, steps)


def non_max_suppression(matches, overlap_thresh=0.3):
    if not matches:
        return []

    boxes = np.array([m["box"] for m in matches], dtype=np.float32)
    scores = np.array([m["score"] for m in matches], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        overlap = inter / areas[order[1:]]

        order = order[1:][overlap <= overlap_thresh]

    return [matches[i] for i in keep]
