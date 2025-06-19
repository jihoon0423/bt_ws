#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import os

OUTPUT_NAME = "comparison_result_8.png"
CENTRAL_FRACTION = 0.3
GLOBAL_THRESH = 30
CENTRAL_THRESH = 20
MIN_AREA = 300
MIN_DENSITY = 0.05


def align_images(img_ref, img_to_align, max_features=500, good_match_percent=0.15):
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_align = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_align, None)

    if des1 is None or des2 is None:
        return img_to_align

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if not matches or len(matches) < 4:
        return img_to_align

    matches = sorted(matches, key=lambda x: x.distance)
    num_good = max(int(len(matches) * good_match_percent), 4)
    matches = matches[:num_good]

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_align = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 회전+이동+스케일 포함한 2D 아핀 행렬 추정
    M, inliers = cv2.estimateAffinePartial2D(pts_align, pts_ref)

    if M is None:
        return img_to_align

    h, w = img_ref.shape[:2]
    aligned = cv2.warpAffine(img_to_align, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return aligned




def detect_changes(img1, img2):
    img2_aligned = align_images(img1, img2)

    # Step 1: Difference
    diff = cv2.absdiff(img1, img2_aligned)
    diff = cv2.convertScaleAbs(diff, alpha=1.2, beta=0)
    cv2.imwrite("step1_diff_raw.png", diff)

    # Step 2: Gaussian blur on diff
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    cv2.imwrite("step2_diff_blurred.png", diff)

    # Step 3: Grayscale + blur
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("step3_gray_blurred.png", gray)

    # Step 4: Global threshold with Otsu
    ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    level = max(GLOBAL_THRESH, ret)
    _, global_th = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY)
    cv2.imwrite("step4_global_threshold.png", global_th)

    # Step 5: Central mask threshold
    h, w = gray.shape
    x1, x2 = int(w * CENTRAL_FRACTION), int(w * (1 - CENTRAL_FRACTION))
    y1, y2 = int(h * CENTRAL_FRACTION), int(h * (1 - CENTRAL_FRACTION))
    central = gray[y1:y2, x1:x2]
    _, cent_th = cv2.threshold(central, CENTRAL_THRESH, 255, cv2.THRESH_BINARY)
    central_mask = np.zeros_like(global_th)
    central_mask[y1:y2, x1:x2] = cent_th
    cv2.imwrite("step5_central_threshold.png", central_mask)

    # Step 6: Overlap mask
    mask_ref = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) > 0
    mask_al = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY) > 0
    overlap = (mask_ref & mask_al).astype(np.uint8) * 255
    cv2.imwrite("step6_overlap.png", overlap)

    # Step 7: Combined threshold + overlap
    combined = cv2.bitwise_or(global_th, central_mask)
    thresh = cv2.bitwise_and(combined, overlap)
    cv2.imwrite("step7_combined_thresh.png", thresh)

    # Step 8: Morphology
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ker, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, ker, iterations=1)
    cv2.imwrite("step8_morph_cleaned.png", clean)

    # Step 9: Contour analysis
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0
    best_box = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx, cy = x + bw // 2, y + bh // 2
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        roi = clean[y:y + bh, x:x + bw]
        dens = cv2.countNonZero(roi) / (bw * bh)
        if dens < MIN_DENSITY:
            continue
        score = dens * area
        if score > best_score:
            best_score = score
            best_box = (x, y, bw, bh)

    # Step 10: Draw result
    output = img2_aligned.copy()
    if best_box is not None:
        x, y, bw, bh = best_box
        cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    cv2.imwrite("step9_final_result_with_box.png", output)

    return output


def main():
    if len(sys.argv) != 3:
        print("Usage: compare_run_photos_final.py <ref> <curr>")
        sys.exit(1)

    p1, p2 = sys.argv[1], sys.argv[2]
    if not os.path.exists(p1) or not os.path.exists(p2):
        print("File not found")
        sys.exit(1)

    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)
    if img1 is None or img2 is None:
        print("Read error")
        sys.exit(1)

    res = detect_changes(img1, img2)
    cv2.imwrite(OUTPUT_NAME, res)
    print("Saved:", OUTPUT_NAME)


if __name__ == '__main__':
    main()
