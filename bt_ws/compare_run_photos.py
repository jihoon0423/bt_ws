#!/usr/bin/env python3

import sys
import cv2
import numpy as np


def align_images(img_ref, img_to_align, max_features=500, good_match_percent=0.15):
    gray_ref   = cv2.cvtColor(img_ref,    cv2.COLOR_BGR2GRAY)
    gray_align = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(gray_ref,    None)
    kp2, des2 = orb.detectAndCompute(gray_align, None)
    if des1 is None or des2 is None:
        return img_to_align

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if not matches:
        return img_to_align
    matches = sorted(matches, key=lambda x: x.distance)

    num_good = max(int(len(matches) * good_match_percent), 4)
    matches = matches[:num_good]

    pts_ref   = np.zeros((len(matches), 2), dtype=np.float32)
    pts_align = np.zeros((len(matches), 2), dtype=np.float32)
    for i, m in enumerate(matches):
        pts_ref[i]   = kp1[m.queryIdx].pt
        pts_align[i] = kp2[m.trainIdx].pt

    H, status = cv2.findHomography(pts_align, pts_ref, cv2.RANSAC)
    if H is None:
        return img_to_align

    height, width = img_ref.shape[:2]
    aligned = cv2.warpPerspective(img_to_align, H, (width, height))
    return aligned


def detect_changes(img1, img2, min_area=100):
    """
    Align img2 to img1, enhance diff, then detect and draw bounding boxes around changes.
    """
    # 1) Align
    img2_aligned = align_images(img1, img2)

    # 2) Compute difference and boost contrast
    diff = cv2.absdiff(img1, img2_aligned)
    diff = cv2.convertScaleAbs(diff, alpha=2.0, beta=0)

    # 3) Grayscale + blur
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4) Threshold (binary + OTSU)
    ret, _   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh   = cv2.threshold(gray, max(25, ret), 255, cv2.THRESH_BINARY)[1]

    # 5) Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  kernel, iterations=1)
    clean = cv2.dilate(clean, kernel, iterations=2)

    # 6) Find contours and draw boxes
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img2_aligned.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return output


def main():
    if len(sys.argv) != 3:
        print("Usage: compare_photos <first_image> <second_image>")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"Error: could not read images '{img1_path}' or '{img2_path}'")
        sys.exit(1)

    result = detect_changes(img1, img2)
    out_name = 'comparison_result.png'
    cv2.imwrite(out_name, result)
    print(f"Saved result with bounding boxes: {out_name}")

if __name__ == '__main__':
    main()