#!/usr/bin/env python3
# compare_run_photos_ssim.py

import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def align_images(img_ref, img_to_align, max_features=500, good_match_percent=0.15):
    gray_ref   = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_align = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_align, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
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


def detect_changes_ssim(img_ref, img_curr, min_area=100):
    # 정렬
    img_aligned = align_images(img_ref, img_curr)

    # 회색 변환
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)

    # SSIM 계산
    score, diff = ssim(gray_ref, gray_aligned, full=True)
    diff = (1 - diff) * 255  # SSIM은 유사도이므로, 변화 정도 = (1 - SSIM)
    diff = diff.astype(np.uint8)

    # 이진화
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 팽창으로 연결된 부분 확장
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img_aligned.copy()

    # 바운딩 박스 그리기
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

    result = detect_changes_ssim(img1, img2, min_area=100)
    out_name = 'comparison_result_ssim.png'
    cv2.imwrite(out_name, result)
    print(f"Saved SSIM-based result: {out_name}")


if __name__ == '__main__':
    main()
