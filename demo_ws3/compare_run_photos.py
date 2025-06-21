#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import argparse

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

def detect_changes(img1, img2, min_area=500, thresh_val=None, alpha=1.0, blur_ksize=(5,5),
                   morph_kernel_size=(5,5), morph_close_iters=2, morph_open_iters=1, dilate_iters=2):
    """
    Align img2 to img1, compute diff, 이진화, 모폴로지로 정리한 뒤
    면적이 min_area 이상인 영역에만 bounding box를 그림.
    - thresh_val: None이면 Otsu로 구한 값과 기본값(max(25, otsu_val)) 중 큰 값을 쓰되,
                  외부 지정 시 고정 임계값(thresh_val) 사용.
    - alpha: diff 대비 대비 증폭 정도 (너무 크면 작은 변화도 커보이므로 1.0~1.5 정도 권장)
    """
    # 1) Align
    img2_aligned = align_images(img1, img2)

    # 2) Compute difference 및 대비 증폭
    diff = cv2.absdiff(img1, img2_aligned)
    # alpha가 크면 미세 변화도 강조되어 잡히므로, 너무 크지 않게 설정
    diff = cv2.convertScaleAbs(diff, alpha=alpha, beta=0)

    # 3) Grayscale + Blur
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # 노이즈 억제를 위해 가우시안 블러 또는 미디안 블러(선택) 강화
    gray = cv2.GaussianBlur(gray, blur_ksize, 0)
    # 예: 작은 노이즈가 많으면 cv2.medianBlur(gray, 5) 등도 시도 가능

    # 4) Threshold (binary)
    if thresh_val is None:
        # Otsu 기준 구하기
        ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 작동 실험: ret이 너무 낮으면 작게 잡히므로 최소값 지정
        thresh_level = max(30, ret)  # 기본 30 이상. 필요시 조절.
    else:
        thresh_level = thresh_val
    _, thresh = cv2.threshold(gray, thresh_level, 255, cv2.THRESH_BINARY)

    # 5) Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iters)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  kernel, iterations=morph_open_iters)
    clean = cv2.dilate(clean, kernel, iterations=dilate_iters)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Compare two photos and draw bounding boxes around significant changes.")
    parser.add_argument("first_image", help="Path to the first (reference) image")
    parser.add_argument("second_image", help="Path to the second image to compare")
    parser.add_argument("--min-area", type=int, default=500,
                        help="변화 영역의 최소 픽셀 면적(min_area). 기본 500. 실험하며 적절히 조절")
    parser.add_argument("--thresh", type=int, default=None,
                        help="Threshold 값. 지정 없으면 Otsu 기반(max(30, otsu_val)). 너무 민감하면 값을 높여보세요.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="diff 대비 증폭 계수(alpha). 기본 1.0. 작게 설정하면 미세 변화 덜 강조됨.")
    parser.add_argument("--blur-kernel", type=int, nargs=2, metavar=('W','H'), default=(5,5),
                        help="가우시안 블러 커널 크기. 기본 5 5. 노이즈 많으면 키워보기.")
    parser.add_argument("--morph-kernel", type=int, nargs=2, metavar=('W','H'), default=(5,5),
                        help="모폴로지 연산 커널 크기. 기본 5 5.")
    parser.add_argument("--close-iters", type=int, default=2, help="모폴로지 close 반복 횟수")
    parser.add_argument("--open-iters", type=int, default=1, help="모폴로지 open 반복 횟수")
    parser.add_argument("--dilate-iters", type=int, default=2, help="dilate 반복 횟수")
    parser.add_argument("--output", "-o", default="comparison_result.png", help="결과 이미지 파일명")
    return parser.parse_args()

def main():
    args = parse_args()
    img1 = cv2.imread(args.first_image)
    img2 = cv2.imread(args.second_image)
    if img1 is None or img2 is None:
        print(f"Error: could not read images '{args.first_image}' or '{args.second_image}'")
        sys.exit(1)

    result = detect_changes(
        img1, img2,
        min_area=args.min_area,
        thresh_val=args.thresh,
        alpha=args.alpha,
        blur_ksize=tuple(args.blur_kernel),
        morph_kernel_size=tuple(args.morph_kernel),
        morph_close_iters=args.close_iters,
        morph_open_iters=args.open_iters,
        dilate_iters=args.dilate_iters
    )
    cv2.imwrite(args.output, result)
    print(f"Saved result with bounding boxes: {args.output}")

if __name__ == '__main__':
    main()
