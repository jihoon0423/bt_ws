#!/usr/bin/env python3
import os
import glob
import re
import cv2
import compare_run_photos  
import pandas as pd

def test_comparisons(photo_dir):

    photo_dir = os.path.expanduser(photo_dir)
    if not os.path.isdir(photo_dir):
        print(f"Directory does not exist: {photo_dir}")
        return pd.DataFrame()

    results = []
    waypoint_paths = sorted(glob.glob(os.path.join(photo_dir, "waypoint_*.png")))
    if not waypoint_paths:
        print(f"No waypoint_*.png files found in {photo_dir}")
    for path in waypoint_paths:
        basename = os.path.basename(path)
        m = re.match(r"waypoint_(\d+)\.png$", basename)
        if not m:
            print(f"Skipping non-matching file: {basename}")
            continue
        idx = m.group(1)
        ref_filename = f"reference{idx}.png"
        ref_path = os.path.join(photo_dir, ref_filename)
        if not os.path.isfile(ref_path):
            print(f"Reference file not found for idx {idx}: {ref_path}")
            continue
        ref_img = cv2.imread(ref_path)
        target_img = cv2.imread(path)
        if ref_img is None:
            print(f"Failed to load reference image: {ref_path}")
            continue
        if target_img is None:
            print(f"Failed to load waypoint image: {path}")
            continue

        try:
            result_img = compare_run_photos.detect_changes(ref_img, target_img)
        except Exception as e:
            print(f"detect_changes error for idx={idx}: {e}")
            continue


        out_filename = f"comparison_result_{idx}.png"
        out_path = os.path.join(photo_dir, out_filename)
        try:
            ok = cv2.imwrite(out_path, result_img)
            if not ok:
                print(f"Failed to write comparison image: {out_path}")
                saved = False
            else:
                print(f"Saved comparison image: {out_path}")
                saved = True
        except Exception as e:
            print(f"Error saving comparison image for idx={idx}: {e}")
            saved = False

        results.append({
            "idx": int(idx),
            "reference": ref_path,
            "waypoint": path,
            "comparison_result": out_path if saved else None
        })

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    photo_dir = "~/waypoint_photos"
    df_results = test_comparisons(photo_dir)
    if not df_results.empty:
        print("\n===== Comparison Summary =====")
        print(df_results.to_string(index=False))
    else:
        print("No comparison results.")
