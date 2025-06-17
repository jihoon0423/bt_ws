#!/usr/bin/env python3
# compare_node.py

import py_trees
from py_trees.common import Access, Status
import os
import cv2
import rclpy
import re
import compare_run_photos 

class CompareNode(py_trees.behaviour.Behaviour):

    def __init__(self, name="CompareNode"):
        super(CompareNode, self).__init__(name)
        self.completed = False

    def setup(self, timeout=None):
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(key="photo_paths", access=Access.READ)
        return True

    def initialise(self):
        self.completed = False

    def update(self):
        if self.completed:
            return Status.SUCCESS

        photo_paths = getattr(self.blackboard, "photo_paths", None) or []
        if not photo_paths:
            self.logger.warning("CompareNode: photo_paths가 비어 있음. 비교 대상 waypoint 이미지가 없음.")
            self.completed = True
            return Status.SUCCESS

        photo_dir = os.path.expanduser('~/waypoint_photos')
        if not os.path.isdir(photo_dir):
            self.logger.error(f"CompareNode: photo_dir 경로가 존재하지 않음: {photo_dir}")
            self.completed = True
            return Status.FAILURE

        comparison_results = []

        for path in photo_paths:
            basename = os.path.basename(path)
            m = re.match(r'waypoint_(\d+)\.png$', basename)
            if not m:
                self.logger.warning(f"CompareNode: 파일명 패턴 불일치로 건너뜀: {basename}")
                continue
            idx = m.group(1) 
            ref_filename = f"reference{idx}.png"
            ref_path = os.path.join(photo_dir, ref_filename)
            waypoint_path = path  

            self.logger.info(f"CompareNode: comparing reference {ref_filename} with waypoint {basename}")

            if not os.path.isfile(ref_path):
                self.logger.error(f"CompareNode: reference 파일이 없음: {ref_path}")
                continue
            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                self.logger.error(f"CompareNode: reference 이미지 로드 실패: {ref_path}")
                continue

            target_img = cv2.imread(waypoint_path)
            if target_img is None:
                self.logger.error(f"CompareNode: waypoint 이미지 로드 실패: {waypoint_path}")
                continue

            try:
                result = compare_run_photos.detect_changes(ref_img, target_img)
            except Exception as e:
                self.logger.error(f"CompareNode: detect_changes 예외 for idx={idx}: {e}")
                continue

            out_filename = f"comparison_result_{idx}.png"
            out_path = os.path.join(photo_dir, out_filename)
            try:
                ok = cv2.imwrite(out_path, result)
                if ok:
                    self.logger.info(f"CompareNode: 비교 결과 저장: {out_path}")
                    comparison_results.append(out_path)
                else:
                    self.logger.error(f"CompareNode: cv2.imwrite 실패: {out_path}")
            except Exception as e:
                self.logger.error(f"CompareNode: 결과 저장 예외: {out_path}, error: {e}")
                continue


        self.completed = True
        return Status.SUCCESS
