import unittest

import pandas as pd

from poses.csv_to_pose_json import build_pose_json, pose_id_from_label


class PoseDataTests(unittest.TestCase):
    def test_pose_id_handles_dataset_paths_and_numbered_files(self):
        self.assertEqual(
            pose_id_from_label("TRAIN/Downward_Dog/Images/downward_dog1.jpg"),
            "downward_dog",
        )
        self.assertEqual(pose_id_from_label("chair_pose_12.jpg"), "chair_pose")

    def test_builder_groups_samples_and_clamps_sigma_tolerance(self):
        frame = pd.DataFrame(
            {
                "label": ["TRAIN/Chair/1.jpg", "TRAIN/Chair/2.jpg"],
                "left_knee": [80, 100],
                "right_knee": [82, 98],
            }
        )
        result = build_pose_json(
            frame,
            label_col="label",
            hold_ms=900,
            points=10,
            tol_mode="sigma",
            tol_default=20,
            tol_sigma_mult=2,
            tol_min=8,
            tol_max=25,
            decimals=0,
            include_template_landmarks=False,
        )
        pose = result["poses"][0]
        self.assertEqual(pose["pose_id"], "chair")
        self.assertEqual(pose["target_angles"], {"left_knee": 90, "right_knee": 90})
        self.assertTrue(all(8 <= value <= 25 for value in pose["tolerances"].values()))

    def test_builder_rejects_invalid_schema(self):
        with self.assertRaisesRegex(ValueError, "Label column"):
            build_pose_json(
                pd.DataFrame({"angle": [90]}),
                label_col="label",
                hold_ms=900,
                points=10,
                tol_mode="default",
                tol_default=20,
                tol_sigma_mult=2,
                tol_min=8,
                tol_max=35,
                decimals=0,
                include_template_landmarks=False,
            )


if __name__ == "__main__":
    unittest.main()
