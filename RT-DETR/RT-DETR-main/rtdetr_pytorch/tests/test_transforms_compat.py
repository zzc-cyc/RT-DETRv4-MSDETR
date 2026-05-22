import unittest

import torch

from src.core import GLOBAL_CONFIG
from src.data.datapoints_compat import datapoints
from src.data import transforms


class TransformCompatTests(unittest.TestCase):
    def test_official_sanitize_bounding_box_name_is_registered(self):
        self.assertIn("SanitizeBoundingBox", GLOBAL_CONFIG)

        transform = transforms.Compose(
            [
                {"type": "SanitizeBoundingBox", "min_size": 1},
            ]
        )

        self.assertEqual(len(transform.transforms), 1)
        self.assertEqual(type(transform.transforms[0]).__name__, "SanitizeBoundingBox")

    def test_bounding_box_compat_is_a_type_with_spatial_size(self):
        boxes = datapoints.BoundingBox(
            torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=(20, 30),
        )

        self.assertIsInstance(boxes, datapoints.BoundingBox)
        self.assertEqual(boxes.spatial_size, (20, 30))

    def test_convert_box_handles_torchvision_bounding_boxes(self):
        boxes = datapoints.BoundingBox(
            torch.tensor([[10.0, 20.0, 110.0, 220.0]]),
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=(400, 500),
        )
        resized = transforms.Resize(size=[800, 1000])(boxes)

        converted = transforms.ConvertBox(out_fmt="cxcywh", normalize=True)(resized)

        self.assertLessEqual(float(converted.max()), 1.0)
        self.assertTrue(torch.allclose(converted, torch.tensor([[0.12, 0.30, 0.20, 0.50]]), atol=1e-6))

    def test_pad_to_size_handles_torchvision_bounding_boxes(self):
        boxes = datapoints.BoundingBox(
            torch.tensor([[10.0, 20.0, 110.0, 220.0]]),
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=(400, 500),
        )
        resized = transforms.Resize(size=[800, 1000])(boxes)

        padded = transforms.PadToSize(spatial_size=[900, 1100])(resized)

        self.assertEqual(padded.spatial_size, (900, 1100))


if __name__ == "__main__":
    unittest.main()
