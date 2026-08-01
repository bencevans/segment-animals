import sys
import types
import unittest
from unittest.mock import patch

from segment_animals.detect import _load_detector


class LoadDetectorTests(unittest.TestCase):
    def test_retries_with_detection_model_alias(self):
        yolo = types.ModuleType("models.yolo")
        yolo.Model = type("Model", (), {})
        models = types.ModuleType("models")
        models.yolo = yolo

        missing_class = AttributeError(
            "module 'models.yolo' has no attribute 'DetectionModel'",
            name="DetectionModel",
            obj=yolo,
        )

        with (
            patch.dict(sys.modules, {"models": models, "models.yolo": yolo}),
            patch(
                "segment_animals.detect.run_detector.load_detector",
                side_effect=[missing_class, "loaded model"],
            ) as load,
        ):
            self.assertEqual(_load_detector("redwood"), "loaded model")

        self.assertIs(yolo.DetectionModel, yolo.Model)
        self.assertEqual(load.call_count, 2)

    def test_does_not_hide_unrelated_attribute_errors(self):
        error = AttributeError("unrelated", name="something_else")

        with patch(
            "segment_animals.detect.run_detector.load_detector", side_effect=error
        ):
            with self.assertRaises(AttributeError) as raised:
                _load_detector("redwood")

        self.assertIs(raised.exception, error)


if __name__ == "__main__":
    unittest.main()
