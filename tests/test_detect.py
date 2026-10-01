import sys
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from segment_animals.detect import DetectionModel, _load_detector


class DetectionCategoryTests(unittest.TestCase):
    def test_humans_are_opt_in_and_threshold_still_applies(self):
        detector = Mock()
        detector.generate_detections_one_image.return_value = {
            "detections": [
                {"category": category, "conf": confidence, "bbox": [0.1, 0.2, 0.3, 0.4]}
                for category, confidence in [("1", 0.9), ("2", 0.8), ("3", 0.7), ("2", 0.1)]
            ]
        }
        image = Image.new("RGB", (100, 200))
        with patch("segment_animals.detect._load_detector", return_value=detector):
            animals = DetectionModel().detect(image)
            both = DetectionModel(include_humans=True).detect(image)
        self.assertEqual([d.confidence for d in animals], [0.9])
        self.assertEqual([d.confidence for d in both], [0.9, 0.8])
        self.assertEqual(both[1].bbox, (10, 40, 30, 80))
        self.assertEqual([d.category for d in animals], ["animal"])
        self.assertEqual([d.category for d in both], ["animal", "human"])

    def test_pipeline_forwards_detection_classes(self):
        from segment_animals import AutoSegmenter

        with (
            patch("segment_animals.DetectionModel") as detector,
            patch("segment_animals.SegmentationModel"),
        ):
            AutoSegmenter(detection_classes=("human",))
        self.assertEqual(detector.call_args.kwargs["detection_classes"], ("human",))

    def test_compatibility_wrapper_selects_animals_and_optional_humans(self):
        from segment_animals import AutoAnimalSegmenter, AutoSegmenter

        for include_humans, expected in [(False, ("animal",)), (True, ("animal", "human"))]:
            with (
                self.subTest(include_humans=include_humans),
                patch("segment_animals.DetectionModel") as detector,
                patch("segment_animals.SegmentationModel"),
                self.assertWarnsRegex(DeprecationWarning, "AutoAnimalSegmenter is deprecated") as warning,
            ):
                model = AutoAnimalSegmenter(include_humans=include_humans)
                self.assertIsInstance(model, AutoSegmenter)
                self.assertEqual(detector.call_args.kwargs["detection_classes"], expected)
            self.assertEqual(warning.filename, __file__)

    def test_default_pipeline_selects_only_animals_without_deprecation(self):
        import warnings
        from segment_animals import AutoSegmenter

        with (
            patch("segment_animals.DetectionModel") as detector,
            patch("segment_animals.SegmentationModel"),
            warnings.catch_warnings(record=True) as emitted,
        ):
            warnings.simplefilter("always")
            AutoSegmenter()
        self.assertEqual(detector.call_args.kwargs["detection_classes"], ("animal",))
        self.assertEqual(emitted, [])

    def test_selected_classes_filter_before_segmentation(self):
        from segment_animals import AutoSegmenter
        from segment_animals.models import AnimalDetection, Detection

        self.assertIs(AnimalDetection, Detection)
        detector = Mock()
        detector.generate_detections_one_image.return_value = {
            "detections": [
                {"category": c, "conf": 0.9, "bbox": [0.1, 0.2, 0.3, 0.4]}
                for c in ("1", "2", "3")
            ]
        }
        image = Image.new("RGB", (100, 200))
        for selected in [("animal",), ("human",), ("vehicle",), ("animal", "human")]:
            with (
                self.subTest(selected=selected),
                patch("segment_animals.detect._load_detector", return_value=detector),
                patch("segment_animals.SegmentationModel") as segmentor,
            ):
                pipeline = AutoSegmenter(detection_classes=selected)
                detections, masks = pipeline.process_image(image)
                self.assertEqual([d.category for d in detections], list(selected))
                segmentor.return_value.segment.assert_called_once_with(image, detections)
                self.assertIs(masks, segmentor.return_value.segment.return_value)

    def test_invalid_classes_fail_before_loading_models(self):
        from segment_animals import AutoSegmenter

        for selected in [(), ("unknown",), "human"]:
            with (
                self.subTest(selected=selected),
                patch("segment_animals.detect._load_detector") as load,
                patch("segment_animals.SegmentationModel") as segmentor,
            ):
                with self.assertRaises(ValueError):
                    AutoSegmenter(detection_classes=selected)
                load.assert_not_called()
                segmentor.assert_not_called()


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
