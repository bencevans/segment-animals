import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import torch
from transformers import BatchFeature

from segment_animals import AutoSegmenter
from segment_animals.models import AnimalDetection
from segment_animals.segment import MODEL_IDS, SegmentationModel
from segment_animals.viz import extract_masks


class SegmentationTests(unittest.TestCase):
    def test_all_models_load_matching_processor_and_device(self):
        for name, model_id in MODEL_IDS.items():
            with (
                self.subTest(name=name),
                patch('segment_animals.segment.Sam2Model.from_pretrained') as load,
                patch('segment_animals.segment.Sam2Processor.from_pretrained') as processor,
            ):
                model = SegmentationModel(name, device='cpu')
                load.assert_called_once_with(model_id)
                load.return_value.to.assert_called_once_with('cpu')
                load.return_value.to.return_value.eval.assert_called_once_with()
                processor.assert_called_once_with(model_id)
                self.assertIs(model.processor, processor.return_value)

    def test_default_pipeline_uses_sam2(self):
        with (
            patch('segment_animals.DetectionModel'),
            patch('segment_animals.SegmentationModel') as segmentor,
        ):
            AutoSegmenter(segmentation_device='cpu')
            segmentor.assert_called_once_with(model_name='sam2.1_hiera_large', device='cpu')

    def test_unknown_model_fails_before_download(self):
        with patch('segment_animals.segment.Sam2Model.from_pretrained') as download:
            with self.assertRaisesRegex(ValueError, 'Unknown model'):
                SegmentationModel('vit_h')
            download.assert_not_called()

    def test_masks_and_pixel_boxes_for_one_and_multiple_animals(self):
        for count in (1, 2):
            with self.subTest(count=count):
                model = SegmentationModel.__new__(SegmentationModel)
                model.sam = MagicMock(device=torch.device('cpu'))
                model.processor = MagicMock()
                inputs = BatchFeature({'pixel_values': torch.zeros(1, 3, 8, 12),
                                       'original_sizes': torch.tensor([[8, 12]]),
                                       'input_boxes': torch.tensor([[[1, 2, 5, 6]] * count])})
                model.processor.return_value = inputs
                raw = torch.zeros((1, count, 1, 2, 3))
                def predict(**kwargs):
                    self.assertTrue(torch.is_inference_mode_enabled())
                    self.assertFalse(kwargs['multimask_output'])
                    self.assertIs(kwargs['pixel_values'], inputs['pixel_values'])
                    return MagicMock(pred_masks=raw)
                model.sam.side_effect = predict
                restored = torch.zeros((count, 1, 8, 12), dtype=torch.bool)
                restored[:, :, 2:6, 1:5] = True
                model.processor.post_process_masks.return_value = [restored]
                image = Image.new('RGBA', (12, 8))
                detections = [AnimalDetection(bbox=(1, 2, 4, 4), confidence=.9)] * count
                masks = model.segment(image, detections)
                self.assertEqual(model.processor.call_args.kwargs['input_boxes'],
                                 [[[1, 2, 5, 6]] * count])
                self.assertEqual(model.processor.call_args.kwargs['images'].mode, 'RGB')
                post_args = model.processor.post_process_masks.call_args
                self.assertTrue(torch.equal(post_args.args[0], raw))
                self.assertTrue(torch.equal(post_args.args[1], torch.tensor([[8, 12]])))
                self.assertTrue(post_args.kwargs['binarize'])
                self.assertEqual(tuple(masks.shape), (count, 1, 8, 12))
                self.assertEqual(masks.dtype, torch.bool)
                self.assertEqual(masks.device.type, 'cpu')
                self.assertEqual([im.size for im in extract_masks(image, masks)], [(4, 4)] * count)

    def test_no_detections_skips_encoder(self):
        model = SegmentationModel.__new__(SegmentationModel)
        model.processor = MagicMock()
        model.sam = MagicMock()
        masks = model.segment(Image.new('RGB', (12, 8)), [])
        self.assertEqual(tuple(masks.shape), (0, 8, 12))
        self.assertEqual(masks.dtype, torch.bool)
        model.processor.assert_not_called()
        model.sam.assert_not_called()
