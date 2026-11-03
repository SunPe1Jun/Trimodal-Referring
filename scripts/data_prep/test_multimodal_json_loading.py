"""Regression checks for released telemetry serialization, without altering raw data."""
import importlib.util
from pathlib import Path
import tempfile
import sys
import unittest

SPEC = importlib.util.spec_from_file_location('telemetry_prep', Path(__file__).with_name('build_keyframe_grounding_input.py'))
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)
ROOT = Path(__file__).resolve().parents[2]
LOADERS = {'data_prep': PREP.load_multimodal_samples}
ERRORS = [ValueError, PREP.ManifestConversionError]
for name, filename, function, error in (
    ('api', 'scripts/grounding/run_qwen3vl_plus_api_single_event_3d.py', 'load_multimodal_samples', None),
    ('local', 'scripts/grounding/run_qwen3vl_local_keyframe_grounding.py', 'load_event_json', 'GroundingRunError'),
    ('exam2', 'exam2/build_2d_eval_manifest.py', 'load_multimodal_samples', 'Exam2Error'),
    ('exam3', 'exam3_point_grounding/point_grounding_common.py', 'load_multimodal_samples', 'PointGroundingError'),
):
    spec = importlib.util.spec_from_file_location('test_telemetry_' + name, ROOT / filename)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    spec.loader.exec_module(obj)
    LOADERS[name] = getattr(obj, function)
    if error:
        ERRORS.append(getattr(obj, error))


class MultimodalJSONLoadingTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'multimodal_data.json'
            path.write_text(text, encoding='utf-8')
            results = []
            for name, loader in LOADERS.items():
                with self.subTest(loader=name):
                    results.append(loader(path))
            if results:
                self.assertTrue(all(item == results[0] for item in results))
                return results[0]

    def test_valid_array_is_preserved(self):
        self.assertEqual(self.load('[{"timestamp":"a"},{"timestamp":"b"}]'),
                         [{'timestamp':'a'}, {'timestamp':'b'}])

    def test_comma_separated_objects_retain_every_sample_in_order(self):
        self.assertEqual(self.load('{"timestamp":"a"},\n{"timestamp":"b"}'),
                         [{'timestamp':'a'}, {'timestamp':'b'}])

    def test_existing_missing_closing_bracket_repair(self):
        self.assertEqual(self.load('[{"timestamp":"a"}'), [{'timestamp':'a'}])

    def assert_rejected(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'multimodal_data.json'
            path.write_text(text, encoding='utf-8')
            for name, loader in LOADERS.items():
                with self.subTest(loader=name), self.assertRaises(tuple(ERRORS)):
                    loader(path)

    def test_missing_sample_delimiter_is_rejected(self):
        self.assert_rejected('{"timestamp":"a"}{"timestamp":"b"}')

    def test_truncated_sample_content_is_rejected(self):
        self.assert_rejected('{"timestamp":"a"}, {"timestamp":')

if __name__ == '__main__':
    unittest.main()
