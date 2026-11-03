"""Check actual wire payload, retry behavior, cache identity and protocol parsing."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import requests

MODULE_PATH = Path(__file__).resolve().parents[1] / 'run_responses.py'
spec = importlib.util.spec_from_file_location('responses_runner_test', MODULE_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ResponsesTests(unittest.TestCase):
    def test_extracts_answer_not_reasoning(self):
        payload = {'output': [
            {'type': 'reasoning', 'summary': [{'text': 'private reasoning'}]},
            {'type': 'message', 'content': [{'type': 'output_text', 'text': '{"points_3d":[]}'}]},
        ]}
        self.assertEqual(runner.extract_response_text(payload), '{"points_3d":[]}')

    def test_image_payload_and_hash_change_when_input_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (128, 64), (12, 34, 56)).save(path)
            task = {'images': [str(path)], 'prompt_text': 'original', 'system_prompt': 'strict JSON'}
            config = {'model': 'gpt-6.1-sol', 'api_key': 'test-secret', 'max_image_edge': 64}
            body, audits, original_hash = runner.build_request(task, config)
            self.assertNotIn('api_key', body)
            self.assertEqual(body['input'][1]['content'][0]['type'], 'input_image')
            self.assertTrue(body['input'][1]['content'][0]['image_url'].startswith('data:image/jpeg;base64,'))
            self.assertEqual(audits[0]['sent_size'], [64, 32])
            task['prompt_text'] = 'changed'
            self.assertNotEqual(original_hash, runner.build_request(task, config)[2])

    @patch.object(runner.time, 'sleep')
    @patch.object(runner, '_http_post')
    def test_retries_429_and_preserves_usage(self, post, sleep):
        first = requests.Response()
        first.status_code = 429
        first._content = b'{"error":"rate limit"}'
        second = requests.Response()
        second.status_code = 200
        second._content = json.dumps({'status':'completed','output':[], 'usage':{'total_tokens':9}}).encode()
        post.side_effect = [first, second]
        updates = []
        payload, attempts = runner.post_response(
            {'base_url':'https://relay.invalid/v1','api_key':'test-secret','max_attempts':2},
            {'model':'gpt-6.1-sol'}, lambda x:updates.append(list(x)))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[1]['usage']['total_tokens'], 9)
        self.assertEqual(post.call_args.args[0], 'https://relay.invalid/v1/responses')
        self.assertFalse(post.call_args.kwargs['allow_redirects'])
        self.assertEqual(len(updates), 2)

    @patch.object(runner, '_http_post')
    def test_auth_errors_are_not_retried_and_key_is_redacted(self, post):
        response = requests.Response()
        response.status_code = 401
        response._content = b'test-secret denied'
        post.return_value = response
        with self.assertRaises(RuntimeError) as error:
            runner.post_response({'base_url':'https://relay.invalid/v1','api_key':'test-secret'}, {}, lambda _:None)
        self.assertNotIn('test-secret', str(error.exception))
        self.assertEqual(post.call_count, 1)

    @patch.object(runner, 'post_response')
    @patch.object(runner, 'build_request')
    def test_valid_cached_response_never_calls_api(self, build, post):
        build.return_value = ({}, [], 'same-hash')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'experiment1/raw/scene1/row_0.json'
            runner.write_json(path, {'request_sha256':'same-hash',
                                    'endpoint':'https://relay.invalid/v1/responses',
                                    'error_message':'', 'parse_ok':True})
            result = runner.run_task({'setting':'experiment1','scene':'scene1','row_index':0,'key':'scene1::0'},
                                     root, {'base_url':'https://relay.invalid/v1'}, None)
            self.assertTrue(result['parse_ok'])
            post.assert_not_called()

    @patch.object(runner, 'post_response')
    @patch.object(runner, 'build_request')
    def test_different_inputs_cannot_reuse_cache(self, build, post):
        build.return_value = ({}, [], 'new-hash')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.write_json(root / 'experiment1/raw/scene1/row_0.json',
                              {'request_sha256':'old-hash','endpoint':'https://relay.invalid/v1/responses'})
            with self.assertRaisesRegex(RuntimeError, 'Cached input/config differs'):
                runner.run_task({'setting':'experiment1','scene':'scene1','row_index':0,'key':'scene1::0'},
                                root, {'base_url':'https://relay.invalid/v1'}, None)
            post.assert_not_called()

    def test_invalid_point_dimensions_remain_failures(self):
        modules = runner.modules()
        ok, parsed, extra = runner.parse_task({'setting':'experiment3'}, '{"points_3d":[{"point":[1,2,3,4]}]}', modules)
        self.assertFalse(ok)
        self.assertEqual(parsed['points_3d'], [])
        self.assertIn('wrong_dimension', extra['invalid_reason'])


if __name__ == '__main__':
    unittest.main()
