#!/usr/bin/env python3
"""Run the three frozen VR-TriRef protocols through a Responses-compatible API."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import threading
from types import SimpleNamespace
from typing import Any

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'exam3_point_grounding'))
DEFAULT_SELECTION = {
    'experiment1': ['scene1::0', 'scene2::400', 'scene3::500', 'scene5::400'],
    'experiment2': ['scene1::1', 'scene4_room2::100', 'scene5::600'],
    'experiment3': ['scene2::0', 'scene3::400', 'scene4_room4::100'],
}


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract_response_text(payload):
    """Read Responses message content, never reasoning summaries or tool calls."""
    texts = []
    for item in payload.get('output', []):
        if item.get('type') == 'message':
            for content in item.get('content', []):
                if content.get('type') == 'output_text':
                    texts.append(content.get('text', ''))
    return '\n'.join(texts) or str(payload.get('output_text') or '')


def encode_image(path, max_edge):
    path = Path(path)
    with Image.open(path) as image:
        original_size = list(image.size)
        if max(image.size) > max_edge:
            image = image.convert('RGB')
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            data = buffer.getvalue()
            sent_size = list(image.size)
            mime = 'image/jpeg'
        else:
            data = path.read_bytes()
            sent_size = original_size
            mime = Image.MIME.get(image.format, 'image/jpeg')
    return 'data:' + mime + ';base64,' + base64.b64encode(data).decode(), {
        'path': str(path), 'source_sha256': sha(path.read_bytes()),
        'sent_sha256': sha(data), 'original_size': original_size,
        'sent_size': sent_size, 'bytes': len(data),
    }


def build_request(task, config):
    content, image_audit = [], []
    for path in task['images']:
        data_url, audit = encode_image(path, int(config.get('max_image_edge', 1920)))
        content.append({'type': 'input_image', 'image_url': data_url,
                        'detail': config.get('image_detail', 'high')})
        image_audit.append(audit)
    content.append({'type': 'input_text', 'text': task['prompt_text']})
    body = {
        'model': config['model'], 'store': False,
        'max_output_tokens': int(config.get('max_output_tokens', 4096)),
        'input': [
            {'role': 'system', 'content': [{'type': 'input_text', 'text': task['system_prompt']}]},
            {'role': 'user', 'content': content},
        ],
    }
    if config.get('reasoning_effort'):
        body['reasoning'] = {'effort': config['reasoning_effort']}
    body_hash = sha(json.dumps(body, sort_keys=True, separators=(',', ':')).encode())
    return body, image_audit, body_hash


_http_sessions = threading.local()


def _http_post(*args, **kwargs):
    # Reuse a connection within each worker to reduce DNS lookups and TLS setup.
    if not hasattr(_http_sessions,'session'):
        _http_sessions.session = requests.Session()
    return _http_sessions.session.post(*args, **kwargs)


def post_response(config, body, on_attempt):
    endpoint = config['base_url'].rstrip('/') + '/responses'
    key = config.get('api_key') or os.environ.get(config.get('api_key_env', 'TRIMODAL_OPENAI_API_KEY'))
    if not key:
        raise ValueError('Missing API key in private config or configured environment variable')
    attempts = []
    for number in range(1, int(config.get('max_attempts', 2)) + 1):
        started = time.monotonic()
        record = {'attempt': number}
        retry = False
        try:
            response = _http_post(endpoint, headers={'Authorization': 'Bearer ' + key},
                                     json=body, timeout=float(config.get('timeout_seconds', 180)),
                                     allow_redirects=False)
            record.update(http_status=response.status_code, elapsed_seconds=time.monotonic() - started,
                          request_id=response.headers.get('x-request-id'))
            if response.status_code == 200:
                payload = response.json()
                record['usage'] = payload.get('usage')
                record['response_status'] = payload.get('status')
                attempts.append(record)
                on_attempt(attempts)
                return payload, attempts
            record['error'] = response.text[:1500].replace(key, '[REDACTED]')
            retry = response.status_code in (408, 429, 500, 502, 503, 504)
        except requests.RequestException as exc:
            record.update(elapsed_seconds=time.monotonic() - started,
                          error=str(exc).replace(key, '[REDACTED]'))
            retry = True
        attempts.append(record)
        on_attempt(attempts)
        if not retry or number == int(config.get('max_attempts', 2)):
            break
        time.sleep(min(2 ** number, 8))
    raise RuntimeError('Responses request failed: ' + json.dumps(attempts, ensure_ascii=False))


def modules():
    return SimpleNamespace(
        anchor=load_module('closed_anchor', 'scripts/grounding/run_qwen3vl_plus_api_single_event_3d.py'),
        local=load_module('closed_local', 'scripts/grounding/run_qwen3vl_local_single_event_3d.py'),
        frames=load_module('closed_frames', 'scripts/grounding/run_qwen3vl_local_keyframe_grounding.py'),
        builder2=load_module('closed_builder2', 'exam2/build_2d_eval_manifest.py'),
        runner2=load_module('closed_runner2', 'exam2/run_qwen3vl_2d_point_grounding.py'),
        builder3=load_module('closed_builder3', 'exam3_point_grounding/build_point_grounding_manifest.py'),
        parser3=load_module('closed_parser3', 'exam3_point_grounding/point_parser.py'),
        evaluator1=load_module('closed_eval1', 'scripts/eval/evaluate_local_3d_object_match.py'),
    )


def arguments_for_parser(module, argv):
    old = sys.argv
    try:
        sys.argv = [module.__file__, *argv]
        return module.parse_args()
    finally:
        sys.argv = old


def restore_frozen_gt(output):
    """Restore builder input only from archived annotations, never predictions."""
    records = read_csv(ROOT / 'paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv')
    grouped = {}
    for row in records:
        grouped.setdefault(row['scene'], []).append({
            'row_index': row['row_index'], 'event_id': row['event_id'],
            'gt_referents_mapped': ','.join(json.loads(row['gt_anchor_ids'])),
        })
    directory = output / 'frozen_gt'
    for scene, rows in grouped.items():
        write_csv(directory / (scene + '_match_eval.csv'), rows)
    return directory, {r['unified_id']: r for r in records}


def task_base(setting, key, row, images, prompt, system):
    scene, index = key.split('::')
    if not images or not all(Path(p).is_file() for p in images):
        raise RuntimeError('Missing actual model input images for ' + setting + ' ' + key)
    return {'setting': setting, 'key': key, 'scene': scene, 'row_index': int(index),
            'event_id': row['event_id'], 'instruction': row.get('instruction_text', row.get('instruction', '')),
            'images': [str(p) for p in images], 'prompt_text': prompt, 'system_prompt': system}


def prepare(selection, output, m, exp3_evidence_mode="legacy", evidence_frames=8, clock_contract_root=None, prepare_workers=4):
    eval_dir, gt = restore_frozen_gt(output)
    api_rows = {}
    def row_for(key):
        scene, index = key.split('::')
        if scene not in api_rows:
            api_rows[scene] = read_csv(ROOT / 'data' / (scene + '_api_input.csv'))
        return api_rows[scene][int(index)]
    tasks = []
    for key in selection.get('experiment1', []):
        scene, index = key.split('::')
        row = row_for(key)
        frames_dir = output / 'experiment1/frames' / scene / ('row_' + index)
        frames_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(frames_dir.glob('frame_*.jpg'))
        if not paths:
            images = m.frames.load_video_frames(
                Path(row['video_path']), SimpleNamespace(image_module=Image), None, None, 16,
                float(row['t_start']), float(row['t_end']))
            for i, image in enumerate(images, 1):
                path = frames_dir / f'frame_{i:03d}.jpg'
                image.save(path, quality=95)
                paths.append(path)
        anchors = m.anchor.load_scene_anchor_table(ROOT / 'data' / (scene + '_anchor_table.tsv'))
        prompt = m.anchor.build_3d_object_prompt(row, anchors, 'full', prompt_strategy='mention_first', max_evidence_segments=0)
        task = task_base('experiment1', key, row, paths, prompt, m.local.build_system_prompt())
        task['source_time_window'] = [row['t_start'], row['t_end']]
        tasks.append(task)
        print('[prepared] experiment1', key, 'images=', len(paths), flush=True)

    manifest2 = []
    grouped2 = {}
    for key in selection.get('experiment2', []):
        grouped2.setdefault(key.split('::')[0], []).append(int(key.split('::')[1]))
    for scene, indexes in grouped2.items():
        # One invocation for contiguous ranges; sparse smoke keys avoid processing other samples.
        ranges = [(min(indexes), len(indexes))] if sorted(indexes) == list(range(min(indexes), max(indexes)+1)) else [(i, 1) for i in indexes]
        for start, count in ranges:
            args = arguments_for_parser(m.builder2, [
                '--repo_root', str(ROOT), '--eval_dir', str(eval_dir),
                '--output_dir', str(output / 'experiment2/manifest'), '--scenes', scene,
                '--start_index', str(start), '--limit', str(count), '--panels', '3',
                '--auto_video_time_offset', '--auto_video_time_offset_source', 'hybrid',
                '--hybrid_offset_threshold_seconds', '1.0', '--hybrid_video_time_bias_seconds', '0.5',
                '--acceptable_top_k', '2', '--sample_inner_margin_ratio', '0.15',
                '--panel_selection_strategy', 'evidence', '--candidate_step_seconds', '0.5',
                '--evidence_segment_duration_seconds', '0.5', '--gaze_stability_window_seconds', '0.3'])
            rows, _ = m.builder2.process_scene(m.builder2.scene_paths(ROOT, eval_dir, scene), output / 'experiment2/manifest', args, [])
            manifest2.extend(rows)
    args2 = SimpleNamespace(panel_context_mode='paired_crop', gaze_crop_ratio=0.35,
                            crop_output_size=768, overwrite=False)
    grouped_rows = m.runner2.group_manifest(manifest2)
    for key in selection.get('experiment2', []):
        scene, index = key.split('::')
        rows = grouped_rows[(scene, int(index))]
        images, metadata = m.runner2.collect_model_panels((scene, int(index)), rows)
        images, metadata = m.runner2.expand_panel_context_images((scene, int(index)), images, metadata, output / 'experiment2/model_inputs', args2)
        prompt = m.runner2.build_prompt(rows, metadata, 'expected_count')
        task = task_base('experiment2', key, row_for(key), images, prompt, m.runner2.SYSTEM_PROMPT)
        task.update(panel_meta=metadata, coordinate_policy='paired_canvas_map')
        tasks.append(task)
        print('[prepared] experiment2', key, 'images=', len(images), flush=True)
    if manifest2:
        m.builder2.write_manifest(output / 'experiment2/manifest/manifest_all.csv', manifest2)

    manifest3, gt3 = [], []
    template = (ROOT / 'paper_experiment_evidence/prompts_and_configs/exp3_qwen3vl_point_grounding.md').read_text()
    frozen3 = {(r['scene'], r['row_index']): r for r in read_csv(ROOT / 'paper_experiment_evidence/manifests/experiment3_evaluable_manifest.csv')}
    def prepare_one_exp3(key):
        scene, index = key.split('::')
        row = row_for(key)
        anchors = m.builder3.load_anchor_table(ROOT, scene)
        bounds = m.builder3.robust_bounds(anchors)
        if exp3_evidence_mode == 'coverage_pts':
            from temporal_evidence import select_coverage_frames
            frames, stats = select_coverage_frames(row, output / 'experiment3/evidence_frames' / scene / ('row_' + index),
                evidence_frames, clock_contract_root, scene, int(index))
        else:
            frames, stats = m.builder3.select_evidence_frames(row, output / 'experiment3/evidence_frames' / scene / ('row_' + index),
                max_frames=3, candidate_step_seconds=0.5, stability_window_seconds=0.3,
                min_sep_seconds=0.85, ffmpeg_path='ffmpeg', no_extract_frames=False, overwrite_frames=False)
        evidence = m.builder3.build_evidence_payload(frames)
        if exp3_evidence_mode == 'coverage_pts':
            for item, alignment in zip(evidence,stats['frames']):
                item['alignment'] = alignment
            write_json(output / 'experiment3/evidence_frames' / scene / ('row_' + index) / 'alignment.json', stats)
        prompt = m.builder3.build_prompt(template, row, bounds, evidence)
        images = [f.frame_path for f in frames]
        task = task_base('experiment3', key, row, images, prompt,
                         'Return strict JSON only for candidate-free point-supervised 3D referent grounding.')
        task['evidence_json'] = evidence
        task['evidence_mode'] = exp3_evidence_mode
        print('[prepared] experiment3', key, 'images=', len(images), flush=True)
        return task, {k:v for k,v in task.items() if k != 'evidence_json'} | {'evidence_json': json.dumps(evidence)}, frozen3[(scene,index)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=prepare_workers) as prepare_pool:
        for task, manifest_row, gt_row in prepare_pool.map(prepare_one_exp3,selection.get('experiment3', [])):
            tasks.append(task)
            manifest3.append(manifest_row)
            gt3.append(gt_row)
    if gt3:
        write_csv(output / 'experiment3/gt_manifest_eval.csv', gt3)
        write_json(output / 'experiment3/model_manifest.json', manifest3)
    write_json(output / 'prepared_tasks.json', tasks)
    write_json(output / 'selection.json', selection)
    return tasks, gt


def validate_projection_archive(output):
    generated = read_csv(output / 'experiment2/manifest/manifest_all.csv')
    archived = {(r['scene'],r['row_index'],r['referent_name'],r['panel_id']):r
                for r in read_csv(ROOT / 'paper_experiment_evidence/manifests/experiment2_projected_gt_manifest.csv')}
    differences = []
    for row in generated:
        key = (row['scene'],row['row_index'],row['referent_name'],row['panel_id'])
        old = archived.get(key)
        if old is None or row['projection_valid'] != old['projection_valid']:
            differences.append({'key':key,'reason':'missing_key_or_projection_validity_changed'})
        elif row['projection_valid'] == 'True':
            delta = max(abs(float(row['gt_x'])-float(old['gt_x'])),abs(float(row['gt_y'])-float(old['gt_y'])))
            if delta > .02:
                differences.append({'key':key,'reason':'pixel_projection_changed','max_xy_delta':delta})
    result = {'pass':not differences,'checked_rows':len(generated),'differences':differences,
              'temporal_label_equivalence':'unverified; archived evidence_acceptable unavailable'}
    write_json(output / 'experiment2/projection_archive_check.json',result)
    if differences:
        raise RuntimeError('Frozen 2D projection differs; stopped before model requests. See projection_archive_check.json')


def parse_task(task, raw, m):
    if task['setting'] == 'experiment1':
        parsed, warnings = m.anchor.parse_3d_response_text(raw)
        ok = isinstance(parsed, dict) and isinstance(parsed.get('selected_object_names'), list)
        return ok, parsed if ok else {}, {'validation_warnings': warnings}
    if task['setting'] == 'experiment2':
        parsed = m.runner2.extract_json(raw)
        ok = isinstance(parsed, dict) and isinstance(parsed.get('referents'), list)
        if not ok:
            return False, {}, {'raw_parsed_json': parsed}
        normalized = m.runner2.normalize_prediction_payload(parsed)
        mapped = m.runner2.map_prediction_coordinates(normalized, task['panel_meta'], task['coordinate_policy'])
        return True, mapped, {'raw_parsed_json': parsed, 'paired_crop_coordinate_policy': task['coordinate_policy']}
    ok, parsed, reason = m.parser3.parse_points_3d_output(raw)
    return ok, parsed, {'invalid_reason': reason}


def run_task(task, output, config, m):
    directory = output / task['setting'] / 'raw' / task['scene']
    path = directory / f"row_{task['row_index']}.json"
    body, image_audit, body_hash = build_request(task, config)
    endpoint = config['base_url'].rstrip('/') + '/responses'
    history = []
    if path.exists():
        existing = json.loads(path.read_text())
        history = existing.get('request_history', [])
        if config.get('retry_errors') and existing.get('error_message'):
            previous_attempt_path = path.with_suffix('.attempts.json')
            previous_attempts = json.loads(previous_attempt_path.read_text()) if previous_attempt_path.exists() else []
            history = history + [{'error_message':existing['error_message'], 'attempts':previous_attempts}]
        if existing.get('request_sha256') != body_hash or existing.get('endpoint') != endpoint:
            raise RuntimeError('Cached input/config differs; choose a new output directory: ' + str(path))
        # Final invalid model outputs are preserved; transport failures need explicit --retry-errors.
        if not existing.get('error_message') or not config.get('retry_errors'):
            print('[cached]', task['setting'], task['key'], flush=True)
            return existing
    audit = {k:v for k,v in task.items() if k not in ('evidence_json',)}
    audit.update(model=config['model'], endpoint=endpoint, request_sha256=body_hash,
                 prompt_sha256=sha(task['prompt_text'].encode()), image_audit=image_audit,
                 request_options={k:v for k,v in body.items() if k != 'input'}, parse_ok=False,
                 parsed_json={}, model_raw_output='', error_message='', request_history=history)
    try:
        payload, attempts = post_response(config, body, lambda attempts: write_json(path.with_suffix('.attempts.json'), attempts))
        raw = extract_response_text(payload)
        # Persist a successful paid response before any downstream parsing/evaluation.
        audit.update(response=payload, attempts=attempts, model_raw_output=raw,
                     served_model=payload.get('model'), usage=payload.get('usage'), response_status='unparsed')
        write_json(path, audit)
        ok, parsed, extra = parse_task(task, raw, m)
        completed = payload.get('status') == 'completed'
        if not completed:
            ok, parsed = False, {}
            extra['invalid_reason'] = 'response_' + str(payload.get('status', 'unknown'))
        audit.update(response=payload, attempts=attempts, model_raw_output=raw,
                     parsed_json=parsed, parse_ok=ok, response_status='ok' if ok else 'invalid',
                     served_model=payload.get('model'), usage=payload.get('usage'), **extra)
        if task['setting'] == 'experiment1':
            audit['parsed_response'] = parsed
        print('[done]', task['setting'], task['key'], 'parse_ok=', ok,
              'tokens=', (payload.get('usage') or {}).get('total_tokens'), flush=True)
    except Exception as exc:
        key = config.get('api_key', '')
        message = str(exc).replace(key, '[REDACTED]') if key else str(exc)
        audit.update(error_message=message, response_status='error')
        print('[error]', task['setting'], task['key'], message[:400], flush=True)
    write_json(path, audit)
    return audit


def evaluate(results, gt, output, m):
    summaries = {}
    for setting in ['experiment1', 'experiment2', 'experiment3']:
        records = [r for r in results if r['setting'] == setting]
        if not records:
            continue
        if setting == 'experiment1':
            details, selected_gt = [], []
            for scene in sorted({r['scene'] for r in records}):
                aliases, _ = m.evaluator1.load_anchor_alias_map(ROOT / 'data' / (scene + '_anchor_table.tsv'))
                all_gt = []
                scene_gt = sorted([g for g in gt.values() if g['scene'] == scene], key=lambda x:int(x['row_index']))
                for g in scene_gt:
                    all_gt.append({'number':g['row_index'], 'excel_row_index':int(g['row_index'])+2,
                                   'instruction':'', 'referents_raw':','.join(json.loads(g['gt_anchor_ids'])),
                                   'mapped_referents':json.loads(g['gt_anchor_ids']), 'unmapped_referents':[]})
                chosen = [r for r in records if r['scene'] == scene]
                paths = [output / setting / 'raw' / scene / f"row_{r['row_index']}.json" for r in chosen]
                details.extend(m.evaluator1.build_eval_rows(paths, all_gt, aliases))
                selected_gt.extend(all_gt[r['row_index']] for r in chosen)
            summaries[setting] = m.evaluator1.summarize(details, selected_gt)
            m.evaluator1.write_eval_csv(output / setting / 'eval/detail.csv', details)
            write_json(output / setting / 'eval/summary.json', summaries[setting])
        else:
            rows = []
            for r in records:
                parsed = r['parsed_json']
                if setting == 'experiment3' and not r['parse_ok']:
                    parsed = {'points_3d': []}
                rows.append({'scene':r['scene'], 'row_index':r['row_index'], 'event_id':r['event_id'],
                             'instruction':r['instruction'], 'model_raw_output':r['model_raw_output'],
                             'parsed_json':json.dumps(parsed, ensure_ascii=False), 'parse_ok':str(r['parse_ok']),
                             'invalid_reason':r.get('invalid_reason', ''), 'error_message':r['error_message'],
                             'prediction_count':len(parsed.get('referents', [])),
                             'pred_points_json':json.dumps(parsed.get('points_3d', [])),
                             'model_input_image':';'.join(r['images'])})
            pred_path = output / setting / 'predictions.csv'
            write_csv(pred_path, rows)
            if setting == 'experiment2':
                command = [sys.executable, str(ROOT / 'exam2/evaluate_2d_point_grounding.py'), '--manifest', str(output / setting / 'manifest/manifest_all.csv'),
                           '--pred_csv', str(pred_path), '--output_dir', str(output / setting / 'eval'), '--coordinate_mode', 'panel']
                summary_path = output / setting / 'eval/2d_eval_summary.json'
            else:
                command = [sys.executable, str(ROOT / 'exam3_point_grounding/evaluate_point_grounding.py'), '--repo_root', str(ROOT),
                           '--gt_manifest', str(output / setting / 'gt_manifest_eval.csv'), '--pred_csv', str(pred_path),
                           '--output_dir', str(output / setting / 'eval')]
                summary_path = output / setting / 'eval/evaluation_summary.json'
            process = subprocess.run(command, capture_output=True, text=True)
            (output / setting / 'evaluation.log').write_text(process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError('Evaluator failed; see ' + str(output / setting / 'evaluation.log'))
            summaries[setting] = json.loads(summary_path.read_text())
            if setting == 'experiment2':
                sensitivity = []
                for record, row in zip(records, rows):
                    original = m.runner2.extract_json(record['model_raw_output']) if record['parse_ok'] else None
                    mapped = m.runner2.map_prediction_coordinates(original,record['panel_meta'],'source_or_canvas_map')
                    sensitivity.append(dict(row,parsed_json=json.dumps(mapped or {},ensure_ascii=False)))
                alt_dir = output / setting / 'declared_space_sensitivity'
                write_csv(alt_dir / 'predictions.csv',sensitivity)
                alt_command = [sys.executable,str(ROOT / 'exam2/evaluate_2d_point_grounding.py'),
                    '--manifest',str(output / setting / 'manifest/manifest_all.csv'),
                    '--pred_csv',str(alt_dir / 'predictions.csv'),'--output_dir',str(alt_dir / 'eval'),'--coordinate_mode','panel']
                alt_process = subprocess.run(alt_command,capture_output=True,text=True)
                (alt_dir / 'evaluation.log').write_text(alt_process.stdout+alt_process.stderr)
                if alt_process.returncode:
                    raise RuntimeError('Coordinate-space sensitivity evaluator failed')
                summaries['experiment2_declared_space'] = json.loads((alt_dir / 'eval/2d_eval_summary.json').read_text())
    return summaries


def baseline_subset(selection):
    output = {}
    for setting, ids in selection.items():
        selected = set(ids)
        exp = 'exp' + setting[-1]
        output[setting] = {}
        for path in (ROOT / 'paper_experiment_evidence/predictions' / exp).glob('*.csv'):
            rows = [r for r in read_csv(path) if r['unified_id'] in selected]
            output[setting][path.stem] = rows
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path.home() / '.config/trimodal-referring/taomu.json'))
    parser.add_argument('--output', default='closed_models/outputs/taomu_gpt61sol_smoke10')
    parser.add_argument('--selection', help='JSON mapping experiment1/2/3 to scene::row_index lists')
    parser.add_argument('--all', action='store_true', help='Prepare all 4000 samples for selected settings')
    parser.add_argument('--settings', nargs='+', choices=list(DEFAULT_SELECTION), default=list(DEFAULT_SELECTION))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--concurrency', type=int)
    parser.add_argument('--prepare-workers',type=int,default=4)
    parser.add_argument('--validate-projection-archive', action='store_true')
    parser.add_argument('--exp3-evidence-mode', choices=['legacy','coverage_pts'], default='legacy')
    parser.add_argument('--evidence-frames', type=int, default=8)
    parser.add_argument('--clock-contract-root', default=str(ROOT.parent / 'Multiagent/runs/pts_pose_alignment_20261023'))
    parser.add_argument('--exp3-prompt-mode', choices=['legacy','evidence'], default='legacy')
    parser.add_argument('--ablation', choices=['full','no_instruction','no_visual','no_gaze','no_hand'], default='full')
    parser.add_argument('--reuse-prepared', help='Reuse frozen tasks and GT from an earlier output directory; no image changes')
    parser.add_argument('--evaluate-only', action='store_true', help='Reparse cached responses and evaluate without any API calls')
    parser.add_argument('--retry-errors', action='store_true', help='Retry recorded transport errors; never redo valid answers')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    from relay_budget import install_relay_dns_fallback
    install_relay_dns_fallback(config)
    config['retry_errors'] = args.retry_errors
    if args.concurrency is not None:
        if args.concurrency < 1:
            raise SystemExit('--concurrency must be positive')
        config['concurrency'] = args.concurrency
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(args.selection).read_text()) if args.selection else DEFAULT_SELECTION
    if args.all:
        records = read_csv(ROOT / 'paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv')
        selection = {s:[r['unified_id'] for r in records] for s in args.settings}
    selection = {k:v for k,v in selection.items() if k in args.settings}
    for setting, keys in selection.items():
        if len(keys) != len(set(keys)):
            raise SystemExit('Duplicate sample keys in ' + setting)
    total = sum(len(v) for v in selection.values())
    if total == 0:
        raise SystemExit('Selection is empty')
    # Ensure bundled ffmpeg is discoverable without changing the system environment.
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    m = modules()
    started = time.monotonic()
    run_config = {k:v for k,v in config.items() if k != 'api_key'} | {
        'config_path':str(Path(args.config).resolve()), 'selection_count':total,
        'exp3_prompt_mode':args.exp3_prompt_mode, 'ablation_variant':args.ablation,
        'exp3_evidence_mode':args.exp3_evidence_mode,'evidence_frames':args.evidence_frames,
        'endpoint':config['base_url'].rstrip('/') + '/responses', 'protocol':'frozen main prompts; exp2 paired_canvas_map',
        'evidence_status':'reconstructed from original media; not asserted identical to missing original runtime manifests',
        'source_hashes':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [Path(__file__), ROOT / 'exam2/run_qwen3vl_2d_point_grounding.py',
                        ROOT / 'exam2/build_2d_eval_manifest.py', ROOT / 'exam3_point_grounding/build_point_grounding_manifest.py',
                        ROOT / 'exam3_point_grounding/point_grounding_common.py']}}
    history_path = output / 'run_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({'started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'prepare_only':args.prepare_only, 'evaluate_only':args.evaluate_only, **run_config})
    write_json(history_path, history)
    write_json(output / 'run_config.json', run_config)
    if args.reuse_prepared and not (output / 'prepared_tasks.json').exists():
        source = (ROOT / args.reuse_prepared).resolve()
        source_selection = json.loads((source / 'selection.json').read_text())
        if source_selection != selection:
            raise SystemExit('--reuse-prepared selection differs; preserve the original sample selection')
        for name in ['prepared_tasks.json','selection.json','experiment3/gt_manifest_eval.csv','experiment2/manifest/manifest_all.csv']:
            src = source / name
            if src.exists():
                dst = output / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    task_cache = output / 'prepared_tasks.json'
    selection_cache = output / 'selection.json'
    if task_cache.exists() and selection_cache.exists():
        if json.loads(selection_cache.read_text()) != selection:
            raise SystemExit('Output directory already belongs to a different selection; use --output')
        tasks = json.loads(task_cache.read_text())
        if any(t.get('evidence_mode','legacy') != args.exp3_evidence_mode for t in tasks if t['setting']=='experiment3'):
            raise SystemExit('Cached evidence mode differs; use a new output directory')
        if not all(Path(image).is_file() for t in tasks for image in t['images']):
            raise SystemExit('Cached evidence images missing; restore inputs or use a new output directory')
        _, gt = restore_frozen_gt(output)
        print('Reusing frozen prepared inputs:', len(tasks), flush=True)
    else:
        if args.evaluate_only:
            raise SystemExit('--evaluate-only requires prepared_tasks.json and saved responses')
        tasks, gt = prepare(selection, output, m, args.exp3_evidence_mode,args.evidence_frames,args.clock_contract_root,args.prepare_workers)
    if args.validate_projection_archive and 'experiment2' in selection:
        validate_projection_archive(output)
    from protocols import apply_protocols
    legacy_renderer = None
    if args.exp3_prompt_mode == 'legacy' and args.ablation != 'full':
        legacy_renderer = load_module('closed_ablations', 'exam3_point_grounding/ablation_inputs.py').render_prompt
    tasks = apply_protocols(tasks, args.exp3_prompt_mode, args.ablation, legacy_renderer)
    if args.prepare_only:
        print('Prepared', len(tasks), 'requests without calling API', flush=True)
        return
    results = []
    if args.evaluate_only:
        for task in tasks:
            path = output / task['setting'] / 'raw' / task['scene'] / f"row_{task['row_index']}.json"
            result = json.loads(path.read_text())
            if result.get('response'):
                ok, parsed, extra = parse_task(task, result['model_raw_output'], m)
                if result['response'].get('status') != 'completed':
                    ok, parsed = False, {}
                result.update(parse_ok=ok, parsed_json=parsed, **extra)
                if task['setting'] == 'experiment1':
                    result['parsed_response'] = parsed
                write_json(path, result)
            results.append(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(config.get('concurrency', 3))) as pool:
            futures = [pool.submit(run_task, task, output, config, m) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda r:(r['setting'], r['scene'], r['row_index']))
    summaries = evaluate(results, gt, output, m)
    write_json(output / 'baseline_subset_records.json', baseline_subset(selection))
    usage_totals = {k:sum((r.get('usage') or {}).get(k, 0) for r in results) for k in ['input_tokens', 'output_tokens', 'total_tokens']}
    report = {'model_requested':config['model'], 'served_models':sorted({r['served_model'] for r in results if r.get('served_model')}),
              'requests':len(results), 'valid_outputs':sum(r['parse_ok'] for r in results),
              'transport_failures':sum(bool(r['error_message']) for r in results), 'usage':usage_totals,
              'elapsed_seconds':time.monotonic()-started, 'setting_counts':{s:sum(r['setting']==s for r in results) for s in selection},
              'metrics':summaries, 'limitation':'Reconstructed evidence; subset results are not full-benchmark estimates.',
              'exp3_prompt_mode':args.exp3_prompt_mode, 'ablation_variant':args.ablation}
    write_json(output / 'smoke_summary.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'metrics'}, ensure_ascii=False, indent=2), flush=True)
    if report['valid_outputs'] != report['requests']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
