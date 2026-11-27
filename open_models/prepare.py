"""Prepare identical inputs once for all models; never call an inference API."""
import concurrent.futures
import json
import os
from pathlib import Path
from types import SimpleNamespace

from common import ROOT, RUN, SETTINGS, batch_name, batches, digest, protocol, write

SOURCE = ROOT / 'exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv'


def prepare_batch(item):
    stage, offset, keys = item
    out = RUN / 'inputs' / batch_name(stage, offset)
    if (out / 'READY.json').exists():
        return str(out)
    out.mkdir(parents=True, exist_ok=True)
    m = protocol.modules()
    # The existing experiment-1 builder supplies the original prompt and <=16 frames.
    tasks, _ = protocol.prepare({'experiment1': keys}, out, m)
    rows = [r for r in protocol.read_csv(SOURCE) if r['scene'] + '::' + r['row_index'] in set(keys)]
    grouped = m.runner2.group_manifest(rows)
    options = SimpleNamespace(panel_context_mode='paired_crop', gaze_crop_ratio=.35,
                              crop_output_size=768, overwrite=False)
    for key in keys:
        scene, index = key.split('::')
        event_rows = grouped[(scene, int(index))]
        paths, metadata = m.runner2.collect_model_panels((scene, int(index)), event_rows)
        paths, metadata = m.runner2.expand_panel_context_images(
            (scene, int(index)), paths, metadata, out / 'experiment2/model_inputs', options)
        row = protocol.read_csv(ROOT / 'data' / (scene + '_api_input.csv'))[int(index)]
        task = protocol.task_base('experiment2', key, row, paths,
            m.runner2.build_prompt(event_rows, metadata, 'expected_count'), m.runner2.SYSTEM_PROMPT)
        task.update(panel_meta=metadata, coordinate_policy='paired_canvas_map')
        tasks.append(task)
    m.builder2.write_manifest(out / 'experiment2/manifest/manifest_all.csv', rows)
    image_audits = {}
    for task in tasks:
        task['image_audit'] = []
        for path in task['images']:
            if path not in image_audits:
                with protocol.Image.open(path) as image:
                    image_audits[path] = {'path': path, 'sha256': digest(path), 'size': list(image.size)}
            task['image_audit'].append(image_audits[path])
        task['prompt_sha256'] = protocol.sha(task['prompt_text'].encode())
    assert len(tasks) == len(keys) * 2
    write(out / 'prepared_tasks.json', tasks)
    write(out / 'selection.json', {s: keys for s in SETTINGS})
    write(out / 'READY.json', {'events': len(keys), 'tasks': len(tasks),
        'tasks_sha256': digest(out / 'prepared_tasks.json'), 'stage': stage})
    return str(out)


def preflight():
    rows = protocol.read_csv(SOURCE)
    archive = protocol.read_csv(ROOT / 'paper_experiment_evidence/manifests/experiment2_projected_gt_manifest.csv')
    key = lambda r: (r['scene'], r['row_index'], r['referent_name'], r['panel_id'])
    old = {key(r): r for r in archive}
    assert len(rows) == len(archive) == 26422
    assert all(key(r) in old and all(r[k] == old[key(r)][k]
        for k in ['projection_valid', 'gt_x', 'gt_y']) for r in rows)
    assert all(Path(p).is_file() for p in {r['frame_path'] for r in rows})
    for scene in sorted({r['scene'] for r in rows}):
        api = protocol.read_csv(ROOT / 'data' / (scene + '_api_input.csv'))
        assert all(r['instruction_text'].strip() and Path(r['video_path']).is_file() for r in api)
    write(RUN / 'input_preflight.json', {'projected_rows': len(rows),
        'events': len({(r['scene'], r['row_index']) for r in rows}),
        'projection_archive_exact_match': True, 'all_source_frames_present': True,
        'source_manifest': str(SOURCE), 'source_manifest_sha256': digest(SOURCE)})


def main():
    os.environ['PATH'] = str(Path(__import__('sys').executable).parent) + os.pathsep + os.environ['PATH']
    preflight()
    jobs = batches()
    write(RUN / 'input_progress.json', {'status': 'PREPARING', 'batches_total': len(jobs)})
    # Smoke inputs become available before the full queue; workers can start immediately.
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for done, out in enumerate(pool.map(prepare_batch, jobs), 1):
            print('[ready]', done, '/', len(jobs), out, flush=True)
            write(RUN / 'input_progress.json', {'status': 'PREPARING', 'batches_ready': done,
                'batches_total': len(jobs), 'last_batch': out})
    write(RUN / 'input_progress.json', {'status': 'COMPLETE', 'batches_ready': len(jobs), 'batches_total': len(jobs)})


if __name__ == '__main__':
    main()
