"""GT-free event coverage with verified decoded PTS and shared camera-pose sampling.

Reuses Multiagent's camera interpolation/clock contract; no GT enters selection.
Container PTS remains distinct from unmeasured render/exposure time.
"""
from __future__ import annotations
import bisect
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'exam3_point_grounding'))
from select_target_free_evidence import (load_multimodal_samples, collect_timed_samples,
    effective_video_time_offset, nearest_sample, evidence_score, gaze_hit, gaze_direction, hand_hit)
from point_grounding_common import EvidenceFrame


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_coverage(candidates, start, end, budget):
    """Reserve every time bin; stable cues break ties, never target correspondence."""
    if end < start or budget < 1:
        raise ValueError('Invalid temporal interval or frame budget')
    result = []
    for i in range(budget):
        lo=start+(end-start)*i/budget
        hi=start+(end-start)*(i+1)/budget
        middle=(lo+hi)/2
        options=[c for c in candidates if lo <= c['video_time'] <= hi and c not in result]
        if not options:
            continue
        best=max(options,key=lambda c:(c['score'], -abs(c['video_time']-middle)))
        result.append(best)
    return sorted(result,key=lambda c:c['video_time'])


def extract_pts(video, requested, path, pts_table):
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata=path.with_suffix('.pts.json')
    if path.exists() and metadata.exists():
        saved=json.loads(metadata.read_text())
        if saved['source_sha256']==sha(path) and abs(saved['requested_seek_seconds']-requested)<1e-6:
            return saved
        raise ValueError('Cached frame timing or image hash changed')
    command=['ffmpeg','-hide_banner','-loglevel','info','-copyts','-ss',f'{requested:.6f}',
             '-i',str(video),'-frames:v','1','-q:v','2','-vf','showinfo','-y',str(path)]
    proc=subprocess.run(command,capture_output=True,text=True,timeout=60)
    match=re.search(r'\bn:\s*0\s+pts:\s*[-\d]+\s+pts_time:([-\d.eE+]+)',proc.stderr)
    if proc.returncode or not match or not path.exists():
        raise RuntimeError('Failed to decode and bind an actual video frame PTS')
    pts=float(match.group(1))
    index=min(range(len(pts_table)),key=lambda i:abs(pts_table[i]-pts))
    if abs(pts_table[index]-pts)>.00002:
        raise ValueError('Decoded showinfo PTS disagrees with ffprobe frame table')
    record={'status':'VERIFIED_CONTAINER_PTS_SEQUENCE','requested_seek_seconds':requested,
            'actual_container_pts_seconds':pts_table[index],'source_frame_index':index,
            'source_sha256':sha(path),'pts_minus_requested_seconds':pts_table[index]-requested,
            'exposure_time_calibration':'unverified; container PTS is not render exposure time'}
    metadata.write_text(json.dumps(record,indent=2))
    return record


def select_coverage_frames(api_row, directory, budget=8, contract_root=None, scene=None, row_index=None):
    sys.path.insert(0,str(ROOT.parent / 'Multiagent'))
    from MultiAgent.frame_pose_alignment import (CameraPoseTimeline, DEFAULT_CONFIG,
        track_pts_features, MotionScorer, fit_offset, evaluate_fits)
    from MultiAgent.head_motion_diagnostics import camera_records, probe_pts
    samples=load_multimodal_samples(Path(api_row['json_path']))
    timed=collect_timed_samples(samples)
    timeline=CameraPoseTimeline(camera_records(SimpleNamespace(timed_samples=timed)))
    original_offset,source=effective_video_time_offset(Path(api_row['json_path']),Path(api_row['video_path']),samples)
    selected_offset=original_offset
    clock_status='ORIGINAL_CLOCK_UNVALIDATED'
    source_hashes={'video_sha256':sha(api_row['video_path']),'telemetry_sha256':sha(api_row['json_path'])}
    contract = None
    if contract_root:
        contract_path=Path(contract_root)/scene/str(row_index)/'frame_pose_contract.json'
        if contract_path.exists():
            contract=json.loads(contract_path.read_text())
            if contract['source_hashes']!=source_hashes:
                raise ValueError('Clock fit input hashes differ from current video/telemetry')
            clock=contract['clock']
            if clock['status']=='VALIDATED_CANDIDATE':
                selected_offset=clock['offset_seconds'];clock_status='VALIDATED_CANDIDATE'
            else:
                selected_offset=clock.get('original_offset_seconds',original_offset)
                clock_status='ORIGINAL_CLOCK_FALLBACK_PTS_POSE'
    if contract is None:
        # New events: fit only RGB background motion versus camera telemetry, never GT/gaze.
        fit_path = Path(directory) / 'clock_fit.json'
        fit_path.parent.mkdir(parents=True,exist_ok=True)
        if fit_path.exists():
            fitted = json.loads(fit_path.read_text())
            if fitted['source_hashes'] != source_hashes:
                raise ValueError('Cached clock fit source changed')
        else:
            config = dict(DEFAULT_CONFIG)
            try:
                tracks = track_pts_features(api_row['video_path'],'ffmpeg','ffprobe',config)
                scorer = MotionScorer(tracks['pairs'],timeline)
                candidate = fit_offset(scorer,'train',config)
                reverse = fit_offset(scorer,'validation',config)
                if candidate.get('failure'):
                    clock = {'status':'CLOCK_UNCERTAIN','original_offset_seconds':original_offset,
                             'rejection_reasons':[candidate['failure']]}
                else:
                    clock = evaluate_fits(scorer,candidate,original_offset,None,reverse,config)
                fitted = {'source_hashes':source_hashes,'clock':clock,'fit_uses_gt':False,'fit_uses_gaze':False,
                          'fit_config':config,'candidate_fit':candidate,'reverse_fit':reverse}
            except Exception as exc:
                fitted = {'source_hashes':source_hashes,'clock':{'status':'CLOCK_UNCERTAIN',
                          'original_offset_seconds':original_offset,'rejection_reasons':[type(exc).__name__+': '+str(exc)]},
                          'fit_uses_gt':False,'fit_uses_gaze':False}
            fit_path.write_text(json.dumps(fitted,indent=2))
        clock = fitted['clock']
        if clock['status']=='VALIDATED_CANDIDATE':
            selected_offset=clock['offset_seconds'];clock_status='VALIDATED_CANDIDATE'
        else:
            clock_status='ORIGINAL_CLOCK_FALLBACK_PTS_POSE'
    pts_table,_=probe_pts(api_row['video_path'],'ffprobe')
    duration=pts_table[-1]
    start=max(0.,float(api_row['t_start'])+selected_offset)
    end=min(duration-.05,float(api_row['t_end'])+selected_offset)
    if end-start<.5:
        start,end=0.,max(0.,duration-.05)
    candidates=[]
    # A dense, fixed event grid independent of transcript correctness and target count.
    count=max(budget*4,int((end-start)/.25)+1)
    for i in range(count):
        video_time=start+(end-start)*(i+.5)/count
        cue_time,sample=nearest_sample(timed,video_time-selected_offset)
        score,reason,_,_=evidence_score(timed,cue_time,sample,.3)
        candidates.append({'video_time':video_time,'score':score,'reason':reason})
    selected=choose_coverage(candidates,start,end,budget)
    frames=[];audit=[];used_pts=set()
    for c in selected:
        panel=f'P{len(frames)+1}'
        path=Path(directory)/(panel+'.jpg')
        binding=extract_pts(api_row['video_path'],c['video_time'],path,pts_table)
        pts=binding['actual_container_pts_seconds']
        if pts in used_pts:
            continue
        query=pts-selected_offset
        pose=timeline.pose_at(query)
        cue_time,sample=nearest_sample(timed,query)
        aligned=dict(sample)
        cue_fresh=abs(cue_time-query)<=.125
        if not cue_fresh:
            aligned['eyeGaze']={};aligned['handData']={}
        if pose['status']=='POSE_AVAILABLE':
            aligned['cameraPosition']=dict(zip('xyz',pose['position_world']))
            aligned['cameraRotation']=dict(zip('xyzw',pose['rotation_xyzw']))
            aligned['cameraFOV']=pose['fov_vertical_degrees']
        else:
            # Keep the RGB observation, explicitly withhold unavailable camera pose/cues.
            aligned.pop('cameraPosition',None);aligned.pop('cameraRotation',None);aligned.pop('cameraFOV',None)
            aligned['eyeGaze']={};aligned['handData']={}
        used_pts.add(pts)
        frames.append(EvidenceFrame(panel,str(path),pts,cue_time,c['score'],
            'coverage_bin;'+c['reason'],gaze_hit(aligned) is not None or gaze_direction(aligned) is not None,
            hand_hit(aligned) is not None,aligned))
        audit.append({**binding,'panel_id':panel,'camera_pose':pose,'clock_status':clock_status,
                      'clock_offset_seconds':selected_offset,'telemetry_query_seconds':query,
                      'cue_sample_seconds':cue_time,'cue_age_seconds':cue_time-query,'cue_fresh':cue_fresh})
    if not frames:
        raise RuntimeError('No verified observation frames available')
    return frames, {'version':'event_coverage_pts_pose_v1','selection_uses_gt':False,
                    'selection_uses_target_count':False,'selection_uses_asr':False,
                    'original_offset_seconds':original_offset,'original_offset_source':source,
                    'selected_offset_seconds':selected_offset,'clock_status':clock_status,
                    'source_hashes':source_hashes,'budget':budget,'frames':audit}
