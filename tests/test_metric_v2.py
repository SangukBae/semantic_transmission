import numpy as np
import pytest

from semantic_transmission.motion_metric import describe_flow, score_motion, match_events, FlowExtractor
from semantic_transmission.object_metric import score_tracks, associate_frames
from semantic_transmission.metric_v2_cases import render_scene, temporal_variant


def detection(x=10, feature=(1., 0.)):
    mask = np.zeros((32, 64), bool)
    mask[10:20, x:x+10] = True
    return {"mask": mask, "feature": np.asarray(feature)}


def test_static_is_abstention_not_perfect_motion():
    desc = describe_flow(np.zeros((8, 32, 64, 2), np.float32))
    score = score_motion(desc, desc)
    assert score["mte_tail"] is None
    assert score["mte_window_coverage"] == 0


def test_opposed_objects_survive_direction_histogram():
    flow = np.zeros((8, 96, 96, 2), np.float32)
    flow[:, 3:10, 3:10, 0] = 2
    flow[:, 17:24, 3:10, 0] = -2
    d = describe_flow(flow)
    assert np.linalg.norm(d["hist"]) > 0
    assert np.linalg.norm(d["hist"]) > 100 * np.linalg.norm(d["vector"])


def test_equal_motion_zero_and_opposite_detected():
    flow = np.zeros((12, 96, 96, 2), np.float32)
    flow[:, 25:65, 25:65, 0] = 3
    a, b = describe_flow(flow), describe_flow(-flow)
    assert score_motion(a, a)["mte_tail"] == pytest.approx(0)
    assert score_motion(a, b)["mte_tail"] > .5


def test_camera_translation_is_separate():
    a = describe_flow(np.zeros((8, 32, 64, 2), np.float32))
    flow = np.zeros((8, 32, 64, 2), np.float32)
    flow[..., 0] = 2
    b = describe_flow(flow)
    assert score_motion(a, b)["mte_tail"] is None
    assert score_motion(a, b)["camera_displacement_error"] > 0


def test_events_type_time_one_to_one():
    a = [{"time_s": 1., "type": "turn"}, {"time_s": 1.2, "type": "turn"}]
    b = [{"time_s": 1.1, "type": "turn"}, {"time_s": 1.2, "type": "stop"}]
    r = match_events(a,b)
    assert r["event_miss_rate"] == .5
    assert r["event_extra_rate"] == .5
    assert match_events(a,[{"time_s":2.,"type":"turn"}])["event_miss_rate"] == 1


def test_camera_rotation_does_not_cancel():
    y,x = np.mgrid[:64,:64]
    flow = np.stack((-(y-31.5)*.02,(x-31.5)*.02),-1).astype(np.float32)
    a = describe_flow(np.zeros((8,64,64,2),np.float32))
    b = describe_flow(np.repeat(flow[None],8,axis=0))
    assert score_motion(a,b)["camera_displacement_error"] > 0


def test_f1_uses_common_exposure_and_keeps_short_errors():
    a = [{t: detection() for t in range(8)}]
    b = [{t: detection() for t in range(8) if t != 3}]
    r = score_tracks(a,b)
    assert r["otf_f1"] == pytest.approx(14 / 15)
    assert r["oor"] == .125
    assert r["hor"] == 0
    assert r["omission_intervals"] == [{"track":0,"start_s":.375,"end_s":.5,"frames":1}]
    assert r["long_omission_events"] == 0


def test_track_fragmentation_cannot_reuse_reference_identity():
    a = [{t: detection() for t in range(8)}]
    b = [{t: detection() for t in range(4)}, {t: detection() for t in range(4,8)}]
    r = score_tracks(a,b)
    assert len(r["track_pairs"]) == 1
    assert r["otf_f1"] == .5


def test_empty_and_total_omission():
    assert score_tracks([],[])["otf_error"] is None
    r = score_tracks([{t:detection() for t in range(8)}],[])
    assert r["otf_error"] == 1
    assert r["hor"] is None


def test_birth_and_death_not_errors_if_shared():
    a = [{t:detection() for t in range(2,6)}]
    assert score_tracks(a,a)["otf_f1"] == 1


def test_tracking_keeps_distinct_identities():
    detections = [[detection(4+t,(1.,0.)), detection(40-t,(0.,1.))] for t in range(8)]
    tracks = associate_frames(detections)
    assert len(tracks) == 2
    assert [len(t) for t in tracks] == [8,8]


def test_texture_controls_keep_renderer_truth_exact():
    a, ta = render_scene(1234567)
    b, tb = render_scene(1234567, appearance=2)
    assert np.array_equal(ta["masks"],tb["masks"])
    assert np.array_equal(ta["centers"],tb["centers"])
    assert not np.array_equal(a,b)


def test_delay_is_not_aligned_away():
    a,_ = render_scene(1234567)
    b, truth = temporal_variant(a,"lag",.25)
    assert truth["source_index_map"][:5] == [0,0,0,0,0]
    assert not np.array_equal(a,b)


def test_nonfinite_flow_rejected():
    with pytest.raises(ValueError):
        describe_flow(np.full((4,32,32,2),np.nan))


def test_cpu_flow_backend_is_explicit():
    a,_ = render_scene(1234567, count=4)
    f = FlowExtractor("farneback","cpu")(a)
    assert f.shape == (3,192,320,2)


def test_dense_pair_sampling_preserves_playback_and_missing_tail(tmp_path):
    import cv2
    from semantic_transmission.pair_inputs import load_pair
    from semantic_transmission.artifacts import sha256
    def save(path, values):
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),24,(32,32))
        for v in values:writer.write(np.full((32,32,3),v,np.uint8))
        writer.release()
    a,b=tmp_path/"a.mp4",tmp_path/"b.mp4"
    save(a,range(0,96,4));save(b,[0]*3+list(range(0,60,4)))
    row=dict(source=str(a),source_sha256=sha256(a),reconstruction=str(b),reconstruction_sha256=sha256(b))
    x,y,info=load_pair(row,width=32,height=32,sample_fps=8)
    assert info["source_indices"]==[0,3,6,9,12,15]
    assert info["reconstruction_indices"]==info["source_indices"]
    assert info["sample_coverage"]==.75
    assert np.mean(abs(x.astype(float)-y))>5
    with pytest.raises(ValueError,match="sample FPS"):
        load_pair(row,sample_fps=30)


def test_identity_baseline_penalizes_fragmentation_beyond_frame_detection():
    from semantic_transmission.tracking_baselines import identity_baselines
    a=[{t:detection() for t in range(8)}]
    b=[{t:detection() for t in range(4)},{t:detection() for t in range(4,8)}]
    score=identity_baselines(a,b)
    assert score["idf1_mask_error"]==.5
    assert score["frame_mask_error"]==0
    assert score["idtp"]==4 and score["idfp"]==4 and score["idfn"]==4
