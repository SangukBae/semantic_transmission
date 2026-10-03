import json

import pytest

from semantic_transmission.auto_extraction import (
    digest, infer_json, parse_json, proposal_windows, seal, unseal,
    validate_caption, validate_events, verify_video, pixel_change_events,
)
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.assisted_captions import sample_indices
from semantic_transmission.hybrid_selection import choose_keys


def test_sampled_timeline_covers_every_six_fps_frame_and_endpoints():
    for count in [2, 95, 96, 97, 98, 2355, 24000]:
        windows=list(proposal_windows(count))
        observed={i for window in windows for i in window}
        assert observed == set(range(0,count,4)) | {count-1}
        assert all(len(w)<=13 for w in windows)
        assert all(a[-1]==b[0] for a,b in zip(windows,windows[1:]))


def test_model_cannot_invent_unobserved_event_timestamps():
    event=dict(before=0,after=4,kind="occlusion",description="object becomes obscured")
    validate_events(dict(events=[event],uncertainties=[]),[0,4,8])
    for fields in [dict(after=5),dict(before=4),dict(before=True),dict(kind="hallucinated_kind")]:
        with pytest.raises(ValueError):
            validate_events(dict(events=[dict(event,**fields)],uncertainties=[]),[0,4,8])


def test_pixel_guard_detects_partial_overlay_change_without_flooding_smooth_changes():
    import numpy as np
    frames=np.zeros((60,27,48,3),dtype=np.uint8)
    frames[30:,:,:8,:]=255
    events=pixel_change_events(frames)
    assert [(e["before"],e["after"]) for e in events]==[(29,30)]
    smooth=np.broadcast_to(np.arange(60,dtype=np.uint8)[:,None,None,None],(60,27,48,3)).copy()
    assert pixel_change_events(smooth)==[]


def test_invalid_or_empty_caption_is_never_silently_filled():
    for caption in ["", "hello\nworld", "word "*81, None]:
        with pytest.raises(ValueError):validate_caption(dict(caption=caption,uncertainties=[]))
    with pytest.raises(ValueError):validate_caption(dict(caption="Visible object",uncertainties="none"))
    assert parse_json('```json\n{"events": []}\n```')=={"events":[]}
    with pytest.raises(ValueError):parse_json('{"events": []} speculative extra text')


def test_inference_resume_is_bound_to_exact_inputs_and_retries_invalid_json(tmp_path):
    class VLM:
        calls=0
        def generate(self,*args):
            self.calls+=1
            return dict(raw_text="invalid" if self.calls==1 else json.dumps(dict(caption="A visible shape.",uncertainties=[])),
                        truncated=False,seconds=.1)
    model=VLM();path=tmp_path/"caption.json"
    value=infer_json(model,path,{"frame_hash":"a"},[],[],"prompt",validate_caption)
    assert model.calls==2
    assert (tmp_path/"failed_attempts/caption_0.json").exists()
    assert infer_json(model,path,{"frame_hash":"a"},[],[],"prompt",validate_caption)==value
    assert model.calls==2
    with pytest.raises(ValueError,match="identity changed"):
        infer_json(model,path,{"frame_hash":"b"},[],[],"prompt",validate_caption)
    broken=json.loads(path.read_text());broken["result"]["caption"]="Modified"
    write_json(path,broken)
    with pytest.raises(ValueError,match="checksum mismatch"):unseal(path)


def test_completed_video_validation_rejects_temporal_misalignment(tmp_path):
    """A valid-looking caption on the wrong source interval must never pass."""
    config=dict(signature="config",max_gap_frames=24,psss_threshold=.35)
    source=dict(id="test",dataset="WebVid",frames=26,source_frame_hashes={},source_frames=str(tmp_path/"source"))
    frames=tmp_path/"source";frames.mkdir()
    for i in range(26):
        (frames/f"{i}.png").write_bytes(f"frame{i}".encode())
        source["source_frame_hashes"][str(i)]=sha256(frames/f"{i}.png")
    dest=tmp_path/"webvid/test";dest.mkdir(parents=True)
    candidates=[dict(frame=0,mandatory=True),dict(frame=25,mandatory=True)]
    write_json(dest/"proposals.json",seal(dict(config_signature="config",candidates=candidates)))
    keys,records=choose_keys(26,[0,25],[0,25],lambda *_:pytest.fail("unexpected SKEM"))
    (dest/"selected_frames").mkdir()
    for i in keys:(dest/"selected_frames"/f"{i:05d}.png").write_bytes((frames/f"{i}.png").read_bytes())
    write_json(dest/"selection.json",seal(dict(identity=dict(config_signature="config",proposals_sha256=sha256(dest/"proposals.json")),
        indices=keys,records=records,keyframe_png_bytes=1)))
    write_json(dest/"keyframes.json",dict(indices=keys))
    identity=dict(config_signature="config",selection_sha256=sha256(dest/"selection.json"))
    rows=[]
    for n,(a,b) in enumerate(zip(keys,keys[1:])):
        row=dict(segment=n,start=a,end_exclusive=b,source_indices=sample_indices(a,b),caption="A visible shape.",
                 uncertainties=[],removed_claims=[])
        rows.append(row)
        item_id=dict(identity,segment=n,start=a,end_exclusive=b,source_indices=sample_indices(a,b),
                     frame_hashes=[source["source_frame_hashes"][str(i)] for i in sample_indices(a,b)])
        write_json(dest/"captions_draft"/f"{n:05d}.json",seal(dict(identity=dict(item_id,task="draft"),result=dict(row))))
        write_json(dest/"captions_review"/f"{n:05d}.json",seal(dict(identity=dict(item_id,task="visual_review",draft_sha256=digest(row)),result=dict(row))))
    write_json(dest/"captions.json",seal(dict(identity=identity,captions=rows)))
    assert verify_video(tmp_path,config,source)["structural_validation"]=="PASS"
    rows[0]["source_indices"]=[0,1,2,3]
    write_json(dest/"captions.json",seal(dict(identity=identity,captions=rows)))
    with pytest.raises(ValueError,match="timeline mismatch"):verify_video(tmp_path,config,source)
