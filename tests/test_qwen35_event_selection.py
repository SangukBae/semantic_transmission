import numpy as np
import cv2
import pytest

from semantic_transmission import qwen35_event_selection as e


def scan_images(tmp_path, images):
    paths=[]
    for i,im in enumerate(images):
        p=tmp_path/f'{i}.png';cv2.imwrite(str(p),im);paths.append(p)
    return e.detect(paths)


def test_single_frame_small_event_with_identical_endpoints(tmp_path):
    images=[np.zeros((160,288,3),np.uint8) for _ in range(9)]
    images[2][64:72,64:72]=255
    scan=scan_images(tmp_path,images)
    assert scan['transitions']==8
    assert scan['rows'][1]['triggered'] and scan['rows'][2]['triggered']
    assert any(a<=1 and b>=3 for a,b in e.windows(scan))
    class Session:
        def call_event(self,ids,prompt,validator,regions):
            assert 2 in ids and 0 in ids and any(r['frame']==2 for r in regions)
            return validator(dict(decision='resolved',keys=[2],events=[dict(frame=2,reason='Small bright object appears')]))
        def call(self,role,ids,prompt,validator,tokens):
            assert ids==[2,8]  # The earlier transient already replaced the accepted reference.
            return validator(dict(decision='changed',kind='entry_exit',reference_state='Bright object',
                                  candidate_state='Empty scene',evidence_frames=ids))
    keys,_=e.select(9,Session(),scan)
    assert keys==[0,2,8]


def test_static_and_uniform_exposure_do_not_trigger_local(tmp_path):
    base=np.full((160,288,3),60,np.uint8)
    scan=scan_images(tmp_path,[base,base,base+2,base+2,base+2])
    assert not any(r['triggered'] for r in scan['rows'])


def test_camera_translation_is_removed_but_local_change_remains():
    rng=np.random.default_rng(1)
    gray=cv2.GaussianBlur(rng.integers(0,256,(160,288),dtype=np.uint8),(5,5),1)
    a=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    b=cv2.warpAffine(a,np.float32([[1,0,3],[0,1,0]]),(288,160))
    clean=e.pair_signals(a,b)
    b[60:76,100:116]=255
    changed=e.pair_signals(a,b)
    assert clean['local']<12 and changed['local']>32
    assert clean['newly_visible_fraction']>0


def test_accumulated_change_and_no_temporal_skip(tmp_path):
    images=[]
    for i in range(9):
        im=np.zeros((160,288,3),np.uint8);im[64:80,64:80]=i*4;images.append(im)
    scan=scan_images(tmp_path,images)
    assert [r['frame'] for r in scan['rows']]==list(range(1,9))
    assert 'accumulated_change' in scan['rows'][3]['reasons']


@pytest.mark.parametrize('keys',[[3,2],[2,2],[99],[True]])
def test_reject_invalid_event_ids(keys):
    with pytest.raises(ValueError):
        e.validate_event(dict(decision='resolved',keys=keys,events=[]),[1,2,3])


def test_uncertain_is_not_silent_skip(tmp_path):
    images=[np.zeros((80,144,3),np.uint8),np.full((80,144,3),255,np.uint8)]
    scan=scan_images(tmp_path,images)
    class Session:
        def call_event(self,*args):return dict(decision='uncertain',keys=[],events=[])
    with pytest.raises(RuntimeError,match='uncertain'):
        e.select(2,Session(),scan)


def test_opt_in_settings_and_profile():
    from semantic_transmission.qwen35_pairs_video import selection_settings
    from semantic_transmission.bsu_gen import profile,parser
    assert selection_settings(240)['selection_policy']=='semantic'
    assert selection_settings(240,'events')['selector']==e.POLICY
    assert profile('events')['selection_revision']==e.POLICY
    assert 'selection_revision' not in profile()
    assert parser().parse_args(['profile','--selection-policy','events']).selection_policy=='events'


def test_schema_fallback_and_resume_preserve_real_image_identity(tmp_path):
    import json
    from semantic_transmission.artifacts import sha256
    from semantic_transmission.qwen35_backend import MODEL_ID,REVISION
    for i in (0,1):cv2.imwrite(str(tmp_path/f'{i}.png'),np.full((64,64,3),i*255,np.uint8))
    plan=dict(signature='test',max_images=16,fps=24,source_frames=str(tmp_path),
              source_frame_hashes={str(i):sha256(tmp_path/f'{i}.png') for i in (0,1)})
    class Model:
        runtime=dict(model_id=MODEL_ID,revision=REVISION,quantization='nf4')
        calls=0
        def generate(self,images,labels,prompt,tokens):
            self.calls+=1
            key=1 if 'SCHEMA REVIEW:' in prompt else 0
            value=dict(decision='resolved',keys=[key],events=[dict(frame=key,reason='Bright scene')])
            return dict(raw_text=json.dumps(value),truncated=False,model_runtime=self.runtime,seconds=0)
    model=Model();session=e.make_session(tmp_path,plan,model)
    validate=lambda v:e.validate_event(v,[1])
    result=session.call_event([0,1],'SELECTABLE WINDOW IDS=[1]',validate,[])
    assert result['keys']==[1] and model.calls==4
    assert len(list((tmp_path/'failed_attempts/event_window').glob('*.json')))==3
    resumed=e.make_session(tmp_path,plan,model)
    assert resumed.call_event([0,1],'SELECTABLE WINDOW IDS=[1]',validate,[])==result
    assert model.calls==4 and resumed.calls[0]['reused']
