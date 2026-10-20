from semantic_transmission.bsu_event_validation import match


def test_one_reference_cannot_reward_many_predictions():
    events=[dict(frame=10,acceptable_start=8,acceptable_end=12)]
    m=match([8,9,10,11,12],events)
    assert m['tp']==1 and m['precision']==.2 and m['recall']==1


def test_overlapping_ranges_need_maximum_matching():
    events=[dict(frame=10,acceptable_start=9,acceptable_end=11),
            dict(frame=9,acceptable_start=9,acceptable_end=9)]
    m=match([9,11],events)
    assert m['tp']==2 and m['f1']==1


def test_tolerance_and_empty_predictions():
    events=[dict(frame=10,acceptable_start=8,acceptable_end=12)]
    assert match([12],events,1)['tp']==0
    assert match([12],events,2)['tp']==1
    assert match([],events)['recall']==0
    assert match([],[])['f1']==1


def test_event_policy_reaches_bsu_generation_worker(monkeypatch):
    from semantic_transmission import bsu_gen,bsu_webvid
    calls=[]
    monkeypatch.setattr(bsu_webvid,'command',lambda cmd,*_:calls.append(cmd))
    with bsu_gen.generative_pipeline('events'):
        bsu_webvid.command(['python','-m','semantic_transmission.bsu_webvid_base','--prepared','source.json'],None,None)
    assert calls[0][-2:]==['--selection-policy','events']


def test_caption_stage_receives_backbone_with_task_roles_not_crop_wrapper():
    from types import SimpleNamespace
    from semantic_transmission.qwen35_pairs_video import caption_backbone
    class Backbone:
        config={}
        def generate(self,*,task):return self.config['caption_adapter'],task
    base=Backbone();session=SimpleNamespace(model=SimpleNamespace(base=base))
    model=caption_backbone(session,dict(caption_adapter='pinned-caption-lora'))
    assert model is base
    assert model.generate(task='draft')==('pinned-caption-lora','draft')
