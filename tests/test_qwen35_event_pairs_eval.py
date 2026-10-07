from pathlib import Path

from semantic_transmission.qwen35_event_pairs_eval import CODES


def test_extractor_has_no_reference_or_evaluator_dependency():
    import ast
    from semantic_transmission import qwen35_event_pairs
    tree=ast.parse(Path(qwen35_event_pairs.__file__).read_text())
    modules=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):modules.extend([node.module or '',*(a.name for a in node.names)])
        elif isinstance(node,ast.Import):modules.extend(a.name for a in node.names)
    assert all('eval' not in m and 'reference' not in m for m in modules)
    assert len(set(CODES.values()))==2 and 'M71' not in CODES.values()


def test_caption_cache_does_not_reuse_different_source_pixels(monkeypatch,tmp_path):
    from semantic_transmission import qwen35_event_pairs as ep
    import json
    session=ep.Session.__new__(ep.Session)
    session.plan=dict(source_frame_hashes={'10':'actual-image'})
    tail='\nVALID SOURCE FRAME IDS: '+json.dumps([10])+'. Use these exact IDs, never image positions or example numbers. Use compact JSON without indentation.'
    session.parent_cache=[(tmp_path/'call.json',dict(identity=dict(ids=[10],tokens=640,
        prompt='caption'+tail,frame_hashes=['different-image'])))]
    monkeypatch.setattr(ep.recovery.RecoverySession,'call',lambda *a,**k: 'fresh inference')
    assert session.call('state_caption',[10],'caption',lambda x:x)=='fresh inference'
