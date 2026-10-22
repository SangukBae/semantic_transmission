from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from semantic_transmission import sparc_selector_train as old
from semantic_transmission import sparc_selector_train_v2 as train
from semantic_transmission.sparc_selector_cnn import (
    SpatialTemporalSelector, pair_context, state_context, select)
from semantic_transmission.sparc_selector_data import save, sha


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)

    def test_joint_spatial_temporal_gradients_and_reverse_action(self):
        model = SpatialTemporalSelector(width=32, heads=4, layers=2, dropout=0)
        pixels = torch.randint(0,255,(12,3,32,48),dtype=torch.uint8)
        forward = dict(before=[0,11],after=[0,5,11])
        backward = dict(before=[0,5,11],after=[0,11])
        c,g,s = pair_context(12,[forward,backward],[0,11],[])
        y = model(pixels,c,g,s)
        self.assertEqual(y.shape,(2,5))
        torch.testing.assert_close(y[0],-y[1],atol=1e-6,rtol=1e-5)
        y[0].square().sum().backward()
        for group in (model.spatial, model.temporal, model.quality, model.rate):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in group.parameters()
                                   if p.grad is not None),0)

    def test_scene_reset_chunk_context_and_long_gap(self):
        c,g,s = state_context(32,[0,3,15,16,31],[0,31],[16])
        self.assertEqual(c[16,17],1)  # Second scene starts a new I frame.
        self.assertEqual(c[16,15],0)
        self.assertEqual(c[31,15],0)  # First P frame in its own scene.
        self.assertGreater(float(g[4]),4/24)
        with self.assertRaisesRegex(ValueError,'protected'):
            state_context(32,[0,15,31],[0,31],[16])
        self.assertTrue(np.all(s[:16]==0) and np.all(s[16:]==1))

    def test_features_depend_on_spatial_location_and_time(self):
        model = SpatialTemporalSelector(width=32,heads=4,layers=1,dropout=0).eval()
        image = torch.zeros((2,3,48,80),dtype=torch.uint8)
        image[0,:,:16,:16]=255
        image[1,:,:16,-16:]=255
        self.assertFalse(torch.allclose(model.encode_frames(image,torch.arange(2))[0],
                                       model.encode_frames(image,torch.arange(2))[1]))
        a,_,_ = state_context(12,[0,11],[0,11],[],24)
        b,_,_ = state_context(12,[0,11],[0,11],[],12)
        self.assertFalse(np.array_equal(a,b))

    def test_inference_keeps_unlimited_gap_and_protected_cuts(self):
        model = SpatialTemporalSelector(width=32,heads=4,layers=1,dropout=0)
        for p in model.parameters():
            p.data.zero_()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt'
            torch.save(dict(architecture=train.ARCHITECTURE,model=model.state_dict(),
                            model_config=model.config,target_scale=torch.ones(5)),path)
            frames=np.zeros((18,32,48,3),np.uint8)
            result=select(frames,[0,17],[9],path,.01)
            self.assertEqual(result['indices'],[0,8,9,17])
            self.assertIsNone(result['max_gap'])
            self.assertFalse(result['receiver_quality_used'])


class TrainingTests(unittest.TestCase):
    def fixture(self, root):
        rng=np.random.default_rng(42)
        rows=[]
        for name,split in [('a','train'),('b','train'),('c','val'),('d','test')]:
            row=dict(id=name,parent=name,split=split,dataset='one')
            rows.append(row)
            folder=root/'clips'/name
            folder.mkdir(parents=True)
            records=[dict(before=[0,7],after=[0,i,7]) for i in range(1,7)]
            y=rng.normal(0,.01,(6,5)).astype(np.float32)
            if split=='test':
                (folder/'labels.npz').write_bytes(b'sealed test targets')
            else:
                np.savez(folder/'labels.npz',y=y)
            save(folder/'labels.json',dict(samples=6,records=records,input=row,
                                          sha256=sha(folder/'labels.npz')))
            save(folder/'prepared.json',dict(keys=[0,7],cuts=[],
                                            source=dict(video=dict(fps=24.))))
            np.savez(folder/'visual_inputs.npz',
                     frames=rng.integers(0,255,(8,32,48,3),dtype=np.uint8))
            save(folder/'visual_inputs.json',dict(sha256=sha(folder/'visual_inputs.npz'),
                identity=dict(labels=sha(folder/'labels.npz'),prepared=sha(folder/'prepared.json'))))
        p=dict(seed=12,signature='test',target_order=list('abcde'),
            model=dict(width=32,heads=4,layers=1,dropout=.1),
            train=dict(device='cpu',max_epochs=6,patience=3,pairs_per_clip=2,
                       accumulate=2,lr=.001,weight_decay=.0001,warmup_epochs=1))
        return rows,p

    def setUp(self):
        old.STOP=False

    def test_exact_resume_includes_cnn_optimizer_dropout_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            rows,p=self.fixture(a)
            self.fixture(b)
            train.train(a,p,rows,max_steps=4)
            train.train(b,p,rows,max_steps=1)
            train.train(b,p,rows,max_steps=4)
            x=torch.load(a/'training/last.pt',weights_only=False)
            y=torch.load(b/'training/last.pt',weights_only=False)
            for key in x['model']:
                self.assertTrue(torch.equal(x['model'][key],y['model'][key]),key)
            for key in ('epoch','position','steps','best','bad','data_hash','scheduler'):
                self.assertEqual(x[key],y[key])
            self.assertTrue(torch.equal(x['torch_rng'],y['torch_rng']))
            self.assertTrue(torch.equal(x['sampler_rng'],y['sampler_rng']))
            for k in x['optimizer']['state']:
                for f,v in x['optimizer']['state'][k].items():
                    torch.testing.assert_close(v,y['optimizer']['state'][k][f],rtol=0,atol=0)
            self.assertTrue((a/'training/best.pt').exists())

    def test_split_leakage_and_mutated_visual_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows,p=self.fixture(root)
            rows[2]['parent']=rows[0]['parent']
            with self.assertRaisesRegex(ValueError,'input|leakage'):
                train.Corpus(root,rows)
            rows[2]['parent']='c'
            (root/'clips/a/visual_inputs.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                train.Corpus(root,rows)


if __name__=='__main__':
    unittest.main()
