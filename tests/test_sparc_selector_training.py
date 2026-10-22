import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from semantic_transmission import sparc_selector_data as data
from semantic_transmission import sparc_selector_train as training


class SelectionTests(unittest.TestCase):
    def test_parent_split_does_not_depend_on_crop(self):
        parent = 'lavib:177'
        self.assertEqual({data.split(parent) for _ in range(30)}, {data.split(parent)})
        self.assertGreater(len({data.split('lavib:'+str(i)) for i in range(100)}), 2)

    def test_state_changes_features_and_protects_boundaries(self):
        rng = np.random.default_rng(3)
        frames = rng.integers(0,35,(24,32,64,3),dtype=np.uint8)
        x,f,_ = data.source_features(frames)
        keys,cuts = [0,12,23],[6]
        pairs = data.candidates(x,f,keys,cuts,7)
        self.assertTrue(pairs)
        self.assertTrue(any(max(np.diff(before)) > 4 for before,_ in pairs))
        for before,after in pairs:
            self.assertTrue({0,5,6,12,23}.issubset(before))
            self.assertTrue({0,5,6,12,23}.issubset(after))
            self.assertNotEqual(before,after)
            self.assertTrue(np.isfinite(data.action_features(x,f,before,after,cuts,keys)).all())
        a = data.state_features(x,f,[0,23],7)
        b = data.state_features(x,f,[0,4,23],7)
        self.assertFalse(np.array_equal(a,b))

    def test_constant_scene_can_keep_gap_above_four(self):
        import torch
        frames = np.zeros((16,32,64,3),np.uint8)
        x,f,cuts = data.source_features(frames)
        d = len(data.action_features(x,f,[0,15],[0,8,15],cuts,[0,15]))
        model = training.network(d)
        for p in model.parameters():
            p.data.zero_()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            torch.save(dict(input_dim=d,model=model.state_dict(),normalization=dict(
                x_mean=torch.zeros(d),x_std=torch.ones(d),
                y_mean=torch.zeros(5),y_std=torch.ones(5))),path)
            result=training.select(frames,[0,15],[],path,.01)
        self.assertEqual(result['indices'],[0,15])
        self.assertIsNone(result['max_gap'])

    def test_resume_response_must_start_at_exact_offset(self):
        stream = data.RemoteParts('commit',[dict(path='lavib00',size=4)],lambda *args:None)
        stream.offset=2
        class Response:
            status_code=200
            headers={}
            def raise_for_status(self): pass
            def close(self): pass
        with patch('requests.get',return_value=Response()),patch.object(data.time,'sleep'):
            with self.assertRaisesRegex(ValueError,'resume offset'):
                stream.read(2)


class TrainingTests(unittest.TestCase):
    def fixture(self,root):
        rows=[]
        rng=np.random.default_rng(21)
        for name,split in [('a','train'),('b','train'),('c','val'),('d','test')]:
            row=dict(id=name,parent=name,split=split)
            rows.append(row)
            folder=root/'clips'/name
            folder.mkdir(parents=True)
            x=rng.normal(size=(15,11)).astype(np.float32)
            y=rng.normal(size=(15,5)).astype(np.float32)
            if split=='test':
                # Deliberately not an NPZ: training must never load test targets.
                (folder/'labels.npz').write_bytes(b'test values must remain sealed')
            else:
                np.savez(folder/'labels.npz',x=x,y=y)
            data.save(folder/'labels.json',dict(samples=15,sha256=data.sha(folder/'labels.npz')))
        p=dict(seed=22,signature='test_protocol',target_order=['a','b','c','d','e'],
               train=dict(lr=.001,weight_decay=.0001,max_epochs=10,patience=3,batch_size=8))
        return rows,p

    def test_exact_mid_epoch_resume_and_test_isolation(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'full',Path(tmp)/'resumed'
            rows,p=self.fixture(a)
            self.fixture(b)
            training.train(a,p,rows,max_steps=7)
            training.train(b,p,rows,max_steps=2)
            training.train(b,p,rows,max_steps=7)
            x=torch.load(a/'training/last.pt',weights_only=False)
            y=torch.load(b/'training/last.pt',weights_only=False)
            for key in x['model']:
                self.assertTrue(torch.equal(x['model'][key],y['model'][key]),key)
            for key in ('steps','position','epoch','best','bad','data_hash'):
                self.assertEqual(x[key],y[key])
            self.assertTrue(torch.equal(x['order'],y['order']))
            self.assertTrue(torch.equal(x['sampler_rng'],y['sampler_rng']))
            self.assertEqual(x['scheduler'],y['scheduler'])
            for key in x['optimizer']['state']:
                for field,value in x['optimizer']['state'][key].items():
                    actual=y['optimizer']['state'][key][field]
                    if isinstance(value,torch.Tensor):
                        self.assertTrue(torch.equal(value,actual))
                    else:
                        self.assertEqual(value,actual)

    def test_training_rejects_cross_split_parent_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows,p=self.fixture(root)
            rows[2]['parent']=rows[0]['parent']
            with self.assertRaisesRegex(ValueError,'leakage'):
                training.train(root,p,rows,max_steps=1)

    def test_resume_rejects_changed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows,p=self.fixture(root)
            training.train(root,p,rows,max_steps=1)
            (root/'clips/a/labels.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                training.train(root,p,rows,max_steps=2)


if __name__=='__main__':
    unittest.main()
