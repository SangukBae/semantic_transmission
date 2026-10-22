import json
from pathlib import Path
import tempfile
import unittest

from semantic_transmission.dcvc_uf_codec import qp_schedule
from semantic_transmission import sparc_uf_rate_transport as t
from semantic_transmission.sparc_uf_rate import adaptive_schedules
from semantic_transmission.sparc_uf_rate_watch import rows
from semantic_transmission.webvid_ablation import Stages
from semantic_transmission.wire import pack


class ScheduleTests(unittest.TestCase):
    def test_schedule_size_bounds_and_boolean_rejection(self):
        self.assertEqual(qp_schedule(10,31),[31,31,31])
        self.assertEqual(qp_schedule(9,[32,28]),[32,28])
        for bad in ([32], [32,64], [32,True], [32,-1], 64, True):
            with self.assertRaises(ValueError):
                qp_schedule(9,bad)

    def test_adaptive_preserves_intra_and_rewards_high_error(self):
        plan=dict(indices=list(range(25)),cuts=[])
        calibration=dict(per_frame=dict(psnr_db=[40]*9+[30]*8+[20]*8,
                                         lpips_vgg=[.1]*9+[.3]*8+[.5]*8))
        records=[dict(position=p,valid_frames=n,entropy_bytes=100,qp=31)
                 for p,n in [(0,1),(1,8),(9,8),(17,8)]]
        value=adaptive_schedules(plan,calibration,dict(groups=[dict(records=records)]))
        self.assertEqual(len(value['proposals']),6)
        for row in value['proposals']:
            self.assertEqual(row['qps'][0],row['center'])
            self.assertLess(row['qps'][1],row['qps'][-1])
            self.assertTrue(all(abs(q-row['center'])<=row['strength'] for q in row['qps']))

    def test_equal_errors_do_not_invent_difficulty(self):
        plan=dict(indices=list(range(17)),cuts=[])
        quality=dict(per_frame=dict(psnr_db=[30]*17,lpips_vgg=[.2]*17))
        records=[dict(position=p,valid_frames=n,entropy_bytes=100,qp=31)
                 for p,n in [(0,1),(1,8),(9,8)]]
        proposals=adaptive_schedules(plan,quality,dict(groups=[dict(records=records)]))['proposals']
        self.assertTrue(all(p['changed_chunks']==0 for p in proposals))


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.video=dict(frames=18,width=64,height=64,fps=24.)
        self.keys=[0,9,17];self.weights={'image':'a','hts':'b'}
        self.coded=[(list(range(9)),[32,30],b'ab'*100),
                    (list(range(9,18)),[31,34],b'cd'*100)]

    def test_roundtrip_scene_resets_and_paid_schedule(self):
        packet=t.make(self.video,self.keys,[9],self.coded,self.weights)
        header,payload=t.parse(packet,self.weights)
        self.assertEqual(payload,b'ab'*100+b'cd'*100)
        self.assertEqual([c['qps'] for c in header['chunks']],[[32,30],[31,34]])
        self.assertEqual([d['intra'] for d in t.descriptors(list(range(18)),[9])],[True,False,True,False])

    def test_rejects_missing_qp_and_wrong_scene_boundary(self):
        packet=t.make(self.video,self.keys,[9],self.coded,self.weights)
        for change in ('qp','cut','offset','padding'):
            h,p=t.parse(packet,self.weights)
            if change=='qp':h['chunks'][0]['qps'].pop()
            elif change=='cut':h['cuts']=[]
            elif change=='offset':h['chunks'][0]['offset']=1
            else:h['chunks'][0]['padded_frames']=1
            with self.assertRaises(ValueError):t.parse(pack(h,p),self.weights)
        with self.assertRaises(ValueError):t.parse(packet[:-1],self.weights)


class ResumeMonitorTests(unittest.TestCase):
    def test_reuse_requires_unchanged_completed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def make(log):
                (root/'result.txt').write_text('completed')
            Stages(root,'identity').step('probe',['result.txt'],['result.txt'],make)
            Stages(root,'identity').step('probe',['result.txt'],['result.txt'],
                                       lambda _:self.fail('completed stage reran'))
            (root/'result.txt').write_text('changed')
            with self.assertRaises(ValueError):
                Stages(root,'identity').step('probe',['result.txt'],['result.txt'],make)

    def test_failure_snapshot_is_read_only_and_has_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'status.json'
            path.write_text(json.dumps(dict(status='FAILED',completed_trials=2,total_trials=36,
                                           error='exit 1',exit_code=1)))
            before=path.read_bytes()
            output,status=rows(root,['GPU load','GPU memory'])
            self.assertEqual(status,'FAILED')
            self.assertTrue(any('exit 1' in line for line in output))
            self.assertTrue(any('재개 명령' in line for line in output))
            self.assertEqual(path.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
