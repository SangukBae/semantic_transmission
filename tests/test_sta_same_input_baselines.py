import runpy

import numpy as np

from semantic_transmission.sta_video_validation import simulate


def test_same_input_distances_reveal_stage_hidden_from_final_rgb():
    distance = runpy.run_path('scripts/evaluate_sta_same_input_baselines.py')['stage_distances']
    a = np.stack([np.full((192, 320, 3), 10 + t, np.uint8) for t in range(32)])
    stage = {'tx_omission': 'source_tx', 'packet_loss': 'tx_rx', 'generation_freeze': 'rx_reconstruction'}
    values = []
    for kind, expected in stage.items():
        sim = simulate(a, kind, .25)
        values.append(sim['reconstruction'])
        scores = distance(a, sim['tx_frames'], sim['rx_frames'], sim['reconstruction'])
        assert {k for k, v in scores.items() if v > 0} == {expected + '_mse', expected + '_mae'}
    assert all(np.array_equal(v, values[0]) for v in values)
