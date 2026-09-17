"""Export score-free visual evidence for an explicitly AI-authored review."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_campaign import read, save, digest
from semantic_transmission.metric_campaign_real import real_inputs


def sheet(frames, path, title):
    width, height, label = 384, 216, 22
    canvas = Image.new('RGB', (4 * width, 32 + ((len(frames) + 3) // 4) * (height + label)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill='black')
    for t, frame in enumerate(frames):
        x, y = (t % 4) * width, 32 + (t // 4) * (height + label)
        canvas.paste(Image.fromarray(frame).resize((width, height)), (x, y))
        draw.text((x + 4, y + height + 3), f'frame {t:02d} | {t / 8:.3f} s', fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def prepare(base, out):
    rows = read(base / '03_natural/real_pairs.json')['rows']
    # Hide steps and scores in the image itself. The reviewer already knows
    # aggregate v5 results: this is not a fully blinded, independent human study.
    order = np.random.default_rng(260916).permutation(len(rows))
    index = []
    for n, j in enumerate(order, 1):
        row = rows[int(j)]
        a, b, rx, status, alignment = real_inputs(row)
        code = f'R{n:02d}'
        files = {}
        for name, frames in (('source', a), ('reconstruction', b)):
            p = out / 'review' / f'{code}_{name}.png'
            sheet(frames, p, f'{code} {name}; 8 Hz, all sampled frames')
            files[str(p.resolve())] = sha256(p)
        p = out / 'review' / f'{code}_rx.png'
        sheet(rx, p, f'{code} received keyframe hold; decoded slots: {[t for t, s in enumerate(status) if s == "ok"]}')
        files[str(p.resolve())] = sha256(p)
        index.append({'review_id': code, 'case_id': row['case_id'],
                      'source_sha256': row['source_sha256'], 'reconstruction_sha256': row['reconstruction_sha256'],
                      'alignment_sha256': digest(alignment), 'sampled_frames': len(a),
                      'evidence_files': files})
    save(out / 'review/index.json', {'schema': 'metric-revision-review-v1', 'rows': index,
         'method': 'score-free raster evidence for AI visual review; aggregate v5 outcomes previously seen',
         'sampling_scope': 'all 32 evaluation samples at 8 Hz; events between samples not certified'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base.resolve(), args.output.resolve())
