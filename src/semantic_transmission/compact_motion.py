"""Small, versioned camera and regional-motion side information.

Sender-only optical tracking. The receiver decodes bytes, never source frames.
The 15 knots describe motion inside an interval; they are not video keyframes.
This is a coarse 2-D motion representation, not object identity/occlusion coding.
"""
import base64
import struct

FIELD = 'bsu_motion_v1'
KNOTS = 15
SCALE = 8192


def encode(values, confidence, frames):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if values.shape != (KNOTS, 12) or confidence.shape != (KNOTS,):
        raise ValueError('motion shape mismatch')
    if not np.isfinite(values).all() or not np.isfinite(confidence).all():
        raise ValueError('nonfinite motion')
    if not 2 <= frames <= 65535 or np.any(np.abs(values) > 32767 / SCALE):
        raise ValueError('motion representation out of range')
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('invalid tracking confidence')
    raw = bytearray(struct.pack('!BBH', 1, KNOTS, frames))
    for row, score in zip(values, confidence):
        raw.extend(struct.pack('!12hB', *np.rint(row * SCALE).astype(int), round(float(score) * 255)))
    return base64.b64encode(raw).decode('ascii')


def decode(text, expected_frames=None):
    import numpy as np
    raw = base64.b64decode(text, validate=True)
    if len(raw) != 4 + KNOTS * 25:
        raise ValueError('invalid motion payload size')
    version, count, frames = struct.unpack('!BBH', raw[:4])
    if version != 1 or count != KNOTS or frames < 2:
        raise ValueError('unsupported motion payload')
    if expected_frames is not None and frames != expected_frames:
        raise ValueError('motion interval differs from decoder interval')
    rows = np.asarray([struct.unpack('!12hB', raw[4+i*25:4+(i+1)*25]) for i in range(count)])
    return dict(frames=frames, values=rows[:, :12].astype('float32')/SCALE,
                confidence=rows[:, 12].astype('float32')/255)


def extract(frames):
    """Track adjacent frames, accumulate similarity and four regional residuals."""
    import cv2
    import numpy as np
    if len(frames) < 2:
        raise ValueError('need at least two source frames')
    cv2.setNumThreads(2)
    cv2.setRNGSeed(2025)
    width, height = 288, 160
    gray = [cv2.cvtColor(cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            for f in frames]
    pose, residual = np.eye(3), np.zeros((4, 2))
    values, confidence = [np.zeros(12)], [1.0]
    failed = 0
    for previous, current in zip(gray, gray[1:]):
        points = cv2.goodFeaturesToTrack(previous, maxCorners=400, qualityLevel=.015, minDistance=5)
        score = 0.0
        if points is not None and len(points) >= 12:
            following, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                winSize=(21, 21), maxLevel=3)
            backward, ok, _ = cv2.calcOpticalFlowPyrLK(current, previous, following, None,
                winSize=(21, 21), maxLevel=3)
            good = (status[:, 0] == 1) & (ok[:, 0] == 1) & (np.linalg.norm(backward-points, axis=2)[:, 0] < 1.5)
            x, y = points[good, 0], following[good, 0]
            matrix, inliers = (cv2.estimateAffinePartial2D(x, y, method=cv2.RANSAC,
                ransacReprojThreshold=2.0, maxIters=1000) if len(x) >= 12 else (None, None))
            if matrix is not None and np.isfinite(matrix).all():
                scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
                if .85 < scale < 1.18:
                    score = float(inliers.mean()) * min(1.0, len(x)/80)
                    step = np.eye(3); step[:2] = matrix
                    pose = step @ pose
                    delta = y - (x @ matrix[:, :2].T + matrix[:, 2])
                    for cell in range(4):
                        subset = ((x[:, 0] >= width/2) == (cell % 2 == 1)) & ((x[:, 1] >= height/2) == (cell // 2 == 1))
                        if subset.sum() >= 4:
                            residual[cell] += np.median(delta[subset], axis=0) / [width, height]
        if score == 0:
            failed += 1
        values.append(np.r_[pose[0, 0]-1, pose[1, 0], pose[0, 2]/width, pose[1, 2]/height, residual.flatten()])
        confidence.append(score)
    positions = np.linspace(0, len(frames)-1, KNOTS)
    values = np.asarray(values)
    knots = np.stack([np.interp(positions, np.arange(len(frames)), values[:, i]) for i in range(12)], axis=1)
    scores = np.interp(positions, np.arange(len(frames)), confidence)
    # Cumulative uncertainty: long stretches of failed tracking must not appear reliable.
    scores *= 1 - failed/(len(frames)-1)
    encoded = encode(knots, scores, len(frames))
    decoded = decode(encoded, len(frames))
    audit = dict(frames=len(frames), knots=KNOTS, raw_bytes=379, base64_bytes=len(encoded),
        failed_transitions=failed, transitions=len(frames)-1, mean_confidence=float(scores.mean()),
        quantization_max_error=float(np.max(np.abs(knots-decoded['values']))),
        camera='accumulated robust 2-D similarity', regions='2 by 2 cumulative median residual flow',
        limitations='coarse regional motion; no object identities, depth, visibility or scene-cut handling')
    return encoded, audit


def add_to_packet(packet, encoded, capacity_bytes=None):
    from .wire import pack, unpack
    header, payload = unpack(packet)
    decode(encoded, header['video']['frames'])
    if FIELD in header:
        raise ValueError('motion already attached')
    header[FIELD] = encoded
    result = pack(header, payload)
    if capacity_bytes is not None and len(result) > capacity_bytes:
        raise ValueError(f'motion packet exceeds budget: {len(result)} > {capacity_bytes}')
    return result
