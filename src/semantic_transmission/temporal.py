"""Frame accounting for adjacent, endpoint-conditioned segments."""


def segment_lengths(indices, overlap_frames=17):
    if len(indices) < 2 or indices[0] != 0 or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("keyframes must start at zero and increase strictly")
    return [b - a + 1 + (overlap_frames if i else 0)
            for i, (a, b) in enumerate(zip(indices, indices[1:]))]


def trim_prefix(segment_index, overlap_frames=17):
    # Remove conditioning overlap AND the shared adjacent-segment endpoint.
    return overlap_frames + 1 if segment_index else 0


def conditioning_indices(frame_count, overlap_frames=17):
    """A fixed overlap, left-padded with the first frame for short segments."""
    if frame_count < 1 or overlap_frames < 1:
        raise ValueError("conditioning needs positive frame counts")
    indices = list(range(max(0, frame_count - overlap_frames), frame_count))
    return [indices[0]] * (overlap_frames - len(indices)) + indices
