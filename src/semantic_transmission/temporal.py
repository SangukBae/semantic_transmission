"""Frame accounting for adjacent, endpoint-conditioned segments."""


def segment_lengths(indices, overlap_frames=17):
    if len(indices) < 2 or indices[0] != 0 or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("keyframes must start at zero and increase strictly")
    return [b - a + 1 + (overlap_frames if i else 0)
            for i, (a, b) in enumerate(zip(indices, indices[1:]))]


def trim_prefix(segment_index, overlap_frames=17, policy="endpoint_exact"):
    resolve_concatenation_policy(policy)
    if policy == "official_release":
        return overlap_frames if segment_index else 0
    # Remove conditioning overlap AND the shared adjacent-segment endpoint.
    return overlap_frames + 1 if segment_index else 0


def output_source_indices(indices, policy="endpoint_exact"):
    """Map the published concatenation's repeated boundary frames explicitly."""
    segment_lengths(indices)  # validate endpoints and ordering
    if policy == "official_release":
        return [frame for a, b in zip(indices, indices[1:]) for frame in range(a, b + 1)]
    if policy != "endpoint_exact":
        raise ValueError(f"unknown temporal policy: {policy}")
    return list(range(indices[-1] + 1))


def resolve_concatenation_policy(decoder_policy="endpoint_exact", concatenation_policy=None):
    policy = decoder_policy if concatenation_policy is None else concatenation_policy
    if policy not in {"official_release", "endpoint_exact"}:
        raise ValueError(f"unknown concatenation policy: {policy}")
    return policy


def concatenate_segments(segments, overlap_frames=17, policy="endpoint_exact"):
    """Only final C,T,H,W concatenation changes; generation references stay intact."""
    import torch
    resolve_concatenation_policy(policy)
    return torch.cat([clip[:, trim_prefix(i, overlap_frames, policy):]
                      for i, clip in enumerate(segments)], dim=1)


def unique_output_positions(indices, policy="official_release"):
    """Keep the previous segment's shared endpoint, matching endpoint_exact trim."""
    mapping = output_source_indices(indices, policy)
    return [i for i, frame in enumerate(mapping) if i == 0 or frame != mapping[i - 1]]


def conditioning_indices(frame_count, overlap_frames=17):
    """A fixed overlap, left-padded with the first frame for short segments."""
    if frame_count < 1 or overlap_frames < 1:
        raise ValueError("conditioning needs positive frame counts")
    indices = list(range(max(0, frame_count - overlap_frames), frame_count))
    return [indices[0]] * (overlap_frames - len(indices)) + indices
