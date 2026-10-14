"""Variable-count Qwen selection against the latest accepted source frame.

There is no target key count, maximum key gap, or pixel-novelty quota. Candidate
sampling frequency is not a keyframe schedule. This is direct Qwen judgment,
not a reproduction of InternVL's PSSS probability calibration.
"""
import time

POLICY = "qwen35_sequential_semantic_v1"
PROMPT = '''Compare the latest ACCEPTED REFERENCE source image with the later
CANDIDATE source image for semantic video reconstruction. Decide whether the
candidate establishes a meaningful visible change relative to that reference.
Meaningful changes include object entry/exit, occlusion, changed appearance,
an action/pose or object-position change, a scene cut, or camera movement that
reveals substantially different scene content. Smooth camera motion alone over
the same objects/layout, tiny jitter, compression, lighting noise, or repeating
the same visible state is NOT a new semantic state. Accumulated changes must be
judged against the ACCEPTED REFERENCE, not against the previous candidate.
Use only visible evidence. Do not infer purpose, identity, hidden objects or
unseen motion. Do not select to fill a time interval or a target number of keys.
Return ONLY compact JSON:
{"decision":"changed|same|uncertain","kind":"entry_exit|occlusion|pose|position|appearance|scene_change|viewpoint|none|unknown",
"reference_state":"short visible reference state",
"candidate_state":"short visible candidate state","evidence_frames":[REF,CAND]}.
Use kind=none for same, unknown for uncertain. changed requires a specific
visible difference; merely saying the view/position/scene is maintained is same.
If evidence is insufficient, choose uncertain. Both state descriptions together
must use at most 45 words. Cite the actual reference and candidate source IDs.
Intermediate images, if supplied for review, help resolve those two endpoints;
they do not replace the accepted reference or the candidate being judged.'''
KINDS = {"entry_exit", "occlusion", "pose", "position", "appearance", "scene_change", "viewpoint"}
# Exact subtypes of entry_exit, not guesses about the visual evidence. Keep the
# original spelling in the sealed model response and selection trace.
KIND_ALIASES = {"entry": "entry_exit", "exit": "entry_exit", "entry/exit": "entry_exit"}


def candidates(frames, stride=4):
    if type(frames) is not int or frames < 2 or type(stride) is not int or stride < 1:
        raise ValueError("expected at least two frames and a positive candidate stride")
    return sorted({*range(stride, frames, stride), frames - 1})


def validate_decision(value, reference, candidate):
    if not isinstance(value, dict) or value.get("decision") not in ("changed", "same", "uncertain"):
        raise ValueError("decision must be changed, same, or uncertain")
    decision, kind = value["decision"], value.get("kind")
    allowed = sorted(KINDS) if decision == "changed" else ["none" if decision == "same" else "unknown"]
    canonical = KIND_ALIASES.get(kind, kind) if isinstance(kind, str) else kind
    if canonical not in allowed:
        raise ValueError(f"decision={decision!r} and kind={kind!r} contradict the schema; "
                         f"kind must be one of {allowed}. Keep the decision grounded in the images.")
    states = [value.get(k) for k in ("reference_state", "candidate_state")]
    if any(not isinstance(s, str) or not s.strip() for s in states) or len(" ".join(states).split()) > 45:
        raise ValueError("provide two short visible state descriptions, at most 45 words total")
    evidence = value.get("evidence_frames")
    if (not isinstance(evidence, list) or any(type(i) is not int for i in evidence)
            or evidence != [reference, candidate]):
        raise ValueError(f"evidence_frames must be the actual endpoints {[reference, candidate]}")
    if decision == "changed" and states[0].strip().casefold() == states[1].strip().casefold():
        raise ValueError("identical stated states cannot justify changed")
    return value


def validate_keys(keys, frames):
    if (not isinstance(keys, list) or len(keys) < 2 or any(type(k) is not int for k in keys)
            or keys != sorted(set(keys)) or keys[0] != 0 or keys[-1] != frames - 1):
        raise ValueError("semantic keys must be ordered, unique, in range, and retain both endpoints")


def select(frames, session, *, stride=4, prompt=PROMPT, tokens=256, progress=None):
    keys, trace = [0], []
    schedule = candidates(frames, stride)
    started = time.monotonic()
    for n, candidate in enumerate(schedule):
        reference = keys[-1]
        request = prompt.replace("REF,CAND", f"{reference},{candidate}")
        request += f"\nACCEPTED REFERENCE={reference}; CANDIDATE={candidate}."
        validator = lambda value: validate_decision(value, reference, candidate)
        value = session.call("semantic_select", [reference, candidate], request, validator, tokens)
        initial, reviewed = value, False
        if value["decision"] == "uncertain":
            ids = sorted({reference, candidate, *[round(reference + (candidate-reference)*i/4) for i in (1, 2, 3)]})
            value = session.call("semantic_select_review", ids, request, validator, tokens)
            reviewed = True
        row = dict(reference=reference, candidate=candidate, initial=initial,
                   result=value, reviewed=reviewed, selected=value["decision"] == "changed")
        trace.append(row)
        if value["decision"] == "uncertain":
            raise RuntimeError(f"semantic decision remains uncertain at {reference}->{candidate}; "
                               "calls preserved; selection incomplete, no forced key or silent skip")
        if row["selected"]:
            keys.append(candidate)
        if progress:
            progress(n + 1, len(schedule), started)
    if keys[-1] != frames - 1:
        keys.append(frames - 1)
    validate_keys(keys, frames)
    return keys, trace
