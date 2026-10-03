"""Repair overlapping endpoint conditions without changing non-colliding masks.

The released 17-frame/5-latent alignment is retained wherever conditions do not
overlap. A colliding end key is moved to its requested final latent, not rounded.
Single-latent segments cannot hold two different keys and fail before mutation.
This repairs routing, not the VAE's temporal representation or generated quality.
"""
import ast
from types import SimpleNamespace

from .temporal import segment_lengths

POLICY = "repair_colliding_end_key_only_v1"
TRACE = "receiver/condition_collision_trace.json"


def upstream_functions(repo):
    """Read exact pure mask functions without importing models or loading weights."""
    path = repo / ".local/vendor/Open-Sora/opensora/utils/inference_utils.py"
    tree = ast.parse(path.read_text())
    names = {"parse_mask_strategy", "find_nearest_point", "apply_mask_strategy"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    defaults = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "MASK_DEFAULT" for t in n.targets))
    if len(functions) != len(names):
        raise ValueError("unsupported upstream masking functions")
    ns = {"MASK_DEFAULT": ast.literal_eval(defaults)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), ns)
    return SimpleNamespace(**{name: ns[name] for name in names})


def plan(strategy, reference_lengths, latent_frames, loop, key_count, align, functions):
    if latent_frames < 1 or key_count < 2 or not 0 <= loop < key_count - 1:
        raise ValueError("invalid segment dimensions")
    if len(reference_lengths) != key_count + loop:
        raise ValueError("unexpected key/generated reference count")
    if any(n != 1 for n in reference_lengths[:key_count]):
        raise ValueError("received keys must each have one latent")
    ops = []
    for segment, ref, rs, ts, length, edit in functions.parse_mask_strategy(strategy):
        if segment != loop:
            continue
        if not 0 <= ref < len(reference_lengths) or length < 1 or edit != 0:
            raise ValueError("unsupported reference/length/edit ratio")
        n = reference_lengths[ref]
        rs = rs if rs >= 0 else n + rs
        ts = ts if ts >= 0 else latent_frames + ts
        if not (0 <= rs < n and 0 <= ts < latent_frames):
            raise ValueError("reference/target position outside tensor")
        source = functions.find_nearest_point(rs, align, n) if align else rs
        target = functions.find_nearest_point(ts, align, latent_frames) if align else ts
        count = min(length, latent_frames - target, n - source)
        if count != length or source < 0 or target < 0:
            raise ValueError("reference length would be truncated")
        ops.append(dict(reference=ref, source=source, requested_target=ts,
                        aligned_target=target, target=target, length=count))
    expected = {0, 1} if loop == 0 else {loop + 1, key_count + loop - 1}
    if len(ops) != 2 or {o["reference"] for o in ops} != expected:
        raise ValueError("unexpected endpoint/history mask strategy")
    end = next(o for o in ops if o["reference"] == loop + 1)
    if end["length"] != 1 or end["source"] != 0 or end["requested_target"] != latent_frames - 1:
        raise ValueError("end condition must reference the final requested latent")
    other = next(o for o in ops if o is not end)
    expected_length = 1 if loop == 0 else 5
    if other["target"] != 0 or other["length"] != expected_length:
        raise ValueError("unexpected starting condition")
    occupied = set(range(other["target"], other["target"] + other["length"]))
    changed = end["target"] in occupied
    if changed:
        if end["requested_target"] in occupied:
            raise ValueError("insufficient separate latent positions for both endpoints; use a longer segment")
        end["target"] = end["requested_target"]
    effective = ";".join(f"{loop},{o['reference']},{o['source']},{o['target']},{o['length']},0" for o in ops)
    return dict(loop=loop, latent_frames=latent_frames, repaired=changed,
                operations=ops, effective_strategy=effective)


def apply_guard(z, refs, strategies, loop, align, key_count, functions):
    """Validate every batch before changing tensors; preserve upstream assignment."""
    import torch
    if len(refs) != len(strategies) or len(refs) != z.shape[0]:
        raise ValueError("conditioning batch sizes differ")
    records = [plan(s, [r.shape[1] for r in batch], z.shape[2], loop, key_count, align, functions)
               for s, batch in zip(strategies, refs)]
    masks = functions.apply_mask_strategy(z, refs, [r["effective_strategy"] for r in records], loop, align=None)
    for batch, record in enumerate(records):
        for op in record["operations"]:
            a, b, n = op["target"], op["source"], op["length"]
            if not torch.equal(z[batch, :, a:a+n], refs[batch][op["reference"]][:, b:b+n]):
                raise ValueError("condition did not survive mask assignment")
            if torch.any(masks[batch, a:a+n] != 0):
                raise ValueError("condition was not fixed by the mask")
    return masks, records


def preview(keys, functions):
    lengths = segment_lengths(keys)
    strategy = "0;" + ";".join(f"{i},{i+1},0,-1,1" for i in range(len(keys)-1))
    result = []
    for loop, frames in enumerate(lengths):
        latent = frames // 17 * 5 + (frames % 17 + 3) // 4
        if loop:
            strategy += f";{loop},{len(keys)+loop-1},-5,0,5,0"
        row = plan(strategy, [1]*len(keys) + [5]*loop, latent, loop, len(keys), 5, functions)
        row.update(start_frame=keys[loop], end_frame=keys[loop+1])
        result.append(row)
    return result


def validate_trace(records, expected):
    if records != [[{k: v for k, v in row.items() if k not in {"start_frame", "end_frame"}}]
                   for row in expected]:
        raise ValueError("runtime condition placement differs from the checked plan")
