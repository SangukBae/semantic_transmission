"""Repair RFLOW's zero temporal ratio for 2..16 input frames only.

Retain the released schedule for one-frame and >=17-frame inputs. This changes
inference scheduling, not STDiT weights, reference masks, or random draws.
"""
from .artifacts import write_json
from .temporal import segment_lengths

POLICY = "short_segment_actual_latent_schedule_v1"
TRACE = "receiver/short_schedule_trace.json"


def transform(t, model_kwargs, original, base_resolution=512 * 512,
              base_num_frames=1, scale=1.0, num_timesteps=1):
    import torch
    frames = model_kwargs["num_frames"]
    if (frames.ndim != 1 or not frames.numel() or not torch.isfinite(frames).all()
            or (frames < 1).any() or (frames != frames.floor()).any()
            or (frames != frames[0]).any()):
        raise ValueError("schedule requires uniform positive integer frame counts")
    if not 1 < frames[0].item() < 17:
        return original(t, model_kwargs, base_resolution, base_num_frames, scale, num_timesteps)
    # Match upstream arithmetic/precision; the sole change is temporal length.
    values = {k: v.float() if v.dtype == torch.float16 else v for k, v in model_kwargs.items()
              if k in {"height", "width", "num_frames"}}
    time_ratio = (((values["num_frames"] + 3) // 4) / base_num_frames).sqrt()
    space_ratio = ((values["height"] * values["width"]) / base_resolution).sqrt()
    ratio = space_ratio * time_ratio * scale
    unit = t / num_timesteps
    return ratio * unit / (1 + (ratio - 1) * unit) * num_timesteps


def validate_times(times):
    import torch
    values = torch.stack(times)
    if (not torch.isfinite(values).all() or (values <= 0).any()
            or (values[:-1] <= values[1:]).any()):
        raise ValueError("invalid sampling schedule: nonfinite, zero, or nondecreasing timesteps")


def validate_trace(records, keys, steps):
    lengths = segment_lengths(keys)
    if len(records) != len(lengths):
        raise ValueError("incomplete short-schedule trace")
    for i, (row, frames) in enumerate(zip(records, lengths)):
        if (row["loop"] != i or row["frames"] != frames or row["policy"] != POLICY
                or row["repaired"] != (1 < frames < 17)
                or len(row["times"]) != steps or not row["conditions_preserved"]
                or not row["finite_output"] or not row["generated_update_nonzero"]):
            raise ValueError("schedule trace differs from expected segment plan")
        import math
        times = row["times"]
        if (not all(math.isfinite(t) and t > 0 for t in times)
                or any(a <= b for a, b in zip(times, times[1:]))):
            raise ValueError("invalid recorded sampling schedule")


def install(keys, path):
    """Install before the frozen scoped-noise decoder; audit every sampled segment."""
    import torch
    from opensora.schedulers import rf
    original_transform, original_sample = rf.timestep_transform, rf.RFLOW.sample
    lengths, records = segment_lengths(keys), []

    def fixed(t, model_kwargs, **kwargs):
        return transform(t, model_kwargs, original_transform, **kwargs)

    def sample(scheduler, model, text_encoder, z, *args, **kwargs):
        loop = len(records)
        frames = lengths[loop]
        if (z.shape[0] != 1 or not scheduler.use_timestep_transform
                or scheduler.num_sampling_steps < 2):
            raise ValueError("unsupported short-schedule decoder configuration")
        extra, mask = kwargs["additional_args"], kwargs["mask"]
        if extra["num_frames"].tolist() != [frames]:
            raise ValueError("sampling frame count differs from adjacent keyframes")
        expected_latents = frames // 17 * 5 + (frames % 17 + 3) // 4
        if z.shape[2] != expected_latents:
            raise ValueError("VAE temporal shape differs from the checked schedule")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("only fixed conditions and generated slots are supported")
        times = [(1 - i / scheduler.num_sampling_steps) * scheduler.num_timesteps
                 for i in range(scheduler.num_sampling_steps)]
        if scheduler.use_discrete_timesteps:
            times = [int(round(t)) for t in times]
        times = [fixed(torch.tensor([t], device=z.device), extra,
                       num_timesteps=scheduler.num_timesteps) for t in times]
        validate_times(times)  # Fail before model execution if scheduling is invalid.
        before = z.detach().clone()
        actual_times = []

        def check_step(module, inputs, keywords):
            index = len(actual_times)
            if index >= len(times) or not torch.equal(inputs[1], times[index].repeat(2)):
                raise ValueError("actual model timestep differs from checked schedule")
            actual_times.append(float(inputs[1][0]))

        handle = model.register_forward_pre_hook(check_step, with_kwargs=True)
        try:
            result = original_sample(scheduler, model, text_encoder, z, *args, **kwargs)
        finally:
            handle.remove()
        fixed_slots = mask[0] == 0
        preserved = torch.equal(result[:, :, fixed_slots].float(), before[:, :, fixed_slots].float())
        delta = (result.float() - before.float()).square().mean((0, 1, 3, 4)).sqrt()
        changed = not (mask == 1).any() or bool((delta[mask[0] == 1] > 0).any())
        finite = bool(torch.isfinite(result).all())
        if not preserved or not changed or not finite or len(actual_times) != len(times):
            raise ValueError("sampling violated fixed conditions, update, finiteness, or step count")
        records.append(dict(policy=POLICY, loop=loop, frames=frames, latent_frames=expected_latents,
            repaired=1 < frames < 17, times=actual_times, conditions_preserved=preserved,
            generated_update_nonzero=bool(changed), finite_output=finite,
            slot_rms_change=delta.cpu().tolist()))
        write_json(path, records)
        return result

    rf.timestep_transform = fixed
    rf.RFLOW.sample = sample
    return records
