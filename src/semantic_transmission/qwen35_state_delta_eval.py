"""Frozen Astra-reference evaluation of the state/delta experiment.

Only this evaluation module reads reference annotations. Extraction never does.
Anonymous packets may be delivered incrementally; their bytes are immutable.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import html
import json
from pathlib import Path
import shutil

from . import auto_extraction as ae, qwen35_feedback as fb
from . import qwen35_feedback_eval as old_eval
from . import qwen35_state_delta as sd
from .artifacts import sha256, write_json

CODES = dict(fixed_state="M26", budget_state="M71", uniform_state="M48", budget_checked="M93")
LABELS = dict(fixed_state="상태·변화 / 기존 152키 고정", budget_state="상태·변화 / 예산 79키",
              uniform_state="상태·변화 / 균등 79키", budget_checked="예산 79키 / 제한적 대응 검사")


def role_costs(bundle):
    result={}
    for path in {c["path"] for c in bundle["calls"]}:
        record=ae.unseal(path)
        role=record["identity"]["role"].removesuffix("_schema")
        row=result.setdefault(role,dict(calls=0,generation_seconds=0.,input_tokens=0,output_tokens=0))
        row["calls"]+=1;row["generation_seconds"]+=record["seconds"]
        row["input_tokens"]+=record["input_tokens"];row["output_tokens"]+=record["generated_tokens"]
    return result


def freeze_write(path, value):
    if path.exists():
        if ae.read(path) != value:
            raise ValueError(f"frozen artifact changed: {path}")
    else:
        write_json(path, value)


def protocol(root):
    plan = sd.check_plan(root)
    old_root = Path(plan["old_forward"]).parent
    path = old_root / "astra_reference/reference.json"
    freeze = ae.read(old_root / "astra_reference/freeze.json")
    if sha256(path) != freeze["reference_sha256"]:
        raise ValueError("frozen source reference changed")
    prior = ae.read(old_root / "evaluation/protocol.json")
    result = dict(prior, reference_path=str(path), plan_signature=plan["signature"],
        historical_caveat="Historical comparators retained from prior evaluation; no new judge scores fabricated",
        causal_caveat="Fixed keys/samples isolate a changed caption workflow, not equal-compute or prompt-only effects; selection arms share budget and caption rules",
        test_scope="Previously analyzed single development video; no held-out or human-ground-truth claim",
        temporal_domain=[0, plan["frames"]-1],
        source_frames=plan["source_frames"],
        instructions_extra="Judge every candidate using source images, regardless of plausible wording or agreement with other model outputs. Do not inspect private_method_map, extraction code, or methods not delivered. Count only clear source contradictions; retain ambiguity separately. Caption anchor describes its initial state, and subsequent sentences can describe changes within the same interval. Require temporal overlap for recalled reference facts.")
    freeze_write(root / "evaluation/protocol.json", result)
    freeze_write(root / "evaluation/delivery_clarification.json",dict(
        protocol_sha256=sha256(root / "evaluation/protocol.json"),
        actual_delivery="Anonymous method codes, one method per packet, chronological 10-second batches delivered as inference completes",
        simultaneous_randomized_comparison=False,
        correction="The inherited protocol caption_judgment phrase 'order shuffled' describes the earlier experiment template. This experiment's method delivery is sequential, not randomized. Preserve the originally delivered protocol bytes and report this clarification explicitly."))
    freeze_write(root / "evaluation/private_method_map.json", {v:k for k,v in CODES.items()})
    freeze_write(root / "evaluation/agent_provenance.json", dict(
        configured_model="gpt-6-astra", invocation="existing explicitly configured GPT6-Astra agents from the user-requested evaluation",
        agents=["/root/astra_reference", "/root/astra_judge_late"],
        reference_sha256=freeze["reference_sha256"], human_ground_truth=False,
        original_provenance_path=str(old_root / "evaluation/agent_provenance.json"),
        original_provenance_sha256=sha256(old_root / "evaluation/agent_provenance.json")))
    return plan, result, ae.read(path)


def available_captions(root, arm):
    full = root / f"{arm}.json"
    if full.exists():
        return ae.unseal(full)["captions"]
    return [ae.read(p) for p in sorted((root / "partial" / arm).glob("*.json"))]


def prepare(root):
    plan, instructions, reference = protocol(root)
    ready = []
    for arm in sd.ARMS:
        captions = available_captions(root, arm)
        if not captions:
            continue
        if any(r["segment"] != i for i,r in enumerate(captions)):
            raise ValueError("noncontiguous partial extraction")
        for lo in range(0, 60, 10):
            required_end = min((lo+10)*24, plan["frames"]-1)
            if captions[-1]["end_exclusive"] < required_end:
                continue
            bins = []
            for ref in reference["bins"][lo:lo+10]:
                rows = [dict(caption_id=f"{CODES[arm]}_{r['segment']:03d}", start=r["start"],
                             end_exclusive=r["end_exclusive"], caption=r["caption"])
                        for r in captions if r["start"] < ref["end_exclusive"] and ref["start"] < r["end_exclusive"]]
                bins.append(dict(reference=ref, candidates=[dict(method=CODES[arm], captions=rows)]))
            packet = dict(evaluator_model="gpt-6-astra", reference_sha256=instructions["reference_sha256"],
                source_frames=plan["source_frames"], fps=plan["fps"], frames=plan["frames"],
                bins=bins, instructions=instructions)
            path = root / "evaluation/packets" / f"{CODES[arm]}_{lo:02d}_{lo+9:02d}.json"
            freeze_write(path, packet)
            ready.append(str(path))
    files = [root / "plan.json", root / "evaluation/protocol.json",
             Path(instructions["reference_path"]), root / "evaluation/agent_provenance.json",
             root / "evaluation/delivery_clarification.json",
             *sorted((root / "evaluation/packets").glob("*.json"))]
    manifest_path = root / "evaluation/input_manifest.json"
    manifest = ae.read(manifest_path) if manifest_path.exists() else {}
    for path in files:
        key, digest = str(path), sha256(path)
        if key in manifest and manifest[key] != digest:
            raise ValueError("evaluation input bytes changed")
        manifest[key] = digest
    write_json(manifest_path, manifest)
    return dict(status="INCREMENTAL_ANONYMOUS_PACKETS", packets=ready)


def reuse_identical(root):
    """Identical checked/base outputs inherit judgments, never fresh scores."""
    dest = root / "evaluation"
    truth = {}
    for path in (dest / "packets").glob("*.json"):
        for row in ae.read(path)["bins"]:
            for method in row["candidates"]:
                truth[(row["reference"]["id"], method["method"])] = (method,path)
    prior = {}
    for path in (dest / "judgments").glob("*.json"):
        if path.name == "exact_input_reuse.json":
            continue
        for row in ae.read(path)["judgments"]:
            if row["method"] == CODES["budget_state"]:
                prior[row["bin_id"]] = (row,path)
    reused, pending = [], []
    for bin_id in range(60):
        old = truth.get((bin_id,CODES["budget_state"]))
        new = truth.get((bin_id,CODES["budget_checked"]))
        if not old or not new:
            continue
        if old_eval.candidate_identity(old[0]) != old_eval.candidate_identity(new[0]):
            pending.append(bin_id)
            continue
        if bin_id not in prior:
            continue
        original,path = prior[bin_id]
        id_map = {a["caption_id"]:b["caption_id"] for a,b in zip(old[0]["captions"],new[0]["captions"])}
        reused.append(dict(original, method=CODES["budget_checked"],
            unsupported_claims=[dict(err,caption_id=id_map[err["caption_id"]]) for err in original["unsupported_claims"]],
            reuse_of=dict(path=str(path),sha256=sha256(path),method=CODES["budget_state"],
                          bin_id=bin_id,caption_id_mapping=id_map),
            evaluation_mode="exact caption and interval reuse of actual GPT6-Astra judgment"))
    if reused:
        write_json(dest / "judgments/exact_input_reuse.json", dict(evaluator_model="gpt-6-astra",
            reference_sha256=ae.read(dest / "protocol.json")["reference_sha256"],
            packet_sha256s=[sha256(p) for p in sorted((dest / "packets").glob("*.json"))],judgments=reused))
    write_json(dest / "checked_pending.json",dict(changed_bins=pending,reused_bins=[r["bin_id"] for r in reused]))
    return dict(changed_bins=pending,reused_count=len(reused))


def aggregate(root):
    from .qwen35_state_delta_resume import install_normalizer
    install_normalizer()
    extraction = sd.verify(root)
    prepare(root)
    reuse_identical(root)
    rows = old_eval.validate_judgments(root)
    plan, instructions, reference = protocol(root)
    check_audit = audit_correspondence(root, plan)
    if len(rows) != 240:
        raise ValueError("all four methods and 60 bins must be judged")
    metrics = {}
    for arm in sd.ARMS:
        bundle = ae.unseal(root / f"{arm}.json")
        part = [r for r in rows if r["method"] == CODES[arm]]
        errors = [r["bin_id"] for r in part if r["unsupported_claims"]]
        recalled = sum(len(r["covered_fact_ids"]) for r in part)
        total = sum(r["total_facts"] for r in part)
        metrics[arm] = dict(label=LABELS[arm],keyframes=len(bundle["keys"]),captions=len(bundle["captions"]),
            error_bins=errors,error_free_bins=60-len(errors),recalled_facts=recalled,reference_facts=total,
            essential_fact_recall=recalled/total, event_coverage=old_eval.event_coverage(bundle["keys"],reference),
            recommended_frame_matching=old_eval.tolerant_matching(bundle["keys"],reference["recommended_keyframes"]),
            costs=extraction["arms"][arm],costs_by_role=role_costs(bundle))
    old_root = Path(plan["old_forward"]).parent
    previous = ae.read(old_root / "evaluation/RESULT.json")
    old_bundle = ae.unseal(old_root / "forward.json")
    new_bundle = ae.unseal(root / "fixed_state.json")
    if old_bundle["keys"] != new_bundle["keys"]:
        raise ValueError("fixed-key comparison changed keys")
    old_calls = [ae.unseal(c["path"]) for c in old_bundle["calls"] if c["role"] == "caption"]
    new_calls = [ae.unseal(c["path"]) for c in new_bundle["calls"] if c["role"].removesuffix("_schema") == "state_caption"]
    for before, after in zip(old_bundle["captions"], new_bundle["captions"]):
        fields = ("start", "end_exclusive", "source_indices")
        if any(before[k] != after[k] for k in fields):
            raise ValueError("fixed-key caption source samples differ")
        old_call = next(r for r in old_calls if r["identity"]["ids"] == before["source_indices"])
        new_call = next(r for r in new_calls if r["identity"]["ids"] == after["source_indices"])
        if (old_call["identity"]["frame_hashes"] != new_call["identity"]["frame_hashes"] or
                old_call["image_grid_thw"] != new_call["image_grid_thw"]):
            raise ValueError("fixed-key images or visual tokens differ")
    comparisons = {}
    for name,a,b in (("fixed_caption",previous["metrics"]["forward"],metrics["fixed_state"]),
                     ("selection",metrics["uniform_state"],metrics["budget_state"]),
                     ("correspondence",metrics["budget_state"],metrics["budget_checked"])):
        old_errors,new_errors = set(a["error_bins"]),set(b["error_bins"])
        comparisons[name] = dict(resolved_error_bins=sorted(old_errors-new_errors),
            new_error_bins=sorted(new_errors-old_errors), recalled_fact_delta=b["recalled_facts"]-a["recalled_facts"])
    result = dict(status="COMPLETE_ASTRA_REFERENCE_DEVELOPMENT_EVALUATION",video_id=plan["video_id"],
        policy=plan["policy"],signature=plan["signature"],reference_sha256=instructions["reference_sha256"],
        evaluator_model="gpt-6-astra",metrics=metrics,comparisons=comparisons,
        historical_metrics=previous["metrics"],
        historical_result_path=str(old_root / "evaluation/RESULT.json"),
        historical_result_sha256=sha256(old_root / "evaluation/RESULT.json"),
        old_forward_costs_by_role=role_costs(old_bundle),
        fixed_input_comparison=dict(status="PASS_IDENTICAL_TARGET_FRAMES_AND_VISUAL_GRIDS",
            intervals=len(new_bundle["captions"]),text_prompts_equal=False,compute_equal=False),
        correspondence_validation=check_audit,
        judge_delivery=ae.read(root / "evaluation/delivery_clarification.json"),
        human_ground_truth=False,unseen_video_validation=False,reconstruction_evaluated=False,
        workflow_not_equal_compute=True,training_executed=False,extraction=extraction)
    unique = [ae.unseal(p) for p in (root / "calls").glob("*/*.json")]
    result["whole_experiment_cost"] = dict(successful_unique_calls=len(unique),
        successful_generation_seconds=sum(r["seconds"] for r in unique),
        failed_generation_seconds=extraction["failed_inference_seconds"],
        setup_seconds=ae.read(root / "runtime.json")["setup_seconds"],
        execution=ae.read(root / "latest_execution.json"),
        scope="All four arms share cached observations/identical calls; per-arm times count their standalone required successful calls")
    archives = sorted(root.glob("schema_attempt_*"))
    result["whole_experiment_cost"]["earlier_schema_attempts"] = [dict(
        path=str(p),execution=ae.read(p/"latest_execution.json"),
        failed_calls=len(list((p/"failed_attempts").glob("*/*.json"))),
        failed_generation_seconds=sum(ae.unseal(f)["seconds"] for f in (p/"failed_attempts").glob("*/*.json")))
        for p in archives if (p/"latest_execution.json").exists()]
    result["whole_experiment_cost"]["earlier_execution_stages"] = [ae.read(p)
        for p in sorted((root/'execution_stages').glob('*/latest_execution.json'))]
    result["syntax_recovery_validation"] = audit_syntax_recovery(root)
    recovered_parent_paths={r.get('syntax_normalization',{}).get('parent_path') for r in unique}
    recovered_parent_paths.discard(None)
    recovered_seconds=sum(ae.unseal(p)['seconds'] for p in recovered_parent_paths)
    archived_failed_seconds=sum(r['failed_generation_seconds'] for r in result['whole_experiment_cost']['earlier_schema_attempts'])
    result['whole_experiment_cost'].update(
        recovered_parent_generation_seconds_already_in_success_total=recovered_seconds,
        all_recorded_generation_seconds=sum(r['seconds'] for r in unique)+extraction['failed_inference_seconds']+archived_failed_seconds-recovered_seconds,
        generation_accounting='A schema-failed parent accepted by value-preserving normalization is one actual generation, not counted twice as failed and successful')
    result["execution_caveat"] = "Resumed after bounded schema-fallback implementation; exact prior successful requests reused with raw response/parent hashes. Not a fresh uninterrupted wall-time speed benchmark."
    monitor = root / "device_monitor.jsonl"
    if monitor.exists():
        samples = [json.loads(line) for line in monitor.read_text().splitlines() if line.strip()]
        result["sampled_device_usage"] = dict(samples=len(samples),interval_seconds=2,
            peak_used_mib=max(r["used_mib"] for r in samples),min_free_mib=min(r["free_mib"] for r in samples),
            scope="Device-wide sampled usage including other processes; monitor begins after initial model load")
    write_json(root / "evaluation/RESULT.json",result)
    export_conditions(root, plan)
    write_report(root,result,rows,reference,plan)
    return result


def audit_syntax_recovery(root):
    from .qwen35_state_delta_resume import semantic_fields, normalized_json
    rows=[]
    for path in (root/'calls').glob('*/*.json'):
        record=ae.unseal(path)
        if 'syntax_normalization' not in record:continue
        trace=record['syntax_normalization']
        if sha256(trace['extension_path'])!=trace['extension_sha256']:
            raise ValueError('syntax recovery parent or declaration changed')
        extension=ae.read(trace['extension_path'])
        if sha256(extension['module'])!=extension['module_sha256']:
            raise ValueError('syntax recovery code changed')
        expected=semantic_fields(record['raw_text'])
        if 'parent_path' in trace:
            if sha256(trace['parent_path'])!=trace['parent_sha256']:raise ValueError('syntax parent changed')
            parent=ae.unseal(trace['parent_path'])
            if parent['identity']!=record['identity'] or parent['raw_text']!=record['raw_text']:
                raise ValueError('syntax recovery changed original raw response')
        value,mode=normalized_json(record['raw_text'])
        if value!=record['result'] or mode!=trace['mode'] or semantic_fields(json.dumps(value))!=expected:
            raise ValueError('syntax normalization changed caption or evidence values')
        if json.loads(json.dumps(expected))!=trace['preserved_fields']:
            raise ValueError('syntax recovery field trace changed')
        rows.append(dict(path=str(path),parent=trace.get('parent_path'),ids=record['identity']['ids']))
    return dict(status='PASS_SYNTAX_NORMALIZATION_ALL_FIELDS_UNCHANGED',calls=len(rows),records=rows,
        runtime_extension_declared=(root/'syntax_recovery_extension_v3.json').exists())


def audit_correspondence(root, plan):
    from .qwen35_state_delta_match import complete_clause_schema
    extension_path = root / 'correspondence_schema_extension.json'
    extension = ae.read(extension_path) if extension_path.exists() else None
    if extension and sha256(extension['module']) != extension['module_sha256']:
        raise ValueError('correspondence schema extension code changed')
    base = ae.unseal(root / "budget_state.json")
    final = ae.unseal(root / "budget_checked.json")
    ordered = sorted(base["captions"],key=lambda r:(int(r["needs_check"]),len(r["uncertainties"]),
                     len(r["structured"]["changes"]),-r["segment"]),reverse=True)[:plan["max_checks"]]
    if [r["segment"] for r in ordered] != [r["segment"] for r in final["correspondence_audits"]]:
        raise ValueError("correspondence targets differ from frozen uncertainty/change ranking")
    calls = [ae.unseal(c["path"]) for c in final["calls"] if c["role"].removesuffix("_schema") == "correspondence"]
    for row,audit in zip(ordered,final["correspondence_audits"]):
        a,b = row["start"],row["end_exclusive"]
        ids=sorted(set(row["source_indices"]) | {max(0,a-6),min(plan["frames"]-1,b+5)})
        target_ids=[i for i in ids if a<=i<b]
        metadata=json.dumps(dict(target=[a,b],target_ids=target_ids,context_ids=[i for i in ids if i not in target_ids],clauses=sd.clauses(row["structured"])))
        prompt="INPUT METADATA (not the output schema):\n"+metadata+"\n"+sd.explicit_source_examples(plan["prompts"]["match"],target_ids)
        prompts = [prompt, sd.explicit_source_examples(prompt,ids)]
        if extension:
            extended = complete_clause_schema(prompt)
            prompts.extend([extended, sd.explicit_source_examples(extended,ids)])
        valid=[r for r in calls if r["identity"]["ids"]==ids and r["result"]==audit["result"]
               and any(r["identity"]["prompt"].startswith(p+"\nVALID SOURCE") for p in prompts)]
        if audit["ids"]!=ids or len(valid)!=1:
            raise ValueError("correspondence draft/source/prompt differs from actual inference")
    return dict(status="PASS_BOUNDED_SOURCE_CORRESPONDENCE_TRACE",checked_intervals=len(ordered),
        schema_extension_path=str(extension_path) if extension else None,
        schema_extension_sha256=sha256(extension_path) if extension else None,
        source_intervals=[dict(segment=r['segment'],start_seconds=r['start']/plan['fps'],
                              end_seconds=r['end_exclusive']/plan['fps']) for r in ordered],
        base_needs_check_count=sum(r['needs_check'] for r in base['captions']),
        base_nonempty_uncertainty_count=sum(bool(r['uncertainties']) for r in base['captions']),
        verdict_counts=dict(Counter(v['match'] for a in final['correspondence_audits'] for v in a['result']['verdicts'])),
        changed_intervals=[r["segment"] for r in final["correspondence_audits"] if r["decision"]["applied"]],
        removed_clauses=sum(len(r["decision"]["removed"]) for r in final["correspondence_audits"]),
        anchor_unconfirmed=[r["segment"] for r in final["correspondence_audits"] if r["decision"]["reason"]=="anchor_not_confirmed"])


def export_conditions(root, plan):
    """Inspectable original key PNGs and captions, with explicit provenance."""
    for arm in sd.ARMS:
        bundle = ae.unseal(root / f"{arm}.json")
        dest = root / "conditions" / arm
        images = dest / "selected_frames"
        images.mkdir(parents=True, exist_ok=True)
        for index in bundle["keys"]:
            source = Path(plan["source_frames"]) / f"{index}.png"
            output = images / f"{index:05d}.png"
            if output.exists() and sha256(output) != plan["source_frame_hashes"][str(index)]:
                raise ValueError("exported keyframe differs from original")
            if not output.exists():
                shutil.copyfile(source, output)
        write_json(dest / "keyframes.json", dict(indices=bundle["keys"],fps=plan["fps"]))
        write_json(dest / "captions.json",dict(provider="local_qwen35_state_delta",model_id=fb.MODEL_ID,
            revision=fb.REVISION,quantization="nf4",arm=arm,captions=bundle["captions"]))
        write_json(dest / "provenance.json",dict(signature=plan["signature"],policy=plan["policy"],
            extraction_sha256=sha256(root/f"{arm}.json"),source_video_sha256=plan["normalized_sha256"],
            original_png_sha256={str(i):sha256(images/f"{i:05d}.png") for i in bundle["keys"]},
            keyframes_sha256=sha256(dest/"keyframes.json"),captions_sha256=sha256(dest/"captions.json"),
            additional_model=False,training_executed=False,astra_reference_used_in_extraction=False,
            reconstruction_executed=False,existing_four_sample_importer_compatible=False))


def write_report(root,result,judgments,reference,plan):
    esc = html.escape
    bundles = {arm:ae.unseal(root/f"{arm}.json") for arm in sd.ARMS}
    body = ['<!doctype html><meta charset="utf-8"><title>Qwen 상태·변화 실험</title>',
        '<style>body{font-family:system-ui;max-width:1400px;margin:30px auto;background:#f7f8fa;color:#162035}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd4de;text-align:left}section{margin:25px 0;padding:20px;background:white}img{width:31%;margin:4px}.error{color:#a12020}.fact{color:#126334}small{color:#536171}</style>',
        '<h1>Qwen3.5-9B: 상태·변화 캡션과 키프레임 예산</h1>',
        '<p>tv_low_08 · 60초 · 고정 Astra 원본 기준 · 단일 개발 영상. 사람 정답, 미사용 영상, 복원 품질 평가는 아님.</p>',
        '<p>CodecCap/LFS의 아이디어를 차용한 추론 구조이며 원 논문의 학습 모델 재현은 아님. 시간은 성공한 고유 추론 합계로 로딩·실패·입출력 제외.</p>',
        '<p>152키 고정 조건은 과거에 추출된 키를 공급받으므로 그 키를 만들었던 비용이 제외됨. 예산·균등 조건은 자체 선택. 전체 비교 시간의 차이를 동일 조건의 속도 향상으로 해석하지 않음.</p>',
        '<p>평가 방법 이름은 가렸지만 완성 순서대로 한 방법씩 전달했으며, 동시 무작위 비교가 아님. 이전 프로토콜에서 상속된 순서 무작위화 문구의 정정은 delivery_clarification.json에 보존.</p>',
        '<table><tr><th>방법</th><th>키프레임</th><th>사건 전후</th><th>오류 구간 / 60</th><th>핵심 사실 / 123</th><th>추론 분</th></tr>']
    prior = result["historical_metrics"]["forward"]
    body.append(f'<tr><td>기존 Qwen 공유 관찰 · 피드백 전</td><td>{prior["keyframes"]["keyframes"]}</td><td>{prior["keyframes"]["event_coverage"]["covered"]}/18</td><td>{len(prior["error_bins"])}</td><td>{prior["recalled_facts"]}</td><td>{prior["costs"]["inference_seconds"]/60:.1f}</td></tr>')
    for arm,m in result["metrics"].items():
        body.append(f'<tr><td>{esc(m["label"])}</td><td>{m["keyframes"]}</td><td>{m["event_coverage"]["covered"]}/18</td><td>{len(m["error_bins"])}</td><td>{m["recalled_facts"]}</td><td>{m["costs"]["inference_seconds"]/60:.1f}</td></tr>')
    body.append('</table>')
    fixed,budget,uniform,checked=(result['metrics'][arm] for arm in sd.ARMS)
    audit=result['correspondence_validation']
    body.append('<h2>해석</h2><p>' + esc(
        f"같은 152키와 151구간 원본 표본을 유지한 상태·변화 표현은 오류 구간 {len(prior['error_bins'])}→{len(fixed['error_bins'])}, "
        f"사실 재현 {prior['recalled_facts']}→{fixed['recalled_facts']}입니다. 표현만으로 전반적 정확도 개선은 확인되지 않았습니다. "
        f"같은 79키에서 예산 선택은 균등 선택보다 사건 포착 {uniform['event_coverage']['covered']}→{budget['event_coverage']['covered']}/18, "
        f"오류 구간 {len(uniform['error_bins'])}→{len(budget['error_bins'])}, 사실 재현 {uniform['recalled_facts']}→{budget['recalled_facts']}/123입니다. "
        "사건 포착과 오류 감소의 이점, 사실 누락의 불리함이 공존합니다.")+'</p>')
    body.append('<p>' + esc(
        f"추가 대응 검사 {audit['checked_intervals']}구간, 수정 {len(audit['changed_intervals'])}구간, "
        f"판정 분포 {audit['verdict_counts']}. 캡션 {budget['captions']}개 중 needs_check=true는 "
        f"{audit['base_needs_check_count']}개, 불확실성 기록은 {audit['base_nonempty_uncertainty_count']}개였습니다. "
        f"우선순위 동률을 시간순으로 푼 결과 검사 범위가 {min(r['start_seconds'] for r in audit['source_intervals']):.3f}–"
        f"{max(r['end_seconds'] for r in audit['source_intervals']):.3f}초에 몰렸습니다. "
        "검사 후 문장·구간이 같은 항목은 기존 실제 Astra 판정을 재사용했으며 독립 재평가로 세지 않습니다.")+'</p>')
    body.append('<p>기본 추출 경로의 교체는 보류합니다. 다음 후보는 불확실성 자기 보고 대신 원본 변화·시간 분산으로 검사 대상을 고정하고, 미사용 영상에서 같은 키 예산으로 확인하는 것입니다. 이번 결과에 맞춰 재선택하거나 평가 정답을 추출기에 넣지 않았습니다.</p>')
    by_key = {(r["bin_id"],r["method"]):r for r in judgments}
    for ref in reference["bins"]:
        body.append(f'<section id="bin-{ref["id"]}"><h2>{ref["id"]}–{ref["id"]+1}초</h2>')
        for frame in [ref["start"],(ref["start"]+ref["end_exclusive"]-1)//2,min(plan["frames"]-1,ref["end_exclusive"]-1)]:
            path=Path(plan["source_frames"])/f"{frame}.png"
            data=base64.b64encode(path.read_bytes()).decode()
            body.append(f'<img title="원본 {frame} / {frame/24:.3f}s" src="data:image/png;base64,{data}">')
        body.append('<p>기준 사실: '+esc(json.dumps(ref["facts"],ensure_ascii=False))+'</p>')
        for arm,bundle in bundles.items():
            judge=by_key[(ref["id"],CODES[arm])]
            body.append(f'<h3>{esc(LABELS[arm])}</h3>')
            for row in bundle["captions"]:
                if row["start"]<ref["end_exclusive"] and ref["start"]<row["end_exclusive"]:
                    body.append(f'<p><small>{row["start"]/24:.3f}–{row["end_exclusive"]/24:.3f}s</small> {esc(row["caption"])}</p>')
            body.append('<p class="fact">재현: '+esc(', '.join(judge["covered_fact_ids"]))+'</p>')
            for error in judge["unsupported_claims"]:
                body.append('<p class="error">'+esc(error["quote"]+' — '+error["reason"])+'</p>')
            if judge["uncertain_claims"]:
                body.append('<p>불확실: '+esc(json.dumps(judge["uncertain_claims"],ensure_ascii=False))+'</p>')
        body.append('</section>')
    path=root/'evaluation/report.html'
    path.write_text('\n'.join(body),encoding='utf-8')
    write_json(root/'evaluation/report_manifest.json',dict(report_sha256=sha256(path),
        result_sha256=sha256(root/'evaluation/RESULT.json'),reference_sha256=result['reference_sha256']))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','reuse','report'))
    parser.add_argument('--root',type=Path,default=sd.ROOT)
    args=parser.parse_args()
    fn={'prepare':prepare,'reuse':reuse_identical,'report':aggregate}[args.action]
    print(json.dumps(fn(args.root.resolve()),ensure_ascii=False))
