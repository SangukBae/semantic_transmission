"""Apply the current FC-style prompt without modifying frozen inference engines.

The word-count gate checks formatting, not visual correctness. A failed caption
is retained with an explicit failure status and a nonzero exit code; it is never
silently shortened by cutting off words or sentences.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import qwen35_caption as base

PROMPT_PATH = base.REPO / 'configs/captions/faithful_visible_under80.txt'
POLICY_ID = 'fc_visible_facts_under80_v1'
MAX_WORDS = 79


def validate_output(path, prompt):
    record = json.loads(path.read_text())
    caption = record['caption']
    words = len(caption.split())
    valid = 0 < words <= MAX_WORDS and not record.get('truncated', False)
    original_status = record.get('status')
    record['caption_policy'] = dict(
        id=POLICY_ID, default_prompt_file=str(PROMPT_PATH),
        default_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        effective_prompt_sha256=hashlib.sha256(record['prompt'].encode()).hexdigest(),
        default_prompt_used=record['prompt'] == prompt,
        max_words=MAX_WORDS, word_count=words,
        word_count_method='nonempty whitespace-delimited tokens; consistent with prior audits',
        format_passed=valid,
        visual_factuality_verified=False,
    )
    if not valid:
        record['generation_status_before_policy_check'] = original_status
        record['status'] = 'CAPTION_POLICY_CHECK_FAILED'
        record['caption_policy']['failure_reason'] = (
            'empty_caption' if not words else
            'truncated_generation' if record.get('truncated', False) else
            'word_limit_exceeded'
        )
    base.write_json(path, record)
    return valid, words


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--engine', choices=['basic', 'advanced'], required=True)
    parser.add_argument('--show-prompt', action='store_true')
    args, child_args = parser.parse_known_args()
    prompt = PROMPT_PATH.read_text().strip()
    if args.show_prompt:
        print(prompt)
        return 0
    # The legacy modules read their default when constructing their CLI parser.
    # Their files and previous experiment hashes remain unchanged.
    base.PROMPT = prompt
    if args.engine == 'advanced':
        import qwen35_advanced_caption as engine
    else:
        engine = base
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument('--output', type=Path)
    output_args, _ = output_parser.parse_known_args(child_args)
    sys.argv = [str(Path(engine.__file__)), *child_args]
    if output_args.output is None:
        return engine.main()
    buffer = io.StringIO()
    existed_before = output_args.output.exists()
    try:
        with contextlib.redirect_stdout(buffer):
            code = engine.main()
    except SystemExit as exc:
        # The advanced engine writes a diagnostic before exiting on truncation.
        # Argparse help/errors can also exit here without producing a caption.
        code = exc.code
    if existed_before or not output_args.output.is_file():
        print(buffer.getvalue(), end='')
        return code
    valid, words = validate_output(output_args.output, prompt)
    if not valid:
        print(f'Caption rejected by format check: {words} words, maximum {MAX_WORDS}. '
              f'Details retained in {output_args.output}', file=sys.stderr)
        return 2
    if code not in (None, 0):
        return code
    print(buffer.getvalue(), end='')
    print(f'Caption format checked: {words}/{MAX_WORDS} words. Visual accuracy is not automatically certified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
