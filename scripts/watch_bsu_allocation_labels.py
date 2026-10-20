"""Read-only allocation monitor; redraw changed rows without blanking the screen."""
import argparse
import math
import os
from pathlib import Path
import shutil
import sys
import time

from semantic_transmission.bsu_allocation_labels import DEFAULT, status_lines
from semantic_transmission.terminal_progress import fit


def redraw(previous, rows, columns, height):
    visible = tuple(fit(row, columns) for row in rows[:max(1, height-1)])
    changes = []
    for index in range(min(max(len(previous), len(visible)), max(1, height-1))):
        row = visible[index] if index < len(visible) else ''
        old = previous[index] if index < len(previous) else None
        if old != row:
            changes.append(f'\x1b[{index+1};1H{row}\x1b[K')
    return visible, ''.join(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--interval', type=float, default=3.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be finite and positive')
    live = sys.stdout.isatty() and os.environ.get('TERM') != 'dumb' and not args.once
    if not live:
        print('\n'.join(status_lines(args.output.resolve())), flush=True)
        return
    previous, size = (), None
    try:
        # Alternate screen keeps the shell intact. Subsequent frames only touch
        # changed rows, in one write, after collecting the complete new values.
        sys.stdout.write('\x1b[?1049h\x1b[?25l'); sys.stdout.flush()
        while True:
            try:
                rows = status_lines(args.output.resolve())
            except (OSError, ValueError) as error:
                rows = ['Monitor     Could not read current status; retrying', f'Error       {error}']
            current_size = shutil.get_terminal_size((100, 24))
            if current_size != size:
                previous = (); size = current_size
            previous, output = redraw(previous, rows, size.columns, size.lines)
            if output:
                sys.stdout.write(output); sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write('\x1b[?25h\x1b[?1049l'); sys.stdout.flush()


if __name__ == '__main__':
    main()
