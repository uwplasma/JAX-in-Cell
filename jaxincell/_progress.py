"""A progress meter for a long run, on the host and outside the traced region.

The obvious place to put one is inside the loop, with ``jax.debug.callback``. It does not
belong there. ``_run`` is ``jax.jit``-ed, so its body runs once per *compilation*: a bar
built there is trace-time state, and the second call with the same static arguments reuses
the compiled program and the bar that was closed at the end of the first. Beyond that, a
debug callback fires on the forward pass only under ``grad``, is unrolled across the mapped
axis under ``vmap`` so that one bar receives ``B`` times its updates, raises on more than one
device when ordered, and dispatches asynchronously, so its output can arrive after the
function has returned. And it puts a side effect in the differentiated path, which is what
the separation between the kernels and the orchestration exists to prevent.

So the meter lives here instead. The run is split into groups on the host, each group is the
same compiled program, and the host blocks once per group and prints. The state carries
everything, so a grouped run *is* the ungrouped run -- the same guarantee a restart rests on.
It costs about 1.5 %.

Nothing here touches the simulation, the random numbers or the output. It reports two
host-side facts, how many steps have finished and how long that took.
"""
import sys
import time

__all__ = ["Progress", "reporter"]


def _clock(seconds):
    """``h:mm:ss``, or ``--:--`` for a rate that is not yet known."""
    if not seconds < float("inf"):
        return "--:--"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


class Progress:
    """Elapsed time, fraction done, rate and an estimate of what is left, on one line.

    A terminal gets a line rewritten with a carriage return; anything else -- a log, a CI
    transcript, a pipe -- gets a new line each time, because a carriage return in a log file is
    a line nobody can read.
    """

    def __init__(self, total, stream=None, interactive=None):
        self.total = total
        self.stream = sys.stderr if stream is None else stream
        self.interactive = getattr(self.stream, "isatty", lambda: False)() \
            if interactive is None else interactive
        self.start = time.perf_counter()
        self.open = False

    def __call__(self, done, total=None):
        total = self.total if total is None else total
        elapsed = time.perf_counter() - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        left = (total - done) / rate if rate > 0 else float("inf")
        line = (f"{done}/{total} steps  {100 * done / max(total, 1):5.1f} %  "
                f"{rate:,.0f} steps/s  {_clock(elapsed)} elapsed  {_clock(left)} left")
        self.stream.write(f"\r{line}   " if self.interactive else f"{line}\n")
        self.stream.flush()
        self.open = True
        if done >= total:
            self.close()

    def close(self):
        """End the line, so that an interrupted run leaves a terminal someone can type in."""
        if self.open and self.interactive:
            self.stream.write("\n")
            self.stream.flush()
        self.open = False


def reporter(verbose, total):
    """What ``run(verbose=...)`` means: ``None`` for no meter, or something to call with the
    number of steps finished and the number there are.

    ``True`` is :class:`Progress`. Anything else callable is used as it is, which is how a
    ``tqdm`` bar goes in without ``tqdm`` becoming a dependency of this package::

        bar = tqdm.tqdm(total=steps)
        out = simulation.run(steps, verbose=lambda done, total: bar.update(done - bar.n))

    One reporter, one code path, and the one the tests exercise is the one that ships.
    """
    if verbose is True:
        return Progress(total)
    if verbose is False or verbose is None:
        return None
    if not callable(verbose):
        raise ValueError(f"verbose is True, False, or something to call with (done, total), not {verbose!r}")
    return verbose
