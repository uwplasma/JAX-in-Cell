# tests/test_plot_full.py
import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim

matplotlib.use("Agg")

import jax.numpy as jnp
import jaxincell._plot as plot_mod


# ----------------------------
# Small helpers to maximize coverage
# ----------------------------

def _synthetic_output(
    *,
    T=6,
    X=12,
    Ne=20,
    Ni=15,
    L=0.01,
    use_species=True,
    include_B=True,
    include_J=True,
    include_energy=True,
    zero_out_B=False,
):
    rng = np.random.default_rng(0)

    grid = np.linspace(-L / 2, L / 2, X).astype(np.float32)
    time_array = np.linspace(0.0, 1.0, T).astype(np.float32)

    # fields: (T, X, 3)
    E = np.zeros((T, X, 3), dtype=np.float32)
    B = np.zeros((T, X, 3), dtype=np.float32)
    J = np.zeros((T, X, 3), dtype=np.float32)

    # Make E_x nonzero; keep E_y ~0 to exercise threshold filtering.
    for t in range(T):
        E[t, :, 0] = 1e-2 * np.sin(2 * np.pi * (grid / L) + 0.5 * t)

    if include_B:
        # Put B_z nonzero so "has_B" becomes True
        for t in range(T):
            B[t, :, 2] = 5e-4 * np.cos(2 * np.pi * (grid / L) - 0.25 * t)
        if zero_out_B:
            B[:] = 0.0

    if include_J:
        # Make J_y largest so _pick_best_component_field picks 'y'
        for t in range(T):
            J[t, :, 1] = 2e-3 * np.sin(2 * np.pi * (grid / L) + 0.3 * t)

    charge_density = (1e-6 * rng.standard_normal((T, X))).astype(np.float32)

    def _make_species(charge, N, vscale):
        # positions/velocities: (T, N, 3)
        pos = np.zeros((T, N, 3), dtype=np.float32)
        vel = np.zeros((T, N, 3), dtype=np.float32)

        # x positions in [-L/2, L/2]
        pos[:, :, 0] = rng.uniform(-L / 2, L / 2, size=(T, N)).astype(np.float32)

        # velocities: x and z components; y ~0
        vel[:, :, 0] = (vscale * rng.standard_normal((T, N))).astype(np.float32)
        vel[:, :, 2] = (0.7 * vscale * rng.standard_normal((T, N))).astype(np.float32)
        return {"charge": charge, "positions": jnp.asarray(pos), "velocities": jnp.asarray(vel)}

    e_sp = _make_species(-1.0, Ne, vscale=2.0)
    i_sp = _make_species(+1.0, Ni, vscale=0.2)

    out = {
        "grid": grid,
        "time_array": time_array,
        "plasma_frequency": np.array(1.0, dtype=np.float32),
        "total_steps": int(T - 1),
        "length": float(L),
        "electric_field": jnp.asarray(E),
        "charge_density": jnp.asarray(charge_density),
    }

    if include_B:
        out["magnetic_field"] = jnp.asarray(B)
    if include_J:
        out["current_density"] = jnp.asarray(J)

    # Provide either species view or legacy split view.
    if use_species:
        out["species"] = [e_sp, i_sp]
    else:
        out["position_electrons"] = e_sp["positions"]
        out["velocity_electrons"] = e_sp["velocities"]
        out["position_ions"] = i_sp["positions"]
        out["velocity_ions"] = i_sp["velocities"]

    if include_energy:
        # Make strictly positive for log scale
        te = np.linspace(1.0, 1.2, T).astype(np.float32)
        out["total_energy"] = te
        out["electric_field_energy"] = np.linspace(0.2, 0.3, T).astype(np.float32)
        out["kinetic_energy_electrons"] = np.linspace(0.5, 0.6, T).astype(np.float32)
        out["kinetic_energy_ions"] = np.linspace(0.3, 0.3, T).astype(np.float32)
        out["magnetic_field_energy"] = np.linspace(1e-3, 2e-3, T).astype(np.float32)

    return out


class _FakeAnimation:
    """Lightweight replacement for matplotlib.animation.FuncAnimation."""
    def __init__(self, fig, func, frames, blit, interval, repeat_delay):
        self.fig = fig
        self.func = func
        self.frames = frames

        # Touch a couple frames to execute update logic (good for coverage)
        if isinstance(frames, int):
            to_try = [0, min(1, frames - 1)]
        else:
            to_try = [0, min(1, len(frames) - 1)]
        for i in to_try:
            func(i)

    def save(self, path, writer=None, dpi=None):
        # Don’t actually encode video; just create a file to prove we hit the path.
        with open(path, "wb") as f:
            f.write(b"FAKE_VIDEO")


def test_parse_direction_branches():
    assert plot_mod._parse_direction("x") == ["x"]
    assert plot_mod._parse_direction("xz") == ["x", "z"]

    with pytest.raises(TypeError):
        plot_mod._parse_direction(None)

    with pytest.raises(ValueError):
        plot_mod._parse_direction("xx")

    with pytest.raises(ValueError):
        plot_mod._parse_direction("abc")


def test_pick_best_component_field_picks_largest():
    out = _synthetic_output(include_B=True, include_J=True)
    # force y component to be largest in current_density (already is)
    best = plot_mod._pick_best_component_field(out, "current_density", threshold=1e-12, allowed_axes=None)
    assert best is not None
    axis, data = best
    assert axis == "y"
    assert data.ndim == 2  # (T, X)


def test_ffmpeg_helpers_without_ffmpeg(monkeypatch):
    # Ensure ffmpeg missing path (covers shutil.which None branch)
    plot_mod._ffmpeg_encoders_text.cache_clear()
    monkeypatch.setattr(plot_mod.shutil, "which", lambda _: None)
    assert plot_mod._ffmpeg_encoders_text() == ""
    assert plot_mod._ffmpeg_has_encoder("libx264") is False


def test_ffmpeg_helpers_with_fake_ffmpeg(monkeypatch):
    # Fake ffmpeg output path (covers subprocess branch + encoder matching)
    plot_mod._ffmpeg_encoders_text.cache_clear()
    monkeypatch.setattr(plot_mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        plot_mod.subprocess,
        "check_output",
        lambda *args, **kwargs: " V....D libx264 \n V....D h264_videotoolbox \n",
    )
    assert plot_mod._ffmpeg_has_encoder("libx264") is True
    assert plot_mod._ffmpeg_has_encoder("h264_videotoolbox") is True
    assert plot_mod._ffmpeg_has_encoder("libx265") is False


def test_plot_species_view_no_save(monkeypatch):
    # Covers species path in _combine_by_charge_sign + no-save branch
    out = _synthetic_output(use_species=True, include_B=False, include_J=True, include_energy=True)
    monkeypatch.setattr(plot_mod.plt, "show", lambda: None)
    plot_mod.plot(out, direction="xz", show=False, save_mp4=None)


def test_plot_legacy_split_with_B_and_J_and_energy(monkeypatch):
    # Covers legacy split path + has_B gating + pick_best_component_field + overlays
    out = _synthetic_output(use_species=False, include_B=True, include_J=True, include_energy=True)
    monkeypatch.setattr(plot_mod.plt, "show", lambda: None)
    plot_mod.plot(out, direction="xz", show=False, save_mp4=None)


def test_plot_save_mp4_success(monkeypatch, tmp_path):
    # Covers save_mp4 branch without real ffmpeg by patching FuncAnimation + writer factory
    out = _synthetic_output(use_species=True, include_B=True, include_J=True, include_energy=True)

    monkeypatch.setattr(plot_mod, "FuncAnimation", _FakeAnimation)
    monkeypatch.setattr(plot_mod.plt, "show", lambda: None)

    # Return any dummy object as "writer"
    monkeypatch.setattr(plot_mod, "_make_ffmpeg_writer_auto", lambda **kwargs: object())

    out_path = tmp_path / "anim.mp4"
    plot_mod.plot(out, direction="xz", show=False, save_mp4=str(out_path), save_stride=2, fps=10, dpi=80)
    assert out_path.exists()
    assert out_path.read_bytes().startswith(b"FAKE_VIDEO")


def test_plot_save_mp4_failure_warns(monkeypatch, tmp_path):
    # Covers save_mp4 exception path (warnings.warn)
    out = _synthetic_output(use_species=True, include_B=True, include_J=True, include_energy=True)

    monkeypatch.setattr(plot_mod, "FuncAnimation", _FakeAnimation)
    monkeypatch.setattr(plot_mod.plt, "show", lambda: None)

    def _boom(**kwargs):
        raise RuntimeError("no ffmpeg")

    monkeypatch.setattr(plot_mod, "_make_ffmpeg_writer_auto", _boom)

    out_path = tmp_path / "anim.mp4"
    with pytest.warns(RuntimeWarning):
        plot_mod.plot(out, direction="x", show=False, save_mp4=str(out_path))


class _DummyWriter:
    def __init__(self, fps, codec, bitrate, extra_args):
        self.fps = fps
        self.codec = codec
        self.bitrate = bitrate
        self.extra_args = list(extra_args)


def test_ffmpeg_encoders_text_exception_branch(monkeypatch):
    plot_mod._ffmpeg_encoders_text.cache_clear()
    monkeypatch.setattr(plot_mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def boom(*args, **kwargs):
        raise RuntimeError("ffmpeg broken")

    monkeypatch.setattr(plot_mod.subprocess, "check_output", boom)
    assert plot_mod._ffmpeg_encoders_text() == ""


def test_auto_codec_order_for_mp4_filters_and_keeps_order(monkeypatch):
    # Make only these encoders "available"
    available = {"h264_videotoolbox", "libx264", "libx265"}

    monkeypatch.setattr(plot_mod, "_ffmpeg_has_encoder", lambda name: name in available)
    got = plot_mod._auto_codec_order_for_mp4()

    # Must preserve the candidate ordering from the function:
    # h264_videotoolbox ... libx264 ... libx265
    assert got == ["h264_videotoolbox", "libx264", "libx265"]


def test_make_ffmpeg_writer_webm_vp9(monkeypatch):
    # Force VP9 branch
    monkeypatch.setattr(plot_mod, "_ffmpeg_has_encoder", lambda name: name == "libvpx-vp9")
    monkeypatch.setattr(mpl_anim, "FFMpegWriter", _DummyWriter)

    w = plot_mod._make_ffmpeg_writer_auto(out_path="out.webm", fps=12, crf=None, preset=None)
    assert isinstance(w, _DummyWriter)
    assert w.codec == "libvpx-vp9"
    # default q for vp9 is 38 when crf=None
    assert "-crf" in w.extra_args
    assert "38" in w.extra_args
    assert "-row-mt" in w.extra_args
    assert "-speed" in w.extra_args
    assert "-pix_fmt" in w.extra_args


def test_make_ffmpeg_writer_webm_fallback_libx264(monkeypatch):
    # VP9 unavailable => fallback to libx264 in webm branch
    monkeypatch.setattr(plot_mod, "_ffmpeg_has_encoder", lambda name: False)
    monkeypatch.setattr(mpl_anim, "FFMpegWriter", _DummyWriter)

    w = plot_mod._make_ffmpeg_writer_auto(out_path="out.webm", fps=20, crf=None, preset=None)
    assert w.codec == "libx264"
    # webm fallback uses basic vf_even + pix_fmt
    assert "-vf" in w.extra_args
    assert "-pix_fmt" in w.extra_args


def test_make_ffmpeg_writer_mp4_auto_libx264_adds_avc1_and_crf_preset(monkeypatch):
    monkeypatch.setattr(plot_mod, "_auto_codec_order_for_mp4", lambda: ["libx264"])
    monkeypatch.setattr(mpl_anim, "FFMpegWriter", _DummyWriter)

    w = plot_mod._make_ffmpeg_writer_auto(out_path="out.mp4", fps=30, crf=None, preset=None)
    assert w.codec == "libx264"

    # defaults for libx264: preset veryfast, crf 30
    # (order in list matters; just check inclusion)
    assert "-preset" in w.extra_args
    assert "veryfast" in w.extra_args
    assert "-crf" in w.extra_args
    assert "30" in w.extra_args

    # QuickTime tag for H.264 in mp4
    assert "-tag:v" in w.extra_args
    assert "avc1" in w.extra_args

    # even-dimension filter + faststart always on mp4
    assert "+faststart" in w.extra_args


def test_make_ffmpeg_writer_mp4_auto_libx265_adds_hvc1_and_crf_preset(monkeypatch):
    monkeypatch.setattr(plot_mod, "_auto_codec_order_for_mp4", lambda: ["libx265"])
    monkeypatch.setattr(mpl_anim, "FFMpegWriter", _DummyWriter)

    w = plot_mod._make_ffmpeg_writer_auto(out_path="out.mp4", fps=30, crf=None, preset=None)
    assert w.codec == "libx265"

    # defaults for HEVC/libx265: crf 32
    assert "-preset" in w.extra_args
    assert "veryfast" in w.extra_args
    assert "-crf" in w.extra_args
    assert "32" in w.extra_args

    # QuickTime tag for HEVC in mp4
    assert "-tag:v" in w.extra_args
    assert "hvc1" in w.extra_args


def test_make_ffmpeg_writer_mp4_hardware_encoder_hits_pass_branch(monkeypatch):
    # This hits the "hardware encoders: ... pass" branch
    monkeypatch.setattr(plot_mod, "_auto_codec_order_for_mp4", lambda: ["h264_videotoolbox"])
    monkeypatch.setattr(mpl_anim, "FFMpegWriter", _DummyWriter)

    w = plot_mod._make_ffmpeg_writer_auto(out_path="out.mp4", fps=30, crf=None, preset=None)
    assert w.codec == "h264_videotoolbox"

    # Should still include mp4 compatibility args
    assert "-movflags" in w.extra_args
    assert "+faststart" in w.extra_args
    assert "-tag:v" in w.extra_args
    assert "avc1" in w.extra_args

    # Hardware branch should NOT prepend crf/preset like libx264/libx265
    assert "-crf" not in w.extra_args
    assert "-preset" not in w.extra_args


def test_make_ffmpeg_writer_mp4_codec_override_missing_raises(monkeypatch):
    monkeypatch.setattr(plot_mod, "_ffmpeg_has_encoder", lambda name: False)
    with pytest.raises(RuntimeError):
        plot_mod._make_ffmpeg_writer_auto(
            out_path="out.mp4",
            fps=30,
            crf=None,
            preset=None,
            codec_override="libx264",
        )


def test_cbar_label_top_fontsize_branch():
    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(4).reshape(2, 2))
    cb = fig.colorbar(im, ax=ax)
    plot_mod._cbar_label_top(cb, "counts", fontsize=9, pad=6)
    plt.close(fig)


def test_plot_show_true_calls_show(monkeypatch):
    out = _synthetic_output(use_species=True, include_B=True, include_J=True, include_energy=True)

    # Make animation cheap + deterministic
    monkeypatch.setattr(plot_mod, "FuncAnimation", _FakeAnimation)

    called = {"n": 0}
    monkeypatch.setattr(plot_mod.plt, "show", lambda: called.__setitem__("n", called["n"] + 1))

    plot_mod.plot(out, direction="x", show=True, save_mp4=None)
    assert called["n"] == 1
