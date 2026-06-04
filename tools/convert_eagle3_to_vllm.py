#!/usr/bin/env python3
"""
Convert a `speculators`-trained EAGLE-3 checkpoint into the layout that
vLLM's speculative-decoding loader (`Eagle3LlamaForCausalLM`) expects.

The trainer at `speculators/scripts/train.py` already saves tensors with
the vLLM-compatible keys:

    layers.0.self_attn.{q,k,v,o}_proj.weight
    layers.0.mlp.{gate,up,down}_proj.weight
    layers.0.{input_layernorm,post_attention_layernorm,hidden_norm}.weight
    embed_tokens.weight
    fc.weight           # (H, 3H)  auxiliary-hidden-state projection
    lm_head.weight      # (draft_vocab, H)
    norm.weight
    d2t                 # renamed to draft_id_to_target_id by vLLM
    t2d                 # skipped by vLLM

vLLM routes any config containing `speculators_config` through
`SpeculatorsConfig.from_pretrained`, which calls the `eagle3` algo-updater
in vllm/transformers_utils/configs/speculators/algos.py to build a clean
LlamaConfig with the Eagle-3 extras (`draft_vocab_size`,
`norm_before_residual`, `norm_before_fc`, `eagle_aux_hidden_state_layer_ids`).
So the weight file can be used as-is; we only sanitize config.json.

Sanitization performed:
1. Drop `auto_map` - malformed ("" key) + trust-remote-code trap.
2. Force top-level `architectures` to ["Eagle3LlamaForCausalLM"].
3. Drop stale HF housekeeping keys (`transformers_version`, top-level
   `dtype`) that trip Transformers-v5 validation.
4. Leave `speculators_config`, `speculators_model_type`, `draft_vocab_size`,
   `norm_before_residual`, `norm_before_fc`, `target_hidden_size`,
   `eagle_aux_hidden_state_layer_ids`, and the full
   `transformer_layer_config` (with `rope_parameters`) untouched.

MRoPE preservation:
    vLLM's `LlamaAttention._init_rotary_emb` reads `getattr(config,
    "rope_parameters", None)` and hands it to `get_rope`, which dispatches
    on `"mrope_section" in rope_parameters` -> `MRotaryEmbedding` (or
    `MRotaryEmbeddingInterleaved` when `mrope_interleaved=True`). Because
    `_sanitize_config` only edits top-level keys, the nested
    `transformer_layer_config.rope_parameters` block (including
    `mrope_section`, `mrope_interleaved`, `partial_rotary_factor`,
    `rope_theta`, `rope_type`) passes through byte-for-byte and is picked
    up verbatim by vLLM at load time. We additionally *validate* the
    block here so a corrupted trainer checkpoint fails at conversion
    instead of silently falling back to plain 1D RoPE at server startup.

Usage:
    python3 convert_eagle3_to_vllm.py \\
        --src /home/ray/qwen36_dflash/checkpoints/qwen36_caption_eagle3_v1/checkpoint_best \\
        --dst /home/ray/qwen36_dflash/checkpoints/qwen36_caption_eagle3_v1/vllm_ready
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REQUIRED_WEIGHT_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

_TOP_LEVEL_KEYS_TO_DROP: tuple[str, ...] = (
    "auto_map",
    "transformers_version",
    "dtype",
)

_VLLM_EAGLE3_ARCH = "Eagle3LlamaForCausalLM"

# rope_parameters fields that materially affect the rotary math. We compare
# these byte-for-byte between the draft and its verifier. `rope_type` is
# included because vLLM's `get_rope` dispatches on it (default vs yarn vs
# linear, etc.); a type mismatch silently produces different cos/sin tables.
_ROPE_PARITY_FIELDS: tuple[str, ...] = (
    "rope_type",
    "rope_theta",
    "partial_rotary_factor",
    "mrope_interleaved",
    "mrope_section",
)


def _get_verifier_rope_parameters(verifier_path: str) -> dict | None:
    """Load the verifier's text-side `rope_parameters`, or return None.

    Handles three config layouts seen in the Qwen family:
      * Qwen3-Omni Thinker: `thinker_config.text_config.rope_parameters`
      * Qwen3.5 / Qwen3.6 MoE (flat): `text_config.rope_parameters`
      * Text-only Qwen3 (decoder-only): top-level `rope_parameters`

    Returns None on any failure (missing transformers, offline/unreachable
    model path, no rope_parameters on the text config, etc.) so that the
    parity check can warn-and-continue instead of hard-failing on hosts
    that intentionally export without the verifier mounted.
    """
    try:
        from transformers import AutoConfig  # noqa: PLC0415
    except ImportError:
        return None
    try:
        cfg = AutoConfig.from_pretrained(verifier_path, trust_remote_code=True)
    except Exception:  # noqa: BLE001  (network, filesystem, parse errors)
        return None

    # Dispatch on layout.
    thinker = getattr(cfg, "thinker_config", None)
    if thinker is not None:
        text = getattr(thinker, "text_config", None) or thinker
    else:
        text = getattr(cfg, "text_config", None) or cfg
    rp = getattr(text, "rope_parameters", None)
    if isinstance(rp, dict):
        return rp
    return None


def _equivalent_under_full_head_hack(
    draft_rp: dict, verifier_rp: dict, head_dim: int,
) -> bool:
    """Return True if ``draft_rp`` equals ``verifier_rp`` after applying the
    ``--draft-mrope-full-head-hack`` transformation to the verifier.

    The hack (see ``speculators/scripts/train.py``) rescales
    ``mrope_section`` by ``1/partial_rotary_factor`` and pins
    ``partial_rotary_factor=1.0`` so HF's rotate_half-style rotation (used
    by the trainer) pairs the same channels as vLLM's neox-partial rotation
    (used at inference). When that hack is active the draft's
    rope_parameters INTENTIONALLY diverge from the verifier's — on those
    two fields only. This helper normalizes the verifier side so the
    parity check accepts intentional-hack drift while still catching
    unrelated drift.
    """
    verif_partial = float(verifier_rp.get("partial_rotary_factor", 1.0))
    if verif_partial >= 1.0:
        return False  # no hack needed; fields should match as-is
    inv = 1.0 / verif_partial
    if abs(inv - round(inv)) > 1e-6:
        return False  # can't cleanly rescale; hack wouldn't apply
    scale = int(round(inv))
    verif_section = verifier_rp.get("mrope_section")
    if not isinstance(verif_section, list):
        return False
    expected_section = [int(x) * scale for x in verif_section]
    if 2 * sum(expected_section) != head_dim:
        return False
    # Compare every parity field, applying the hack transformation to the
    # verifier side.
    for key in _ROPE_PARITY_FIELDS:
        draft_v = draft_rp.get(key)
        if key == "partial_rotary_factor":
            verif_v = 1.0
        elif key == "mrope_section":
            verif_v = expected_section
        else:
            verif_v = verifier_rp.get(key)
        if draft_v is None and verif_v is None:
            continue
        if draft_v != verif_v:
            return False
    return True


def _check_rope_matches_verifier(draft_rp: dict, spec_cfg: dict | None,
                                  head_dim: int | None = None) -> None:
    """Fail if the draft's `rope_parameters` diverges from the verifier's.

    This closes the class of bugs where the draft `config.json` is
    hand-edited (e.g., to work around a rotary mismatch) but the
    speculators training run was NOT re-done on the same patched config:
    the deployed model would then rotate a different set of head-dim
    channels at inference than the trainer used, producing the exact
    sub-percent cosine divergence that tanked Eagle3-v4 acceptance.

    Hard-fails on any divergence of a member of `_ROPE_PARITY_FIELDS`.
    Soft-skips (prints a warning) when the verifier config cannot be
    loaded from its `name_or_path`, so exports can still run on hosts
    that do not mount the verifier weights.

    ONE intentional divergence is silently accepted: when the trainer was
    run with ``--draft-mrope-full-head-hack`` (default ON), the draft's
    ``partial_rotary_factor`` and ``mrope_section`` are rescaled versions
    of the verifier's. ``_equivalent_under_full_head_hack`` detects this
    case and lets the export proceed without noise.
    """
    if not isinstance(spec_cfg, dict):
        return  # other validators will complain about this separately
    verifier_path = spec_cfg.get("verifier", {}).get("name_or_path")
    if not verifier_path:
        return  # other validators will complain about this separately

    verifier_rp = _get_verifier_rope_parameters(verifier_path)
    if verifier_rp is None:
        print(
            f"[WARN] could not load verifier rope_parameters from "
            f"{verifier_path!r}; skipping draft/verifier parity check. "
            f"This is usually fine on export-only hosts, but you should "
            f"re-run export on a box that can load the verifier to catch "
            f"train/inference RoPE drift."
        )
        return

    # Intentional divergence from the full-head-rotation training hack.
    if head_dim is not None and _equivalent_under_full_head_hack(
        draft_rp, verifier_rp, head_dim
    ):
        print(
            "[OK] draft rope_parameters differ from verifier as expected "
            "under --draft-mrope-full-head-hack (partial_rotary_factor=1.0, "
            f"mrope_section rescaled by 1/{verifier_rp.get('partial_rotary_factor')}); "
            "skipping parity check on those fields."
        )
        return

    mismatches: list[tuple[str, object, object]] = []
    for key in _ROPE_PARITY_FIELDS:
        draft_v = draft_rp.get(key, None)
        verif_v = verifier_rp.get(key, None)
        # Both absent => OK (field simply isn't used by this model family).
        if draft_v is None and verif_v is None:
            continue
        if draft_v != verif_v:
            mismatches.append((key, draft_v, verif_v))

    if mismatches:
        lines = [
            "rope_parameters drift between draft and verifier "
            "(train/inference mismatch risk):"
        ]
        for key, draft_v, verif_v in mismatches:
            lines.append(
                f"  {key}: draft={draft_v!r}  verifier={verif_v!r}"
            )
        lines.append(
            "Either (a) revert the draft's rope_parameters to match the "
            "verifier and retrain, or (b) if this divergence is intentional "
            "(e.g. the trainer was explicitly run on the patched config), "
            "bypass this check by setting "
            "EAGLE3_CONVERT_ALLOW_ROPE_DRIFT=1."
        )
        if os.environ.get("EAGLE3_CONVERT_ALLOW_ROPE_DRIFT") == "1":
            print("[WARN] " + "\n       ".join(lines))
        else:
            raise ValueError("\n".join(lines))


def _sanitize_config(cfg: dict) -> dict:
    """Pure-Python O(#top-level-keys) transform. Returns a new dict."""
    # Shallow copy the top level only -- nested dicts (transformer_layer_config,
    # speculators_config) pass through byte-for-byte, preserving rope_parameters,
    # YaRN config, verifier.name_or_path, proposal_methods, etc.
    out = {k: v for k, v in cfg.items() if k not in _TOP_LEVEL_KEYS_TO_DROP}

    # SpeculatorsConfig builds the inner pre_trained_config with the correct
    # architecture, but ModelConfig.__init__ reads the outer list before
    # spec-decode expansion -- keep them in sync.
    out["architectures"] = [_VLLM_EAGLE3_ARCH]

    # Eagle-3 invariant checks (fail loudly now rather than at vLLM load).
    tlc = out.get("transformer_layer_config")
    if not isinstance(tlc, dict):
        raise ValueError("config.json is missing `transformer_layer_config`")
    for required in ("hidden_size", "num_attention_heads", "num_key_value_heads",
                     "head_dim", "intermediate_size", "vocab_size",
                     "rms_norm_eps", "rope_parameters"):
        if required not in tlc:
            raise ValueError(
                f"transformer_layer_config missing required key: {required!r}"
            )

    # --- MRoPE preservation guard -------------------------------------------
    # vLLM's get_rope() in
    #   vllm/model_executor/layers/rotary_embedding/__init__.py
    # dispatches on `"mrope_section" in rope_parameters` to pick
    # MRotaryEmbedding (or MRotaryEmbeddingInterleaved when
    # mrope_interleaved=True). If any of these fields are missing / malformed
    # vLLM silently falls back to plain 1D RoPE, which produces wrong cos/sin
    # at inference and usually shows up as near-zero acceptance rate.
    # Trap that here instead.
    rp = tlc["rope_parameters"]
    if not isinstance(rp, dict):
        raise ValueError(
            f"transformer_layer_config.rope_parameters must be a dict, "
            f"got {type(rp).__name__}"
        )
    if "mrope_section" in rp:
        mrope_section = rp["mrope_section"]
        if not (isinstance(mrope_section, list)
                and len(mrope_section) == 3
                and all(isinstance(x, int) and x > 0 for x in mrope_section)):
            raise ValueError(
                f"rope_parameters.mrope_section must be a list of exactly 3 "
                f"positive ints, got {mrope_section!r}"
            )
        # Check the mrope_section dims are consistent with the rotary dim
        # that vLLM will build. get_rope uses
        #   rotary_dim = int(head_dim * partial_rotary_factor)
        # and MRotaryEmbedding.__init__ asserts
        #   sum(mrope_section) == rotary_dim // 2
        partial = rp.get("partial_rotary_factor", 1.0)
        rotary_dim = int(tlc["head_dim"] * partial)
        if sum(mrope_section) != rotary_dim // 2:
            raise ValueError(
                f"mrope_section={mrope_section} sums to {sum(mrope_section)}, "
                f"but vLLM expects sum == rotary_dim // 2 = "
                f"{rotary_dim // 2} (head_dim={tlc['head_dim']}, "
                f"partial_rotary_factor={partial}). "
                f"This would trip an assert in MRotaryEmbedding.__init__."
            )
        # mrope_interleaved is optional; vLLM defaults it to False, but
        # our trainer always writes it explicitly -- complain on stray types.
        mi = rp.get("mrope_interleaved", False)
        if not isinstance(mi, bool):
            raise ValueError(
                f"rope_parameters.mrope_interleaved must be bool, "
                f"got {type(mi).__name__}={mi!r}"
            )

        # --- Train/inference parity guard -----------------------------------
        # Self-consistency of the draft's `rope_parameters` (above) is
        # necessary but NOT sufficient. The real bug that blew up Eagle3-v4
        # (cos-sim of draft logits 0.93, pos0 acceptance -17pp) was a
        # SILENT divergence between the draft config and what the trainer
        # actually did: upstream `Qwen3OmniMoeThinkerTextRotaryEmbedding`
        # ignored `partial_rotary_factor` and rotated the full head_dim.
        # That was fixed in speculators' `_select_rotary_emb_class`
        # (PartialMRoPE subclass). To keep this from regressing we now
        # also check that the draft's `rope_parameters` matches the
        # verifier's `text_config.rope_parameters` byte-for-byte: any
        # divergence means someone hand-edited the draft config (e.g.,
        # the old Option C workaround `partial_rotary_factor=1.0,
        # mrope_section=[44,44,40]`) which produces a train/inference
        # mismatch unless the trainer was also run on the patched config.
        # ``head_dim`` is threaded through so the intentional
        # ``--draft-mrope-full-head-hack`` drift is recognized and allowed.
        _check_rope_matches_verifier(
            rp, spec_cfg=out.get("speculators_config"),
            head_dim=tlc.get("head_dim"),
        )
        # ---------------------------------------------------------------------
    # ------------------------------------------------------------------------

    spec_cfg = out.get("speculators_config")
    if not isinstance(spec_cfg, dict):
        raise ValueError("config.json is missing `speculators_config`")
    if spec_cfg.get("algorithm") != "eagle3":
        raise ValueError(
            f"speculators_config.algorithm must be 'eagle3', "
            f"got {spec_cfg.get('algorithm')!r}"
        )
    if not spec_cfg.get("proposal_methods"):
        raise ValueError("speculators_config.proposal_methods is empty")
    if "speculative_tokens" not in spec_cfg["proposal_methods"][0]:
        raise ValueError(
            "speculators_config.proposal_methods[0].speculative_tokens missing"
        )
    if not spec_cfg.get("verifier", {}).get("name_or_path"):
        raise ValueError("speculators_config.verifier.name_or_path missing")

    if out.get("speculators_model_type") != "eagle3":
        raise ValueError(
            f"speculators_model_type must be 'eagle3', "
            f"got {out.get('speculators_model_type')!r}"
        )

    # EAGLE-3 draft has num_hidden_layers=1; vLLM iterates exactly this many
    # layers, while weights only exist under `layers.0.*`.
    if tlc.get("num_hidden_layers", 1) != 1:
        raise ValueError(
            f"EAGLE-3 draft must have num_hidden_layers=1, "
            f"got {tlc.get('num_hidden_layers')}"
        )

    # eagle_aux_hidden_state_layer_ids must have exactly 3 entries --
    # this makes fc.weight shape (H, 3H) meaningful.
    aux_ids = out.get("eagle_aux_hidden_state_layer_ids")
    if not (isinstance(aux_ids, list) and len(aux_ids) == 3):
        raise ValueError(
            f"eagle_aux_hidden_state_layer_ids must be a list of exactly 3 "
            f"layer ids, got {aux_ids!r}"
        )

    return out


def _copy_weights(src: Path, dst: Path) -> None:
    """
    Always perform a real byte-for-byte copy of the safetensors file.

    Rationale: downstream tooling (vLLM workers on remote hosts, container
    image builds, rsync to shared storage, checkpoint archival, etc.) can
    silently break on symlinks/hardlinks when the source directory is
    moved, pruned, or not mounted. A standalone copy makes `dst` fully
    self-contained.

    We resolve any existing symlink at `dst` first (to avoid overwriting
    the source through a dangling link), then use `shutil.copyfile` which
    internally uses `os.sendfile` / `copy_file_range` on Linux for
    zero-copy kernel-side transfer when src and dst live on the same
    filesystem -- so a ~1.34 GB weight file typically completes in
    well under a second on NVMe without touching user-space buffers.
    """
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    # copyfile (not copy2) -- we don't need to preserve the training
    # checkpoint's mtime/permissions on the production artefact.
    shutil.copyfile(src, dst)


def convert(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Source checkpoint dir not found: {src}")
    src_cfg = src / CONFIG_FILE
    src_wts = src / REQUIRED_WEIGHT_FILE
    if not src_cfg.is_file():
        raise FileNotFoundError(f"Missing {CONFIG_FILE} in {src}")
    if not src_wts.is_file():
        raise FileNotFoundError(f"Missing {REQUIRED_WEIGHT_FILE} in {src}")

    dst.mkdir(parents=True, exist_ok=True)

    # 1. rewrite config.json atomically
    with src_cfg.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    new_cfg = _sanitize_config(cfg)

    tmp_cfg = dst / (CONFIG_FILE + ".tmp")
    with tmp_cfg.open("w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_cfg, dst / CONFIG_FILE)

    # 2. copy weights (real byte-for-byte copy so `dst` is self-contained)
    _copy_weights(src_wts.resolve(), dst / REQUIRED_WEIGHT_FILE)

    # 3. Do NOT copy training-only files: config.py would re-introduce the
    # auto_map trap, optimizer_state_dict.pt wastes ~540 MB.

    print(f"[OK] vLLM-ready EAGLE-3 checkpoint written to: {dst}")
    print(f"     - {CONFIG_FILE}            (sanitized)")
    print(f"     - {REQUIRED_WEIGHT_FILE}   (copied from source)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", type=Path, required=True,
                   help="Path to speculators training checkpoint_best dir.")
    p.add_argument("--dst", type=Path, required=True,
                   help="Output dir passed to vLLM --speculative-config.model")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(args.src.resolve(), args.dst.resolve())
