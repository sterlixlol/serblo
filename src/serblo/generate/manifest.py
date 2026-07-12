"""Build the M3 generation manifest: the model-agnostic prompt list (PRD §7.4, §7.9).

Loads ``configs/generation-config.yaml``, samples persona axes *combinatorially* per
register (never from a flat menu: the cross-product IS the within-register diversity),
fills the register template, then appends the §7.9 transport wrapper so every generation
comes back as ``<<<DOC>>> … <<<END>>>`` that the converter can slice cleanly.

Each record is a neutral prompt-spec (``id``, ``register``, ``axes``, ``prompt``); the
per-model submit step (Vertex batch / MiMo / NIM …) reshapes these into each backend's
wire format. This module touches no keys and no network; it is a pure function of
``(config, counts, seed)``. Same seed → identical manifest (reproducibility is the
deliverable, so the whole dataset can be rebuilt from the config + a seed).

Sampling: we index the axis cross-product as a mixed-radix number and draw ``n`` DISTINCT
indices with a seeded RNG, so no two prompts share an axis combo until the space is
exhausted (only past that do we allow repeats: different temp/seed/grounding at
generation time still diverges the output).
"""
import random

import yaml

# --- §7.9 transport wrapper (prose mode) -------------------------------------
# Sentinels are collision-proof ASCII tags that never occur in organic Serbian.
# The converter (step 4) extracts everything between them and strips the markers
# before tokenization; a hard open+close pair lets it drop any model preamble/chatter.
DOC_OPEN = "<<<DOC>>>"
DOC_CLOSE = "<<<END>>>"
TRANSPORT_INSTRUCTION = (
    f"Begin your output with a line containing only {DOC_OPEN} and end it with a line "
    f"containing only {DOC_CLOSE}. Write the requested text between those two markers "
    f"and nothing outside them."
)

# Canonical axis order, fixed so index→combo decoding is stable across runs/machines.
CONV_AXIS_ORDER = [
    "voice", "region_lean", "emotional_state",
    "platform", "topic", "sloppiness", "output_shape",
]
PARTICLE_K = 2                                            # particles sampled per prompt into {particles}
# fallback if the config omits the pool (keeps older configs working)
DEFAULT_PARTICLES = ["pa", "ma", "bre", "ej", "hej", "ajde", "jao", "ono", "kao", "vala"]
# grounding rate fallback: essay carries the measured homogenization (gate:
# Flash-Lite 1.83x / MiMo 1.51x real self-BLEU-4); conv OFF until a true chat/forum
# reference pool exists (subtitle anchors would drag chat toward dubbed dialogue).
DEFAULT_GROUNDING_RATE = {"conversational": 0.0, "essay": 0.75}
ESSAY_AXIS_ORDER = [
    "essay_type", "author_voice", "topic_domain", "length", "thesis", "density",
]
REG_PREFIX = {"conversational": "conv", "essay": "essay"}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- axis assembly -----------------------------------------------------------
def conversational_axes(cfg):
    """Ordered [(name, values)]. output_shape lives in its own top-level list."""
    src = cfg["conversational_axes"]
    axes = [(k, list(src[k])) for k in CONV_AXIS_ORDER if k != "output_shape"]
    axes.append(("output_shape", list(cfg["conversational_output_shapes"])))
    return axes


def essay_axes(cfg):
    src = cfg["essay_axes"]
    return [(k, list(src[k])) for k in ESSAY_AXIS_ORDER]


def axes_for(cfg, register):
    return conversational_axes(cfg) if register == "conversational" else essay_axes(cfg)


def space_size(axes):
    n = 1
    for _, values in axes:
        n *= len(values)
    return n


def decode(index, axes):
    """Mixed-radix decode: integer index -> {axis: value} in canonical order.

    The last axis varies fastest (odometer). Presented in canonical order for readability.
    """
    picked = {}
    for name, values in reversed(axes):
        index, r = divmod(index, len(values))
        picked[name] = values[r]
    return {name: picked[name] for name, _ in axes}


def sample_indices(axes, n, seed):
    """n DISTINCT combo indices (shuffled) while n <= space; repeats only past that."""
    total = space_size(axes)
    rng = random.Random(seed)
    if n <= total:
        return rng.sample(range(total), n)          # distinct, uniform, shuffled
    idx = list(range(total))                         # exhaust the space first...
    rng.shuffle(idx)
    idx += [rng.randrange(total) for _ in range(n - total)]   # ...then allow repeats
    return idx


# --- template filling --------------------------------------------------------
def particles_for(cfg, seed, seq):
    """A deterministic rotating subset of discourse particles for prompt #seq.

    Seeded by (seed, seq) so the manifest stays reproducible while each prompt primes
    different particles, so no single one dominates the corpus (the 'ej' fix)."""
    pool = cfg.get("conversational_particles") or DEFAULT_PARTICLES
    rng = random.Random(seed * 1_000_003 + seq)
    return ", ".join(rng.sample(pool, min(PARTICLE_K, len(pool))))


def _fill_conversational(cfg, combo, particles):
    t = cfg["templates"]
    return t["conversational_base"].format(
        voice=combo["voice"],
        region_lean=combo["region_lean"],
        emotional_state=combo["emotional_state"],
        platform=combo["platform"],
        topic=combo["topic"],
        sloppiness=combo["sloppiness"],
        particles=particles,
        output_shape_instruction=t["output_shape_instructions"][combo["output_shape"]],
    )


def _fill_essay(cfg, combo):
    t = cfg["templates"]
    return t["essay_base"].format(
        length=combo["length"],
        essay_type=combo["essay_type"],
        author_voice=combo["author_voice"],
        topic_domain=combo["topic_domain"],
        density=combo["density"],
        thesis_instruction=t["thesis_instructions"][combo["thesis"]],
    )


def build_prompt(body):
    """Register body + the §7.9 transport wrapper as one ready-to-send string."""
    return body.rstrip() + "\n\n" + TRANSPORT_INSTRUCTION


# --- grounding (style anchors) -------------------------------------------------
def grounding_rate(cfg, register):
    rates = (cfg.get("grounding") or {}).get("rate") or {}
    return rates.get(register, DEFAULT_GROUNDING_RATE[register])


def anchor_for(cfg, register, seed, seq, refpool):
    """(anchor_record | None): seeded per-record grounding decision + anchor draw.

    Offset RNG stream so adding grounding does not perturb particle sampling for the
    same (seed, seq). Nuance: at seed=0 the multipliers vanish,
    so this stream equals the particle stream shifted by one: deterministic and fine
    (different records, different consumption patterns), but not truly independent.
    Do NOT change the seeding scheme casually: manifests v2 and every artifact generated
    from them reproduce only under THIS scheme. A change = a manifest-version event
    (archive + regen + STATUS note)."""
    rate = grounding_rate(cfg, register)
    if rate <= 0 or not refpool:
        return None
    rng = random.Random(seed * 2_000_003 + seq + 1)
    if rng.random() >= rate:
        return None
    return refpool[rng.randrange(len(refpool))]


def grounding_block(cfg, anchor, topic):
    """The config's style_transfer text with {reference}/{topic} filled.

    .replace, NOT .format: real anchor text may contain braces and must land verbatim."""
    tmpl = cfg["grounding"]["modes"]["style_transfer"]
    return tmpl.replace("{reference}", anchor["text"]).replace("{topic}", topic).rstrip()


def make_record(cfg, register, combo, seq, seed=0, refpool=None):
    if register == "conversational":
        particles = particles_for(cfg, seed, seq)
        body = _fill_conversational(cfg, combo, particles)
        axes = {**combo, "particles": particles}         # record what was primed (provenance)
        topic = combo["topic"]
    else:
        body = _fill_essay(cfg, combo)
        axes = dict(combo)
        topic = combo["topic_domain"]
    anchor = anchor_for(cfg, register, seed, seq, refpool)
    if anchor is not None:
        body = body.rstrip() + "\n\n" + grounding_block(cfg, anchor, topic)
        axes["anchor"] = anchor["id"]                    # grounding provenance
    return {
        "id": f"{REG_PREFIX[register]}-{seq:07d}",
        "register": register,
        "axes": axes,
        "prompt": build_prompt(body),
    }


def iter_records(cfg, register, n, seed, refpool=None):
    """Yield n prompt-specs for a register, streaming (no full-list buffering of records)."""
    axes = axes_for(cfg, register)
    for seq, index in enumerate(sample_indices(axes, n, seed)):
        yield make_record(cfg, register, decode(index, axes), seq, seed, refpool)
