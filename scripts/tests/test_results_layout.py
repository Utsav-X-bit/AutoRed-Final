import hashlib
from experiment.results_layout import slugify_model_id, resolve_model_id


def test_slugify_hf_id_replaces_slash():
    assert slugify_model_id("meta-llama/Meta-Llama-3-8B-Instruct") == "meta-llama--Meta-Llama-3-8B-Instruct"


def test_slugify_bare_name_unchanged():
    assert slugify_model_id("gpt2") == "gpt2"


def test_slugify_strips_surrounding_separators():
    assert slugify_model_id("/org/name/") == "org--name"


def test_slugify_unsafe_chars_replaced():
    out = slugify_model_id("org/name with space")
    assert " " not in out
    assert "/" not in out


def test_resolve_prefers_explicit_flag():
    assert resolve_model_id("meta-llama/Meta-Llama-3-8B-Instruct", None) == "meta-llama--Meta-Llama-3-8B-Instruct"


def test_resolve_parses_hf_cache_dir():
    p = "/home/user/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/abc"
    assert resolve_model_id(None, p) == "meta-llama--Meta-Llama-3-8B-Instruct"


def test_resolve_local_path_basename_plus_hash():
    p = "/nlsasfs/home/isea/isea38/AutoRed-Final/experiment/results/planner_sft_v2/checkpoint-27"
    expected_hash = hashlib.sha256(p.encode()).hexdigest()[:6]
    assert resolve_model_id(None, p) == f"checkpoint-27_{expected_hash}"


def test_resolve_fallback_unknown():
    assert resolve_model_id(None, None) == "unknown"
