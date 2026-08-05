from reasonforge.inference import generation_finish_metadata


def test_generation_finish_metadata_uses_token_evidence() -> None:
    assert generation_finish_metadata([1, 2], max_new_tokens=4, eos_token_ids={2}) == (
        False,
        "eos_token",
    )
    assert generation_finish_metadata([1, 1, 1, 1], max_new_tokens=4, eos_token_ids={2}) == (
        True,
        "max_new_tokens",
    )
    assert generation_finish_metadata([1], max_new_tokens=4, eos_token_ids={2}) == (
        False,
        "generation_stopped",
    )
