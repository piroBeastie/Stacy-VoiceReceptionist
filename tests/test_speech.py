"""Text that must never reach the voice.

Synthesised on eleven_flash_v2_5 and transcribed back, "[slow] Fourteen Elm
Street" came out as "Slow 14 Elm Street", and "[chuckles] Nope" as "Chuckles.
Nope." The prompt asks for [slow] on every phone number read-back, so every
caller heard the word "slow" in front of their own number.
"""

from __future__ import annotations

import asyncio

import pytest

from stacy.agent.speech import strip_audio_tags, strip_audio_tags_stream


def _streamed(chunks: list[str]) -> str:
    async def source():
        for chunk in chunks:
            yield chunk

    async def collect() -> str:
        return "".join([part async for part in strip_audio_tags_stream(source())])

    return asyncio.run(collect())


class TestWholeStrings:
    @pytest.mark.parametrize(
        "raw, spoken",
        [
            ("[slow] two-zero-three", "two-zero-three"),
            ("[chuckles] Nope, I'm the assistant here.", "Nope, I'm the assistant here."),
            ("Okay. [sighs] Let's get someone out.", "Okay. Let's get someone out."),
            ("[curious] What's it doing?", "What's it doing?"),
        ],
    )
    def test_tags_are_removed(self, raw: str, spoken: str) -> None:
        assert strip_audio_tags(raw).strip() == spoken

    def test_ordinary_brackets_are_left_alone(self) -> None:
        # Only word cues are removed; a "[1]" is not one.
        assert strip_audio_tags("item [1] is ready") == "item [1] is ready"


class TestStreamedText:
    def test_a_tag_split_across_chunks_is_still_removed(self) -> None:
        # Model output arrives a few characters at a time.
        out = _streamed(["Sure, ", "[sl", "ow] two-zero", "-three"])
        assert "slow" not in out.lower()
        assert "two-zero-three" in out

    def test_text_without_tags_passes_through_unchanged(self) -> None:
        assert _streamed(["We're open ", "eight to five."]) == "We're open eight to five."

    def test_a_tag_at_the_very_end_is_removed(self) -> None:
        out = _streamed(["Got it ", "[chuck", "les]"])
        assert "chuckles" not in out.lower()
        assert out.strip() == "Got it"

    def test_a_lone_bracket_is_released_rather_than_stalling_speech(self) -> None:
        # Held back only until it cannot be a tag. It must still be spoken.
        text = "[" + "x" * 40
        assert _streamed([text[:10], text[10:]]) == text
