class DiarizationConfig:

    # For prompt segmentation.
    # max prompt length (chars)
    EMIT_INPUT_LENGTH = 3000 # 1000 6000
    EMIT_TARGET_LENGTH = 3000 # 1000 6000

    # Prefix and suffix for prompt and completion.
    # As a reference, OpenAI finetuning API usually suggests:
    # - No prompt prefix
    # - Prompt suffix: " -> "
    # - Completion suffix: " END"
    PROMPT_PREFIX = ""
    PROMPT_SUFFIX = " --> "
    COMPLETION_SUFFIX = " [eod]"

    # How do we represent the speaker token.
    speaker_prefix = "<speaker:"
    speaker_suffix = ">"
