"""
Production inference parameters for dots.mocr OCR pipeline.
Sourced from nm-dots-ocr-service production config.
"""

# Sampling parameters
TEMPERATURE = 0.1
TOP_P = 0.9
FREQUENCY_PENALTY = 0.03
PRESENCE_PENALTY = 0.0
MAX_COMPLETION_TOKENS = 8192

# Logit bias: suppress garbage/filler tokens that cause repetitive output
# These tokens get -8 bias (effectively suppressed)
LOGIT_BIAS = {
    42269: -8,   # dots (.)
    3513: -8,    # dashes (-)
    46932: -8,   # underscore (_)
    4077: -8,    # asterisk (*)
    8152: -8,    # equals (=)
    9822: -8,    # slash (/)
    13067: -8,   # hash (#)
    79518: -8,   # percent (%)
    68700: -8,   # dollar ($I)
    11304: -8,
    1177: -8,
    481: -8,
}

# Image processing
PDF_DPI = 200
IMAGE_MIN_PIXELS = 3136
IMAGE_MAX_PIXELS = 11289600  # Native resolution — no downscaling

# EOS token IDs for dots.mocr (Qwen2 tokenizer)
EOS_TOKEN_IDS = {151643, 151645}
