"""
utils/constants.py — Shared constants for the NSFW scanner.
"""

SEVERITY = {
    "SAFE": 0,
    "SUGGESTIVE": 1,
    "REVIEW": 1,
    "NSFW": 2,
    "BLOCK": 2,
    "EXPLICIT": 3,
}

BLOCKING_VERDICTS = {"BLOCK", "NSFW", "EXPLICIT"}
REVIEW_VERDICTS = {"REVIEW", "SUGGESTIVE"}
