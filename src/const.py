"""A module containing constants used in different files.

Constants that only appear in one place are usually left as a raw number. They are seen as reasonable
defaults and having them directly at the place of their usage makes the code more readable."""

DEFAULT_FS = 8000  # sampling rate used as the default in many functions
SAMPLE_VALUES = 256  # how many different values can the samples have (from 0 to SAMPLE_VALUES-1)
