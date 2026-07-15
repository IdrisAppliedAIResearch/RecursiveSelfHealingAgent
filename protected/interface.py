import os

ENTRY_POINT_MODULE = "playground.extractor"
ENTRY_POINT_FUNCTION = "extract"
ENTRY_POINT_PARAMS = 2
# A009-3: default 30 min per the pre-registration (line 193), configurable via env.
# (Was 14400/4h — an undocumented deviation.) The provider already caps each generation
# at EXTRACTION_MAX_NEW_TOKENS, so this bounds a slow *multi-call* extractor.
ITERATION_TIMEOUT_S = int(os.environ.get("STUDY_ITERATION_TIMEOUT_S", 1800))
