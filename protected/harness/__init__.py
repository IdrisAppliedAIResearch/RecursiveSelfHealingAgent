# Shared modules re-exported for Study 001 backward compatibility.
# Study 002 modules import from protected.harness.shared.* directly.

from protected.harness.shared import (
    allowlist,
    anomaly_logger,
    artifact_writer,
    corpus_runner,
    edit_applier,
    edit_protocol,
    episode_store,
    git_ops,
    interface_validator,
    model_performance,
)

# Study 001 agent_caller re-exported for backward compatibility.
from protected.harness.study_001 import agent_caller
