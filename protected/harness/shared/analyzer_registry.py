"""Neutral registry for the shared AttentionAnalyzer instance.

Avoids __main__ vs package name module identity issues when the study runner
is launched via python -m."""

_instance = None


def set_analyzer(instance):
    global _instance
    _instance = instance


def get_analyzer():
    return _instance
