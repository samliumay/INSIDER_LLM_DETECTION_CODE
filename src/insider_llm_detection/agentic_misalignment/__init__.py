"""The three harm classifiers of Anthropic's *Agentic Misalignment* framework, bundled.

Copied unchanged from https://github.com/anthropic-experimental/agentic-misalignment
(`classifiers/`, commit SOURCE_COMMIT, MIT — see LICENSE next to this file). The only
edit is the import of `ChatMessage` / `MessageRole`, which now comes from the small
`model_client.py` shim in this package instead of the framework's API client, so the
classifiers run without that client or its provider SDKs. The judge model is supplied by
the caller (`classify.GeminiJudge`).
"""
from .classifier import Classifier
from .blackmail_classifier import BlackmailClassifier
from .leak_classifier import LeakClassifier
from .murder_classifier import MurderClassifier

SOURCE_REPO = "anthropic-experimental/agentic-misalignment"
SOURCE_COMMIT = "ea0630e1a3eaae7f9f9740fd2703229d3854ccda"

__all__ = ["Classifier", "BlackmailClassifier", "LeakClassifier", "MurderClassifier",
           "SOURCE_REPO", "SOURCE_COMMIT"]
