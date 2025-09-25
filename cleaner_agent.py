"""Cleaner Agent

This module provides a lightweight 'agent' abstraction focused on ensuring
that the data pipeline (data_pipeline.py) can be invoked programmatically
from other parts of the system (e.g., a chat interface) to (re)generate
processed datasets.

It does NOT depend on LLM functionality directly; instead it exposes a
callable interface returning structured status + a human-readable summary.
"""
from __future__ import annotations
import traceback
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import importlib

@dataclass
class CleanerResult:
    ok: bool
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self):
        d = asdict(self)
        return d

class CleanerAgent:
    PIPELINE_MODULE = "data_pipeline"

    def __init__(self):
        self._ensure_pipeline()

    def _ensure_pipeline(self):
        try:
            importlib.import_module(self.PIPELINE_MODULE)
        except Exception as e:
            raise RuntimeError(f"Pipeline module '{self.PIPELINE_MODULE}' not available: {e}")

    def run(self) -> CleanerResult:
        try:
            pipeline = importlib.import_module(self.PIPELINE_MODULE)
            if not hasattr(pipeline, "run_pipeline"):
                return CleanerResult(False, "run_pipeline() not found in data_pipeline module")
            pipeline.run_pipeline()
            return CleanerResult(True, "Data pipeline executed successfully.")
        except Exception:
            return CleanerResult(False, "Pipeline execution failed", {"traceback": traceback.format_exc()})

# Convenience function (could be registered as a tool for a future LLM agent)

def run_data_cleaning() -> Dict[str, Any]:
    agent = CleanerAgent()
    result = agent.run()
    return result.to_dict()

if __name__ == "__main__":
    print(run_data_cleaning())
