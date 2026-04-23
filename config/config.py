from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    cwd: str = str(Path.cwd())
    developer_instructions: str | None = None
    user_instructions: str | None = None
