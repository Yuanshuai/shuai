import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _to_int(v: Optional[str], default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _to_float(v: Optional[str], default: float) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _to_bool(v: Optional[str], default: bool) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _default_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path(os.getcwd()).resolve()


def _runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        try:
            return Path(sys.executable).resolve().parent
        except Exception:
            return Path(os.getcwd()).resolve()
    return Path(os.getcwd()).resolve()


def load_env() -> None:
    paths = []
    runtime = _runtime_dir()
    paths.append(runtime / ".env")
    paths.append(runtime / "_internal" / ".env")
    if hasattr(sys, "_MEIPASS"):
        try:
            paths.append(Path(sys._MEIPASS) / ".env")
        except Exception:
            pass
    paths.append(_default_root() / ".env")
    for p in paths:
        if p.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(str(p), override=False)
                return
            except Exception:
                _load_env_fallback(p)
                return


def _load_env_fallback(path: Path) -> None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        return


@dataclass(frozen=True)
class Settings:
    project_root: Path
    runtime_dir: Path

    pdf_dpi: int
    lin_port: int

    table_exclude_pad_ratio: float
    table_exclude_pad_px: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        project_root = _default_root()
        runtime_dir = _runtime_dir()

        return cls(
            project_root=project_root,
            runtime_dir=runtime_dir,
            pdf_dpi=_to_int(os.getenv("PDF_DPI"), 150),
            lin_port=_to_int(os.getenv("LIN_PORT"), 8000),
            table_exclude_pad_ratio=_to_float(os.getenv("TABLE_EXCLUDE_PAD_RATIO"), 0.03),
            table_exclude_pad_px=_to_int(os.getenv("TABLE_EXCLUDE_PAD_PX"), 8),
        )
