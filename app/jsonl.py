"""Utilitaires JSONL (streaming).

Repris de Chat-Doc-Collection-Qdrant : le pipeline lit et ecrit du JSONL
plutot que du JSON monolithique parce que :
- Les fichiers peuvent etre gros (1 GB pour Eduscol, ~2 MB pour la doc forge).
- Le streaming permet de traiter au fil de l'eau sans exploser la RAM.
- Les lignes corrompues sont ignorees sans casser tout le fichier.
- Diff/grep/wc marchent en ligne de commande.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Itere sur les lignes valides. Ignore les lignes vides ou corrompues."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Ajoute une ligne au fichier (le cree si absent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Reecrit le fichier avec les lignes fournies. Retourne le compte."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def count_jsonl(path: Path) -> int:
    """Compte les lignes valides."""
    if not path.is_file():
        return 0
    return sum(1 for _ in read_jsonl(path))
