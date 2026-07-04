"""Point d'entree en ligne de commande du segmenteur.

Utilise dans GitHub Actions (voir .github/workflows/segment.yml) :

    python -m app.cli \\
        --source-dir src/target \\
        --content-dir content \\
        --site-base-url https://docs.forge.apps.education.fr \\
        --output segments.jsonl \\
        --report report.json

Produit deux fichiers :

- `segments.jsonl` : un segment par ligne, format JSONL streaming.
  Format aligne sur celui de Chat-Doc-Collection-Qdrant/data/chunks.jsonl,
  avec les cles chunk_id, document_id, chunk_index, text, content_hash, etc.
  Le PHP consommateur regroupe par document_id, cree un document Albert et
  envoie les chunks par lots de 64.

- `report.json` : synthese chiffree + liste des fichiers exclus avec raison.
  Le PHP l'utilise pour afficher un rapport et un compteur d'erreurs.

Prerequis : aucun secret. Le CLI ne parle a aucun service externe. Tout
est fait localement dans le runner CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .jsonl import append_jsonl
from .preprocessing import (
    get_breadcrumb_parent,
    is_not_published,
    preprocess,
    synthesize_body_from_meta,
    title_from_body,
)
from .segmenting import segment_markdown


logger = logging.getLogger("segmenteur.cli")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Constantes et helpers
# ---------------------------------------------------------------------------

EXCLUDED_DIRS = {"_includes", "_data", "_site", "node_modules", ".git", ".github"}


def sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def detect_content_dir(source_dir: Path, override: str = "") -> str:
    """Auto-detecte le dossier contenu : override -> content -> src -> ."""
    if override:
        if (source_dir / override).is_dir():
            return override
        # Si override donne mais absent : erreur explicite
        raise FileNotFoundError(
            f"content_dir='{override}' n'existe pas dans {source_dir}"
        )
    for candidate in ("content", "src"):
        if (source_dir / candidate).is_dir():
            return candidate
    return ""  # racine


def collect_excluded_by_11tydata(base: Path) -> set[Path]:
    """Repere les dossiers desactives par un .11tydata.js avec permalink: false."""
    excluded = set()
    for data_file in base.rglob("*.11tydata.js"):
        try:
            text = data_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "permalink" in text and "false" in text:
            # Regex minimaliste pour eviter les faux positifs
            import re
            if re.search(r"permalink\s*:\s*false", text, re.IGNORECASE):
                excluded.add(data_file.parent)
    return excluded


def derive_source_url(
    rel_path: str,
    meta: dict[str, Any],
    content_dir: str,
    site_base_url: str,
) -> str:
    """Construit l'URL publique d'une page a partir de son chemin.

    - Si permalink dans le front matter (string, http ou chemin relatif) : utilise
    - Sinon : content_dir/foo/bar.md -> {site_base_url}/foo/bar/
      Cas special index.md -> dossier parent, sans /index/ dans l'URL
    """
    if not site_base_url:
        return ""
    site_base_url = site_base_url.rstrip("/")

    permalink = meta.get("permalink")
    if isinstance(permalink, str) and permalink.strip() and permalink.strip().lower() != "false":
        pl = permalink.strip()
        if pl.startswith(("http://", "https://")):
            return pl
        return f"{site_base_url}/{pl.lstrip('/')}"

    path = rel_path
    if content_dir:
        prefix = content_dir + "/"
        if path.startswith(prefix):
            path = path[len(prefix):]
    # Retire extension .md/.markdown
    for suffix in (".md", ".markdown"):
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)]
            break
    # index -> ''
    if path.endswith("/index"):
        path = path[:-len("/index")]
    elif path == "index":
        path = ""

    path = path.strip("/")
    if not path:
        return f"{site_base_url}/"
    return f"{site_base_url}/{path}/"


def build_document_id(rel_path: str) -> str:
    """ID de document stable pour idempotence entre runs."""
    return sha1(rel_path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_file(
    md_path: Path,
    source_dir: Path,
    content_dir: str,
    site_base_url: str,
    target_chars: int,
    overlap: int,
    include_prefix: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Traite un seul .md et retourne (document_info, segments).

    document_info contient les meta + statut (excluded ou non).
    segments est vide si document exclu.
    """
    rel_path = md_path.relative_to(source_dir).as_posix()
    document_id = build_document_id(rel_path)

    try:
        raw_content = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return (
            {
                "document_id": document_id,
                "rel_path": rel_path,
                "title": md_path.stem,
                "source_url": "",
                "excluded": True,
                "excluded_reason": f"Lecture impossible : {exc}",
                "segments_count": 0,
            },
            [],
        )

    meta, body = preprocess(raw_content)

    # Titre : front matter > premier heading > nom de fichier
    title = (
        str(meta.get("title")) if meta.get("title") else None
    ) or title_from_body(body) or md_path.stem

    source_url = derive_source_url(rel_path, meta, content_dir, site_base_url)
    breadcrumb_parent = get_breadcrumb_parent(meta)

    # Exclusion 1 : permalink: false
    if is_not_published(meta):
        return (
            {
                "document_id": document_id,
                "rel_path": rel_path,
                "title": title,
                "source_url": "",
                "excluded": True,
                "excluded_reason": "permalink: false (page non publiee)",
                "segments_count": 0,
            },
            [],
        )

    # Cas synthese : body vide, mais front matter riche (cartes catalogues)
    synthesized = False
    if len(body.strip()) < 40:
        synth = synthesize_body_from_meta(title, meta)
        if synth:
            _, body = preprocess(
                synth,
                strip_11ty_templates=False,
                preserve_image_alt_texts=False,
                convert_container_blocks=False,
            )
            synthesized = True

    # Exclusion 2 : toujours vide apres synthese
    if len(body.strip()) < 40:
        return (
            {
                "document_id": document_id,
                "rel_path": rel_path,
                "title": title,
                "source_url": source_url,
                "excluded": True,
                "excluded_reason": "Body vide et front matter inexploitable",
                "segments_count": 0,
            },
            [],
        )

    # Segmentation
    raw_segments = segment_markdown(
        body,
        page_title=title,
        source_url=source_url,
        rel_path=rel_path,
        breadcrumb_parent=breadcrumb_parent,
        target_chars=target_chars,
        overlap=overlap,
        include_prefix=include_prefix,
        extra_metadata={
            "source_type": "markdown",
            "synthesized": synthesized,
        },
    )

    # Enrichissement : chunk_id + content_hash + text separe (pattern Qdrant)
    segments = []
    for idx, seg in enumerate(raw_segments):
        content = seg["content"]
        content_hash = sha256(content)
        chunk_id = sha1(f"{document_id}:{idx}:{content_hash}")
        segments.append({
            "kind": "segment",
            "document_id": document_id,
            "chunk_id": chunk_id,
            "chunk_index": idx,
            "content": content,
            "metadata": seg["metadata"],
            "content_hash": content_hash,
            "chars": len(content),
        })

    return (
        {
            "document_id": document_id,
            "rel_path": rel_path,
            "title": title,
            "source_url": source_url,
            "breadcrumb_parent": breadcrumb_parent,
            "excluded": False,
            "excluded_reason": None,
            "synthesized": synthesized,
            "segments_count": len(segments),
        },
        segments,
    )


def run(
    source_dir: Path,
    output_path: Path,
    report_path: Path,
    content_dir_override: str,
    site_base_url: str,
    target_chars: int,
    overlap: int,
    include_prefix: bool,
) -> dict[str, Any]:
    """Execute le pipeline complet."""
    started = time.monotonic()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source introuvable : {source_dir}")

    content_dir = detect_content_dir(source_dir, content_dir_override)
    base = source_dir / content_dir if content_dir else source_dir
    logger.info("Source : %s", source_dir)
    logger.info("Content dir detecte : %s", content_dir or "(racine)")
    logger.info("Site base URL : %s", site_base_url or "(vide, pas d'URLs)")

    # 11tydata.js -> dossiers exclus
    excluded_dirs = collect_excluded_by_11tydata(base)
    if excluded_dirs:
        logger.info(
            "Dossiers exclus par .11tydata.js (permalink: false) : %d",
            len(excluded_dirs),
        )

    # Reset output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    # Statistiques
    stats = {
        "files_scanned": 0,
        "documents_created": 0,
        "documents_excluded": 0,
        "documents_synthesized": 0,
        "segments_total": 0,
        "errors": 0,
    }
    documents_index: list[dict[str, Any]] = []
    excluded_list: list[dict[str, Any]] = []

    # Parcours des .md
    md_files = sorted(base.rglob("*.md"))
    for md_path in md_files:
        # Skip dossiers 11ty internes
        if any(part in EXCLUDED_DIRS for part in md_path.parts):
            continue
        # Skip dossiers desactives par 11tydata
        if any(md_path.is_relative_to(d) for d in excluded_dirs):
            continue

        stats["files_scanned"] += 1
        try:
            doc_info, segments = process_file(
                md_path,
                source_dir,
                content_dir,
                site_base_url,
                target_chars,
                overlap,
                include_prefix,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur sur %s", md_path)
            stats["errors"] += 1
            continue

        # Ecrit d'abord la ligne "document" (marker de debut de doc)
        doc_line = {"kind": "document", **doc_info}
        append_jsonl(output_path, doc_line)
        documents_index.append(doc_info)

        if doc_info["excluded"]:
            stats["documents_excluded"] += 1
            excluded_list.append({
                "rel_path": doc_info["rel_path"],
                "title": doc_info["title"],
                "reason": doc_info["excluded_reason"],
            })
            continue

        if doc_info.get("synthesized"):
            stats["documents_synthesized"] += 1

        # Puis chaque segment
        for seg in segments:
            append_jsonl(output_path, seg)
        stats["documents_created"] += 1
        stats["segments_total"] += len(segments)

    duration = time.monotonic() - started
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_seconds": round(duration, 2),
        "source": {
            "path": str(source_dir),
            "content_dir": content_dir,
            "site_base_url": site_base_url,
        },
        "params": {
            "target_chars": target_chars,
            "overlap": overlap,
            "include_prefix": include_prefix,
        },
        "stats": stats,
        "excluded": excluded_list,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Termine en %.2fs : %d fichiers, %d documents, %d segments, %d exclus, %d erreurs",
        duration,
        stats["files_scanned"],
        stats["documents_created"],
        stats["segments_total"],
        stats["documents_excluded"],
        stats["errors"],
    )

    return report


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="segmenteur",
        description="Segmente un dossier de fichiers Markdown pour ingestion RAG.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Racine du dossier a analyser (repo clone).",
    )
    parser.add_argument(
        "--content-dir",
        default="",
        help="Sous-dossier des .md (auto-detecte : content -> src -> racine).",
    )
    parser.add_argument(
        "--site-base-url",
        default="",
        help="URL publique du site pour construire source_url avec ancres.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("segments.jsonl"),
        help="Fichier JSONL de sortie (segments + documents).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("report.json"),
        help="Rapport JSON de synthese.",
    )
    parser.add_argument("--target-chars", type=int, default=1600)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Desactive le prefixe 'Titre / Section / Source' dans chaque segment.",
    )
    args = parser.parse_args(argv)

    try:
        run(
            source_dir=args.source_dir,
            output_path=args.output,
            report_path=args.report,
            content_dir_override=args.content_dir,
            site_base_url=args.site_base_url,
            target_chars=args.target_chars,
            overlap=args.overlap,
            include_prefix=not args.no_prefix,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Echec du pipeline")
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
