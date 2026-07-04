"""Segmentation Markdown avec conscience des sections H1/H2/H3.

Basé sur langchain-text-splitters :
- MarkdownHeaderTextSplitter : découpe par hiérarchie de titres, garde le
  chemin de section en metadata automatiquement.
- RecursiveCharacterTextSplitter : sous-découpe chaque section trop grande
  en respectant paragraphes puis phrases.

Chaque segment produit inclut :
- Un préfixe contextuel (Titre / Section / Source) optionnel qui aide
  l'embedding à situer le contenu même quand la question ne mentionne pas
  la page.
- Une metadata riche : section_path, section_anchor, source_url avec ancre,
  chunk_index, page_title.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]


def slugify(text: str) -> str:
    """Slug 11ty-style : minuscule, sans accents, tirets."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def build_section_path(header_meta: dict[str, str]) -> str:
    """Construit 'H1 > H2 > H3' à partir de la metadata d'un split header."""
    parts = [header_meta[key] for key in ("H1", "H2", "H3") if key in header_meta]
    return " > ".join(parts)


def build_prefix(page_title: str, section_path: str, source_url: str) -> str:
    """Construit le préfixe contextuel qu'on colle en tête de chaque segment.

    Format volontairement lisible : le préfixe apparaîtra aussi bien dans
    l'embedding (utile pour la retrieval) que dans les extraits affichés
    par le chat (utile pour l'utilisateur).
    """
    lines = [f"Titre : {page_title}"]
    if section_path and section_path != page_title:
        lines.append(f"Section : {section_path}")
    if source_url:
        lines.append(f"Source : {source_url}")
    lines.append("")
    lines.append("Contenu :")
    return "\n".join(lines) + "\n"


def make_source_url_with_anchor(base_url: str, anchor: str) -> str:
    """Ajoute une ancre à l'URL si présente. Base URL supposée déjà propre."""
    if not base_url:
        return ""
    if not anchor:
        return base_url
    return f"{base_url}#{anchor}"


def segment_markdown(
    body: str,
    *,
    page_title: str,
    source_url: str = "",
    rel_path: str = "",
    breadcrumb_parent: str | None = None,
    target_chars: int = 1600,
    overlap: int = 200,
    min_chunk_chars: int = 250,
    include_prefix: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Segmente un body markdown déjà prétraité.

    Retourne une liste de segments prêts à être uploadés (format Albert
    /v1/documents/{id}/chunks, mais générique : dict content+metadata).
    """
    if not body.strip():
        return []

    effective_page_title = (
        f"{breadcrumb_parent} > {page_title}"
        if breadcrumb_parent and breadcrumb_parent.strip().lower() != page_title.strip().lower()
        else page_title
    )

    # 1. Split par hiérarchie H1/H2/H3 : chaque bloc reste sémantiquement
    #    cohérent, on ne coupe pas au milieu d'une explication.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # on garde le "# Titre" dans le contenu
    )
    header_chunks = header_splitter.split_text(body)

    # 2. Sous-découpage : si un bloc est encore trop grand, on le coupe par
    #    paragraphes puis phrases.
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        length_function=len,
    )

    segments: list[dict[str, Any]] = []
    global_index = 0

    for header_chunk in header_chunks:
        section_path = build_section_path(header_chunk.metadata)
        # Le dernier heading (le plus profond) donne l'ancre
        deepest_heading = None
        for key in ("H3", "H2", "H1"):
            if key in header_chunk.metadata:
                deepest_heading = header_chunk.metadata[key]
                break
        anchor = slugify(deepest_heading) if deepest_heading else ""
        segment_url = make_source_url_with_anchor(source_url, anchor)

        pieces = size_splitter.split_text(header_chunk.page_content)
        for i, piece in enumerate(pieces):
            piece_clean = piece.strip()
            if not piece_clean:
                continue
            # Fusion des micro-pieces dans la même section
            if (
                len(piece_clean) < min_chunk_chars
                and segments
                and segments[-1]["metadata"].get("section_anchor") == anchor
                and i > 0
            ):
                segments[-1]["content"] = _append_body(segments[-1]["content"], piece_clean)
                continue

            content = (
                build_prefix(effective_page_title, section_path, segment_url) + piece_clean
                if include_prefix
                else piece_clean
            )
            metadata = {
                "page_title": page_title,
                "section_path": section_path or page_title,
                "section_anchor": anchor,
                "source_url": segment_url,
                "rel_path": rel_path,
                "chunk_index": global_index,
            }
            if extra_metadata:
                metadata.update({k: v for k, v in extra_metadata.items() if k not in metadata})
            segments.append({"content": content, "metadata": metadata})
            global_index += 1

    return segments


def _append_body(existing_content: str, additional: str) -> str:
    """Ajoute du texte au corps d'un segment déjà préfixé.

    On trouve la ligne 'Contenu :' et on append en dessous. Si pas trouvé
    (préfixe désactivé), on concatène directement.
    """
    marker = "\nContenu :\n"
    idx = existing_content.find(marker)
    if idx == -1:
        return existing_content + "\n\n" + additional
    return existing_content + "\n\n" + additional


def estimate_segment_count(
    body: str,
    *,
    target_chars: int = 1600,
    overlap: int = 200,
) -> int:
    """Estime le nombre de segments sans les matérialiser (utile pour un preview)."""
    if not body.strip():
        return 0
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_chars,
        chunk_overlap=overlap,
    )
    count = 0
    for header_chunk in header_splitter.split_text(body):
        count += len(size_splitter.split_text(header_chunk.page_content))
    return max(1, count)
