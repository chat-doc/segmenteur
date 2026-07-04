"""Tests unitaires de la segmentation."""
from __future__ import annotations

from app.segmenting import (
    build_prefix,
    build_section_path,
    estimate_segment_count,
    segment_markdown,
    slugify,
)


class TestSlugify:
    def test_removes_accents_and_normalizes(self):
        assert slugify("À propos de nous") == "a-propos-de-nous"
        assert slugify("Qu'est-ce qu'une forge ?") == "qu-est-ce-qu-une-forge"

    def test_handles_empty(self):
        assert slugify("") == ""


class TestBuildSectionPath:
    def test_three_levels(self):
        meta = {"H1": "Guide", "H2": "Install", "H3": "Windows"}
        assert build_section_path(meta) == "Guide > Install > Windows"

    def test_partial(self):
        assert build_section_path({"H1": "A", "H2": "B"}) == "A > B"


class TestBuildPrefix:
    def test_includes_all_fields(self):
        prefix = build_prefix("Titre", "Section > Sous", "https://x.fr#sous")
        assert "Titre : Titre" in prefix
        assert "Section : Section > Sous" in prefix
        assert "Source : https://x.fr#sous" in prefix
        assert prefix.endswith("Contenu :\n")

    def test_omits_source_when_empty(self):
        prefix = build_prefix("T", "S", "")
        assert "Source :" not in prefix


class TestSegmentMarkdown:
    def test_produces_at_least_one_segment(self):
        body = "# Titre\n\nUn paragraphe simple avec du contenu."
        segments = segment_markdown(body, page_title="T", source_url="https://x.fr/")
        assert len(segments) >= 1

    def test_splits_by_section(self):
        body = (
            "# Guide\n\nIntroduction générale au guide.\n\n"
            "## Prérequis\n\nAvant de commencer, Python 3.11 est nécessaire.\n\n"
            "## Étapes\n\nSuivre les étapes ci-dessous.\n"
        )
        segments = segment_markdown(
            body,
            page_title="Guide",
            source_url="https://x.fr/guide/",
            target_chars=200,
            min_chunk_chars=50,
        )
        section_paths = [s["metadata"]["section_path"] for s in segments]
        # On doit voir au moins une section Prérequis et une Étapes.
        assert any("Prérequis" in p for p in section_paths)
        assert any("Étapes" in p for p in section_paths)

    def test_anchor_from_deepest_heading(self):
        body = "## Section principale\n\nContenu.\n\n### Détail interne\n\nAutre contenu."
        segments = segment_markdown(body, page_title="Page", source_url="https://x.fr/p/")
        anchors = [s["metadata"]["section_anchor"] for s in segments]
        # Slugified :
        assert "section-principale" in anchors or "detail-interne" in anchors

    def test_prefix_is_included(self):
        body = "# Titre\n\nContenu."
        segments = segment_markdown(body, page_title="Titre", source_url="https://x.fr/")
        assert segments[0]["content"].startswith("Titre :")
        assert "Contenu :" in segments[0]["content"]

    def test_prefix_can_be_disabled(self):
        body = "# Titre\n\nContenu."
        segments = segment_markdown(
            body, page_title="Titre", source_url="https://x.fr/", include_prefix=False
        )
        assert not segments[0]["content"].startswith("Titre :")

    def test_breadcrumb_parent_enriches_title(self):
        body = "# Section\n\nContenu."
        segments = segment_markdown(
            body,
            page_title="Section",
            source_url="https://x.fr/",
            breadcrumb_parent="À propos",
        )
        assert "À propos > Section" in segments[0]["content"]

    def test_empty_body_returns_empty(self):
        assert segment_markdown("", page_title="X") == []

    def test_source_url_has_anchor(self):
        body = "## Prérequis\n\nDu contenu suffisamment long pour ne pas être fusionné."
        segments = segment_markdown(
            body,
            page_title="Page",
            source_url="https://x.fr/p/",
        )
        # Au moins un segment a l'ancre dans son metadata.source_url
        assert any("#prerequis" in s["metadata"]["source_url"] for s in segments)

    def test_metadata_has_expected_keys(self):
        body = "# Titre\n\nContenu."
        segments = segment_markdown(
            body,
            page_title="Titre",
            source_url="https://x.fr/",
            rel_path="content/index.md",
        )
        meta = segments[0]["metadata"]
        assert meta["page_title"] == "Titre"
        assert meta["rel_path"] == "content/index.md"
        assert meta["chunk_index"] == 0
        assert "section_anchor" in meta
        assert "section_path" in meta


class TestEstimateSegmentCount:
    def test_returns_at_least_one(self):
        assert estimate_segment_count("# T\n\nContent") >= 1

    def test_zero_for_empty(self):
        assert estimate_segment_count("") == 0
