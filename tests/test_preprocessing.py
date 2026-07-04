"""Tests unitaires du prétraitement Markdown."""
from __future__ import annotations

from app.preprocessing import (
    convert_containers,
    fix_mojibake,
    get_breadcrumb_parent,
    is_not_published,
    normalize_whitespace,
    preprocess,
    preserve_image_alt,
    split_frontmatter,
    strip_templates,
    synthesize_body_from_meta,
    title_from_body,
)


class TestFrontMatter:
    def test_extracts_simple_yaml(self):
        content = "---\ntitle: Test\norder: 1\n---\n\nBody"
        meta, body = split_frontmatter(content)
        assert meta == {"title": "Test", "order": 1}
        assert body.strip() == "Body"

    def test_extracts_nested_yaml(self):
        content = "---\ntitle: Test\neleventyNavigation:\n  key: X\n  parent: Y\n---\nBody"
        meta, body = split_frontmatter(content)
        assert meta["eleventyNavigation"] == {"key": "X", "parent": "Y"}

    def test_no_frontmatter(self):
        meta, body = split_frontmatter("# Just a heading")
        assert meta == {}
        assert body == "# Just a heading"

    def test_invalid_yaml_falls_back_gracefully(self):
        meta, body = split_frontmatter("---\ntitle: [unbalanced\n---\nBody")
        assert meta == {}


class TestImageAltPreservation:
    def test_image_shortcode(self):
        text = 'Voir {% image "img.png", "Bouton paramètres" %} ci-dessus'
        assert preserve_image_alt(text) == "Voir [Image : Bouton paramètres] ci-dessus"

    def test_video_shortcode(self):
        text = '{% video "url.mp4", "Démo" %}'
        assert preserve_image_alt(text) == "[Vidéo : Démo]"

    def test_no_shortcode(self):
        assert preserve_image_alt("Plain text") == "Plain text"


class TestContainers:
    def test_info_with_header(self):
        text = ":::info Un peu d'histoire\nDu contenu\nsur plusieurs lignes\n:::"
        result = convert_containers(text)
        assert "> **Info : Un peu d'histoire**" in result
        assert "> Du contenu" in result
        assert "> sur plusieurs lignes" in result

    def test_warning_no_header(self):
        text = ":::warning\nAttention danger\n:::"
        result = convert_containers(text)
        assert "> **Warning**" in result
        assert "> Attention danger" in result


class TestStripTemplates:
    def test_removes_nunjucks_tags(self):
        text = "Avant {% if true %} milieu {% endif %} après"
        assert strip_templates(text).strip() == "Avant  milieu  après"

    def test_removes_variables(self):
        text = "Page {{ page.url }} fin"
        assert strip_templates(text).strip() == "Page  fin"

    def test_preserves_code_blocks(self):
        text = "Texte\n```js\nconst x = {% if %};\n```\nFin"
        result = strip_templates(text)
        assert "{% if %}" in result  # préservé dans le bloc de code
        assert "```js" in result


class TestMojibake:
    def test_fixes_common_patterns(self):
        assert fix_mojibake("Ã©cole") == "école"
        assert fix_mojibake("Ã ") == "à "

    def test_leaves_clean_text_alone(self):
        clean = "Bonjour, ceci est un texte propre en français."
        assert fix_mojibake(clean) == clean


class TestNormalizeWhitespace:
    def test_collapses_multiple_newlines(self):
        assert normalize_whitespace("A\n\n\n\nB") == "A\n\nB"

    def test_removes_trailing_spaces_after_newline(self):
        assert normalize_whitespace("Line1\n    Line2") == "Line1\nLine2"


class TestSynthesize:
    def test_from_description_and_urls(self):
        meta = {
            "description": "Un calendrier de l'Avent en site 11ty.",
            "url": {
                "exemple": "https://exemple.forge/",
                "depot": "https://depot.forge/",
            },
        }
        result = synthesize_body_from_meta("Calendrier de l'Avent", meta)
        assert "## Calendrier de l'Avent" in result
        assert "Un calendrier de l'Avent" in result
        assert "Exemple : https://exemple.forge/" in result
        assert "Depot : https://depot.forge/" in result

    def test_empty_meta_returns_empty(self):
        assert synthesize_body_from_meta("Titre", {}) == ""


class TestPublished:
    def test_permalink_false_string(self):
        assert is_not_published({"permalink": "false"}) is True

    def test_permalink_false_bool(self):
        assert is_not_published({"permalink": False}) is True

    def test_permalink_absent(self):
        assert is_not_published({"title": "X"}) is False


class TestBreadcrumb:
    def test_extracts_parent(self):
        meta = {"eleventyNavigation": {"key": "Test", "parent": "À propos"}}
        assert get_breadcrumb_parent(meta) == "À propos"

    def test_returns_none_if_absent(self):
        assert get_breadcrumb_parent({"title": "X"}) is None


class TestTitleFromBody:
    def test_finds_first_h1(self):
        assert title_from_body("## Sub\n# Main\n## Autre") == "Main"

    def test_returns_none_without_h1(self):
        assert title_from_body("Just text") is None


class TestPreprocessPipeline:
    def test_full_pipeline_on_11ty_content(self):
        content = """---
title: Installation
eleventyNavigation:
  parent: Guides
---

# Installation

:::info Prérequis
Python 3.11 requis.
:::

Voir {% image "capture.png", "Écran d'installation" %} ci-dessous.
"""
        meta, body = preprocess(content)
        assert meta["title"] == "Installation"
        assert "[Image : Écran d'installation]" in body
        assert "> **Info : Prérequis**" in body
        assert "Python 3.11 requis." in body
        assert get_breadcrumb_parent(meta) == "Guides"
