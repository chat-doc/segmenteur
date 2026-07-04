"""Prétraitement du Markdown avant segmentation.

Chaîne appliquée dans cet ordre :
  1. Séparation front matter / body
  2. Extraction des alt textes {% image "x.png", "alt utile" %}
  3. Conversion des custom containers :::info ... ::: en blockquote
  4. Retrait des templates 11ty/Nunjucks/Liquid ({% %}, {{ }}, {# #})
  5. Correction des mojibake latin1→utf8 fréquents
  6. Normalisation des blancs
  7. Synthèse de body à partir du front matter si le body est vide
     (cas des cartes catalogues 11ty : title + description + urls)
"""
from __future__ import annotations

import re
from typing import Any, Optional

import yaml


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# Blocs de code triple-backtick à protéger avant strip templates.
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)

# Shortcodes 11ty typiques.
# NB : on utilise des backreferences (\1, \3) pour matcher le meme type de
# quote au debut et a la fin de chaque chaine. Ca permet aux apostrophes
# d'apparaitre a l'interieur d'une chaine double-quotee (cas frequent :
# `{% image "capture.png", "Écran d'installation" %}`).
IMAGE_SHORTCODE_RE = re.compile(
    r"""\{%\s*image\s+(["'])(.+?)\1\s*,\s*(["'])(.+?)\3[^%]*%\}""",
    re.IGNORECASE,
)
VIDEO_SHORTCODE_RE = re.compile(
    r"""\{%\s*video\s+(["'])(.+?)\1\s*,\s*(["'])(.+?)\3[^%]*%\}""",
    re.IGNORECASE,
)

# Templates Nunjucks / Liquid restants.
NUNJUCKS_TAG_RE = re.compile(r"\{%-?.*?-?%\}", re.DOTALL)
NUNJUCKS_VAR_RE = re.compile(r"\{\{-?.*?-?\}\}", re.DOTALL)
NUNJUCKS_COMMENT_RE = re.compile(r"\{#-?.*?-?#\}", re.DOTALL)

# Custom containers markdown-it : :::info Titre optionnel\n...contenu...\n:::
CONTAINER_RE = re.compile(
    r":::(\w+)([^\n]*)\n(.*?)\n:::",
    re.DOTALL,
)

# Mojibake latin1→utf8 fréquents en français (Éduscol, forge, etc.).
MOJIBAKE_MAP = {
    "â€™": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€\x93": "-",
    "â€\x94": "-",
    "â€¦": "...",
    "Å\x93": "oe",
    "Ã©": "é",
    "Ã¨": "è",
    "Ãª": "ê",
    # NB : le motif "Ã " conserve son espace de queue. Le mojibake typique
    # est "Ã\xa0" (Ã + NBSP) qu'on couvre juste apres. Ici on gere aussi
    # le cas defensif "Ã " (avec regular space) pour que la substitution
    # ne mange pas le blanc suivant.
    "Ã ": "à ",
    "Ã\xa0": "à",
    "Ã®": "î",
    "Ã´": "ô",
    "Ã§": "ç",
    "Ã‰": "É",
    "Ã€": "À",
}


def split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Sépare le front matter YAML du corps.

    Retourne (metadata_dict, body). YAML invalide → metadata vide, body inchangé.
    Compatible avec pyyaml, gère l'imbriquation (eleventyNavigation.parent, url.exemple, etc.).
    """
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    raw = match.group(1)
    try:
        meta = yaml.safe_load(raw) or {}
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}
    body = content[match.end():]
    return meta, body


def preserve_image_alt(text: str) -> str:
    """Convertit {% image "path", "alt utile" %} en [Image : alt utile].

    Le alt est souvent porteur d'information (nom d'un bouton, action, contexte).
    Le stripping brut des templates ferait perdre cette info.
    """
    # groupes : 1=quote_ouvrante_fichier, 2=fichier, 3=quote_ouvrante_alt, 4=alt
    text = IMAGE_SHORTCODE_RE.sub(lambda m: f"[Image : {m.group(4).strip()}]", text)
    text = VIDEO_SHORTCODE_RE.sub(lambda m: f"[Vidéo : {m.group(4).strip()}]", text)
    return text


def convert_containers(text: str) -> str:
    """Convertit :::info Titre\\nContenu\\n::: en blockquote markdown.

    Format en sortie :
        > **Info : Titre**
        > Contenu ligne 1
        > Contenu ligne 2
    """
    def replace(match: re.Match) -> str:
        kind = match.group(1).strip().capitalize()
        header = match.group(2).strip()
        body = match.group(3).strip()
        lead = f"> **{kind} : {header}**" if header else f"> **{kind}**"
        body_lines = ["> " + line if line else ">" for line in body.split("\n")]
        return lead + "\n" + "\n".join(body_lines)

    return CONTAINER_RE.sub(replace, text)


def strip_templates(text: str) -> str:
    """Retire les templates Nunjucks/Liquid en préservant les blocs de code.

    On remplace d'abord chaque bloc ``` par un marqueur, on strippe, puis on
    restaure. Ça permet de garder des exemples de code qui contiennent
    volontairement {% %} sans les massacrer.
    """
    marker = f"###CB{id(text) % 1000000}###"
    code_blocks: list[str] = []

    def stash(match: re.Match) -> str:
        code_blocks.append(match.group(0))
        return f"{marker}{len(code_blocks) - 1}{marker}"

    text = CODE_BLOCK_RE.sub(stash, text)
    text = NUNJUCKS_TAG_RE.sub("", text)
    text = NUNJUCKS_VAR_RE.sub("", text)
    text = NUNJUCKS_COMMENT_RE.sub("", text)

    restore_re = re.compile(re.escape(marker) + r"(\d+)" + re.escape(marker))
    text = restore_re.sub(lambda m: code_blocks[int(m.group(1))], text)
    return text


def fix_mojibake(text: str) -> str:
    """Corrige les mojibake latin1→utf8 fréquents.

    On ne touche que les textes qui présentent des marqueurs typiques
    (`Ã`, `â€`, `Å`) pour éviter de casser des textes déjà propres.

    Stratégie :
      1. Applique la table de remplacement ciblée d'abord (deterministe,
         ne mange rien).
      2. Si des marqueurs suspects persistent APRÈS remplacement (donc
         mojibake plus obscur), tente un reencodage global latin1→utf8.

    L'ordre est important : le reencodage global "consomme" les bytes
    `Ã` isolés (fin d'une chaîne, suivis d'un caractère non-continuation)
    et casse `"Ã "` → `" "` sans le fix par map. En appliquant la map en
    premier, on gère les cas courants avant que iconv n'ait l'occasion
    de les massacrer.
    """
    if not ("Ã" in text or "â€" in text or "Å" in text):
        return text

    # 1. Table de remplacement ciblée d'abord.
    for old, new in MOJIBAKE_MAP.items():
        text = text.replace(old, new)

    # 2. Reencodage global uniquement si les marqueurs suspects persistent
    #    (ex: patrons de mojibake pas dans la map).
    if "Ã" in text and any(f"Ã{c}" in text for c in "abcdefghijklmnopqrstuvwxyz"):
        try:
            rebuilt = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if rebuilt and rebuilt != text:
                text = rebuilt
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    return text


def normalize_whitespace(text: str) -> str:
    """Uniformise les fins de ligne et compresse les blancs multiples."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def synthesize_body_from_meta(title: str, meta: dict[str, Any]) -> str:
    """Génère un body markdown à partir du front matter.

    Utile pour les cartes catalogues 11ty (comme les modèles de site) qui n'ont
    que du front matter avec description + urls et pas de contenu markdown.
    Sans ça, ces entrées seraient perdues pour le RAG.
    """
    description = str(meta.get("description") or "").strip()
    urls = meta.get("url")

    if not description and not isinstance(urls, dict):
        return ""

    lines = [f"## {title}"]
    if description:
        lines.append(description)

    if isinstance(urls, dict) and urls:
        lines.append("")
        lines.append("Liens :")
        for key, url in urls.items():
            if isinstance(url, str) and url:
                label = str(key).capitalize()
                lines.append(f"- {label} : {url}")

    tags = meta.get("tags")
    if isinstance(tags, str) and tags:
        lines.append("")
        lines.append(f"Tags : {tags}")
    elif isinstance(tags, list) and tags:
        lines.append("")
        lines.append(f"Tags : {', '.join(str(t) for t in tags)}")

    return "\n\n".join(lines) + "\n"


def preprocess(
    content: str,
    *,
    strip_11ty_templates: bool = True,
    preserve_image_alt_texts: bool = True,
    convert_container_blocks: bool = True,
) -> tuple[dict[str, Any], str]:
    """Pipeline complet : renvoie (metadata_frontmatter, body_pretraite).

    Le body est vide si le fichier ne contient que du front matter.
    """
    meta, body = split_frontmatter(content)
    if preserve_image_alt_texts:
        body = preserve_image_alt(body)
    if convert_container_blocks:
        body = convert_containers(body)
    if strip_11ty_templates:
        body = strip_templates(body)
    body = fix_mojibake(body)
    body = normalize_whitespace(body)
    return meta, body


def is_not_published(meta: dict[str, Any]) -> bool:
    """Détecte les fichiers marqués comme non publiés par 11ty.

    Un `permalink: false` dans le front matter désactive la génération HTML :
    ces pages ne devraient pas être indexées.
    """
    permalink = meta.get("permalink")
    if isinstance(permalink, str) and permalink.strip().lower() == "false":
        return True
    if permalink is False:
        return True
    return False


def get_breadcrumb_parent(meta: dict[str, Any]) -> Optional[str]:
    """Extrait le parent breadcrumb depuis eleventyNavigation.

    Retourne None si absent. Utilisé pour enrichir le préfixe de chaque
    segment (ex. "À propos > Présentation" plutôt que juste "Présentation").
    """
    nav = meta.get("eleventyNavigation")
    if not isinstance(nav, dict):
        return None
    parent = nav.get("parent")
    return str(parent).strip() if parent else None


def title_from_body(body: str) -> Optional[str]:
    """Cherche le premier `# heading` du body si pas de title en front matter."""
    match = re.search(r"^\s*#\s+(.+)$", body, re.MULTILINE)
    return match.group(1).strip() if match else None
