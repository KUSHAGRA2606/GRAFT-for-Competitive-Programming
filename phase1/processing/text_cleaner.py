"""
text_cleaner.py — Unified text-cleaning utilities for all four sources.

Each cleaner transforms raw source format (LaTeX / MDX / Markdown / HTML)
into clean, readable plain text while preserving structural headers and
optionally tagging code blocks for downstream use.
"""

import re
from html import unescape as html_unescape


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHARED HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines to at most two, strip trailing spaces."""
    # Collapse 3+ consecutive newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on every line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text.strip()


def tag_code_blocks_fenced(text: str) -> str:
    """
    Wrap fenced code blocks (```...```) in [CODE_BLOCK]...[/CODE_BLOCK].
    Preserves the language tag if present.
    """
    def _replacer(m: re.Match) -> str:
        lang = m.group(1) or ""
        code = m.group(2).strip()
        lang_tag = f" lang={lang}" if lang else ""
        return f"\n[CODE_BLOCK{lang_tag}]\n{code}\n[/CODE_BLOCK]\n"

    return re.sub(
        r"```(\w*)\s*\n(.*?)```",
        _replacer,
        text,
        flags=re.DOTALL,
    )


def strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks and [CODE_BLOCK] tags entirely."""
    # Remove fenced blocks
    text = re.sub(r"```\w*\s*\n.*?```", "", text, flags=re.DOTALL)
    # Remove tagged blocks
    text = re.sub(
        r"\[CODE_BLOCK[^\]]*\]\s*\n.*?\[/CODE_BLOCK\]",
        "",
        text,
        flags=re.DOTALL,
    )
    return text


def strip_html_code_blocks(html: str) -> str:
    """Remove <pre> and <code> blocks that contain C/C++ code."""
    # <pre ...>...</pre>
    html = re.sub(r"<pre[^>]*>.*?</pre>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Standalone <code>...</code> blocks (multi-line)
    html = re.sub(
        r"<code[^>]*>.*?</code>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return html


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LATEX CLEANER (for CPH)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_latex(text: str) -> str:
    """
    Convert CPH LaTeX source to clean readable text.

    Keeps: chapter/section headers (as # / ## / ###), prose, simple math.
    Removes: TikZ figures, tabular environments, index/label/ref commands.
    Tags: code listings as [CODE_BLOCK].
    """
    # ── Remove document-level wrappers ──
    text = re.sub(r"\\begin\{document\}|\\end\{document\}", "", text)
    text = re.sub(r"\\documentclass(\[.*?\])?\{.*?\}", "", text)
    text = re.sub(r"\\usepackage(\[.*?\])?\{.*?\}", "", text)
    text = re.sub(r"\\maketitle", "", text)

    # ── Convert sectioning commands → Markdown headers ──
    text = re.sub(r"\\chapter\{(.*?)\}", r"# \1", text)
    text = re.sub(r"\\section\{(.*?)\}", r"## \1", text)
    text = re.sub(r"\\subsection\{(.*?)\}", r"### \1", text)
    text = re.sub(r"\\subsubsection\{(.*?)\}", r"#### \1", text)

    # ── Remove TikZ figures entirely ──
    text = re.sub(
        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
        "[FIGURE]",
        text,
        flags=re.DOTALL,
    )
    # ── Remove center environments (often wrap figures) ──
    text = re.sub(
        r"\\begin\{center\}(.*?)\\end\{center\}",
        r"\1",
        text,
        flags=re.DOTALL,
    )

    # ── Remove tabular environments ──
    text = re.sub(
        r"\\begin\{tabular\}.*?\\end\{tabular\}",
        "[TABLE]",
        text,
        flags=re.DOTALL,
    )

    # ── Tag code listings ──
    def _tag_listing(m: re.Match) -> str:
        code = m.group(1).strip()
        return f"\n[CODE_BLOCK lang=cpp]\n{code}\n[/CODE_BLOCK]\n"

    text = re.sub(
        r"\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}",
        _tag_listing,
        text,
        flags=re.DOTALL,
    )

    # ── Remove verbatim environments, tag as code ──
    text = re.sub(
        r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}",
        _tag_listing,
        text,
        flags=re.DOTALL,
    )

    # ── Strip formatting commands (keep inner text) ──
    for cmd in ("textit", "textbf", "texttt", "emph", "underline", "text"):
        text = re.sub(rf"\\{cmd}\{{(.*?)\}}", r"\1", text)

    # ── Strip reference / index commands ──
    for cmd in ("index", "label", "ref", "cite", "pageref", "footnote"):
        text = re.sub(rf"\\{cmd}\{{.*?\}}", "", text)

    # ── Handle itemize / enumerate ──
    text = re.sub(r"\\begin\{(itemize|enumerate)\}", "", text)
    text = re.sub(r"\\end\{(itemize|enumerate)\}", "", text)
    text = re.sub(r"\\item\s*", "• ", text)

    # ── Handle math environments ──
    # Display math → [MATH]...[/MATH]
    for env in ("equation", "equation*", "align", "align*", "displaymath"):
        text = re.sub(
            rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}",
            r"[MATH]\1[/MATH]",
            text,
            flags=re.DOTALL,
        )
    # Inline math $...$ → keep content
    text = re.sub(r"\$([^$]+?)\$", r"\1", text)
    # \[ ... \] display math
    text = re.sub(r"\\\[(.*?)\\\]", r"[MATH]\1[/MATH]", text, flags=re.DOTALL)

    # ── Remove remaining LaTeX commands (\command or \command{}) ──
    text = re.sub(r"\\[a-zA-Z]+\*?(\{[^}]*\})*", "", text)

    # ── Clean up braces and special characters ──
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\\\\", "\n", text)  # Line breaks
    text = re.sub(r"\\&", "&", text)
    text = re.sub(r"\\ ", " ", text)

    # ── Remove LaTeX comments ──
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    return normalize_whitespace(text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MDX CLEANER (for USACO Guide)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_mdx_frontmatter(text: str) -> tuple[dict, str]:
    """
    Split MDX content into YAML frontmatter dict and body text.
    Returns ({}, text) if no frontmatter found.
    """
    import yaml

    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, flags=re.DOTALL)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, m.group(2)


def clean_mdx(text: str) -> str:
    """
    Clean USACO Guide MDX content to readable text.

    Keeps: Markdown headers, prose, info/warning admonitions (as text).
    Removes: import statements, JSX component tags, language-specific sections
             (keeps CPPSection or language-agnostic content).
    Tags: code blocks as [CODE_BLOCK].
    """
    # ── Remove import statements ──
    text = re.sub(r"^import\s+.*$", "", text, flags=re.MULTILINE)

    # ── Remove Java / Python specific sections entirely ──
    for lang in ("JavaSection", "PySection", "PythonSection"):
        text = re.sub(
            rf"<{lang}[^>]*>.*?</{lang}>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # ── Unwrap CPPSection (keep inner content) ──
    text = re.sub(r"</?CPPSection[^>]*>", "", text, flags=re.IGNORECASE)

    # ── Unwrap LanguageSection (keep content) ──
    text = re.sub(r"</?LanguageSection[^>]*>", "", text, flags=re.IGNORECASE)

    # ── Keep content of Info / Warning / Spoiler but strip tags ──
    for tag in ("Info", "Warning", "Spoiler", "Optional", "FocusProblem",
                "Resources", "Resource", "TextBlock", "IncompleteSection"):
        text = re.sub(rf"</?{tag}[^>]*>", "", text, flags=re.IGNORECASE)

    # ── Remove remaining self-closing JSX tags ──
    text = re.sub(r"<[A-Z][A-Za-z]*\s[^>]*/\s*>", "", text)

    # ── Remove remaining paired JSX tags (simple ones) ──
    text = re.sub(r"</?[A-Z][A-Za-z]*[^>]*>", "", text)

    # ── Tag fenced code blocks ──
    text = tag_code_blocks_fenced(text)

    # ── Simplify inline LaTeX ──
    text = re.sub(r"\$([^$]+?)\$", r"\1", text)

    # ── Remove HTML comments ──
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    return normalize_whitespace(text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MARKDOWN CLEANER (for cp-algorithms)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_markdown(text: str) -> str:
    """
    Clean cp-algorithms Markdown to readable text.

    Keeps: headers, prose, algorithmic explanations.
    Converts: MathJax delimiters to simplified form.
    Strips: MkDocs admonition syntax, content tabs.
    Tags: code blocks as [CODE_BLOCK].
    """
    # ── Tag fenced code blocks FIRST (before other processing) ──
    text = tag_code_blocks_fenced(text)

    # ── Strip MkDocs admonitions (!!!, ???) — keep inner content ──
    # Matches lines like:  !!! note "Title"  or  ??? tip
    text = re.sub(r"^(!{3}|\?{3})\s+\w+(\s+\"[^\"]*\")?\s*$", "", text, flags=re.MULTILINE)

    # ── Strip content tab syntax ──
    text = re.sub(r"^===\s+\"[^\"]*\"\s*$", "", text, flags=re.MULTILINE)

    # ── Convert MathJax display math \[ ... \] ──
    text = re.sub(r"\\\[(.*?)\\\]", r"[MATH]\1[/MATH]", text, flags=re.DOTALL)

    # ── Convert MathJax inline math \( ... \) ──
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text)

    # ── Simplify inline $...$ LaTeX ──
    text = re.sub(r"\$([^$]+?)\$", r"\1", text)

    # ── Remove HTML comments ──
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # ── Remove raw HTML tags (but keep content) ──
    text = re.sub(r"</?(?:div|span|details|summary|p|br|hr)[^>]*>", "", text, flags=re.IGNORECASE)

    # ── Un-indent admonition content (4-space indented lines) ──
    # Only un-indent lines that follow an admonition header
    text = re.sub(r"^    ", "", text, flags=re.MULTILINE)

    return normalize_whitespace(text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HTML CLEANER (for Codeforces editorials)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_html(html: str) -> str:
    """
    Clean Codeforces editorial HTML to readable text.

    Removes: all code blocks (C++ solutions), HTML tags.
    Keeps: editorial explanations, observations, complexity analysis.
    Converts: HTML headers to Markdown headers.
    """
    from bs4 import BeautifulSoup

    if not html or not html.strip():
        return ""

    # ── Remove code blocks BEFORE parsing with BS4 ──
    html = strip_html_code_blocks(html)

    # ── Remove fenced code blocks (sometimes in CF blog markdown) ──
    html = re.sub(r"```\w*\s*\n.*?```", "", html, flags=re.DOTALL)

    # ── Parse with BeautifulSoup ──
    soup = BeautifulSoup(html, "lxml")

    # ── Remove remaining <code>, <pre>, <script>, <style> ──
    for tag in soup.find_all(["code", "pre", "script", "style"]):
        tag.decompose()

    # ── Convert headers to Markdown ──
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            prefix = "#" * level
            tag.replace_with(f"\n{prefix} {tag.get_text(strip=True)}\n")

    # ── Convert <br> → newline ──
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # ── Convert <p> → double newline ──
    for p in soup.find_all("p"):
        p.insert_before("\n")
        p.insert_after("\n")

    # ── Convert <li> → bullet ──
    for li in soup.find_all("li"):
        li.insert_before("• ")
        li.insert_after("\n")

    # ── Convert bold/italic ──
    for b in soup.find_all(["b", "strong"]):
        b.replace_with(f"**{b.get_text()}**")
    for i in soup.find_all(["i", "em"]):
        i.replace_with(f"*{i.get_text()}*")

    # ── Extract text ──
    text = soup.get_text()

    # ── Decode HTML entities ──
    text = html_unescape(text)

    # ── Remove MathJax delimiters (keep content) ──
    text = re.sub(r"\$\$\$(.*?)\$\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$\$(.*?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$([^$]+?)\$", r"\1", text)

    return normalize_whitespace(text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HEADER EXTRACTION UTILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_nearest_header(text: str, position: int) -> str:
    """
    Find the nearest Markdown header (# ...) that appears at or before
    the given character *position* in the text.
    Returns the header text (without the # prefix), or "" if none found.
    """
    headers = list(re.finditer(r"^(#{1,4})\s+(.+)$", text[:position], re.MULTILINE))
    if not headers:
        return ""
    return headers[-1].group(2).strip()
