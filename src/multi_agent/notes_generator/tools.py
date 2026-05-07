from datetime import datetime
import uuid

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import os
import html
import re

def _generate_file_path(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]

    return os.path.join(output_dir, f"study_script_{timestamp}_{unique_id}.pdf")

def _md_to_reportlab(text: str) -> str:
    """
    Convert a subset of Markdown inline formatting to ReportLab XML tags,
    then HTML-escape everything else so ReportLab's Paragraph parser never
    chokes on raw '<', '>', or '&' characters.
 
    Order matters:
      1. Pull out spans we want to keep (bold, italic, inline code).
      2. HTML-escape the remainder.
      3. Re-inject the ReportLab tags.
    """
    # --- Step 1: extract inline markup before escaping ---
    # We replace each match with a unique placeholder, escape the whole
    # string, then substitute the placeholder with the ReportLab tag.
 
    placeholders: dict[str, str] = {}
    counter = [0]  # mutable so the inner lambda can increment it
 
    def stash(tag_open: str, tag_close: str, content: str) -> str:
        key = f"\x00RL{counter[0]}\x00"
        counter[0] += 1
        # The content itself may contain '<' / '>' — escape it too
        safe_content = html.escape(content)
        placeholders[key] = f"{tag_open}{safe_content}{tag_close}"
        return key
 
    # Bold-italic  ***text*** or ___text___
    text = re.sub(
        r"\*\*\*(.+?)\*\*\*|___(.+?)___",
        lambda m: stash("<b><i>", "</i></b>", m.group(1) or m.group(2)),
        text,
    )
    # Bold  **text** or __text__
    text = re.sub(
        r"\*\*(.+?)\*\*|__(.+?)__",
        lambda m: stash("<b>", "</b>", m.group(1) or m.group(2)),
        text,
    )
    # Italic  *text* or _text_
    text = re.sub(
        r"\*(.+?)\*|_(.+?)_",
        lambda m: stash("<i>", "</i>", m.group(1) or m.group(2)),
        text,
    )
    # Inline code  `code`
    text = re.sub(
        r"`(.+?)`",
        lambda m: stash("<font name='Courier'>", "</font>", m.group(1)),
        text,
    )
 
    # --- Step 2: escape everything that remains ---
    text = html.escape(text)
 
    # --- Step 3: restore ReportLab markup ---
    for key, tag in placeholders.items():
        text = text.replace(html.escape(key), tag)
        text = text.replace(key, tag)  # in case escape() didn't touch it
 
    return text

def create_pdf(text: str) -> str:
    """
    Render a Markdown-ish string to a PDF at *file_path*.
 
    Supported Markdown:
      # / ## / ###   headings
      - / *          bullet points
      **bold**        bold
      *italic*        italic
      ***bold-italic***
      `inline code`   monospace
      blank lines     vertical spacing
    """

    # Ensure output directory exists
    file_path = _generate_file_path()

    # Create document
    doc = SimpleDocTemplate(
        file_path,
        pagesize = A4,
        leftMargin = 20 * mm,
        rightMargin = 20 * mm,
        topMargin = 20 * mm,
        bottomMargin = 20 * mm,
    )

    styles = getSampleStyleSheet()

    # A tighter body style with proper word-wrap
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        wordWrap="CJK",  # wraps long unbroken strings too
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=12,
        bulletIndent=0,
        spaceAfter=2,
    )

    story = []
    story.append(Paragraph("Study Script", styles["Title"]))
    story.append(Spacer(1, 20))

    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
 
        if not stripped:
            story.append(Spacer(1, 8))
            continue

        # ---- Headings (no inline formatting needed in headings) ----
        if stripped.startswith("### "):
            safe = html.escape(stripped[4:])
            story.append(Paragraph(safe, styles["Heading3"]))
 
        elif stripped.startswith("## "):
            safe = html.escape(stripped[3:])
            story.append(Paragraph(safe, styles["Heading2"]))
 
        elif stripped.startswith("# "):
            safe = html.escape(stripped[2:])
            story.append(Paragraph(safe, styles["Heading1"]))
 
        # ---- Bullet points ----
        elif stripped.startswith("- ") or stripped.startswith("* "):
            content = _md_to_reportlab(stripped[2:])
            story.append(Paragraph(f"• {content}", bullet_style))
 
        # ---- Normal paragraph ----
        else:
            content = _md_to_reportlab(stripped)
            story.append(Paragraph(content, body_style))
 
        story.append(Spacer(1, 6))

    try:
        doc.build(story)
    except Exception as exc:
        raise RuntimeError(
            f"ReportLab failed to build PDF at '{file_path}'. "
            f"Check for unsupported characters in the source text."
        ) from exc
    
    return file_path
