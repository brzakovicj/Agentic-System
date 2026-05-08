from datetime import datetime
import uuid
import os
import html
import re
from bs4 import BeautifulSoup
import markdown

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
    Preformatted,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors

def _generate_file_path(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]

    return os.path.join(output_dir, f"study_script_{timestamp}_{unique_id}.pdf")

def markdown_to_flowables(markdown_text, styles, doc):
    """
    Convert Markdown into ReportLab flowables.
    Supports:
    - headings
    - paragraphs
    - bullet lists
    - tables
    - code blocks
    """

    html_text = markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code"]
    )

    soup = BeautifulSoup(html_text, "html.parser")

    flowables = []

    body_style = styles["BodyText"]

    for element in soup.find_all(recursive=False):

        # ---------------- HEADINGS ----------------

        if element.name == "h1":
            flowables.append(
                Paragraph(element.get_text(), styles["Heading1"])
            )
            flowables.append(Spacer(1, 10))

        elif element.name == "h2":
            flowables.append(
                Paragraph(element.get_text(), styles["Heading2"])
            )
            flowables.append(Spacer(1, 8))

        elif element.name == "h3":
            flowables.append(
                Paragraph(element.get_text(), styles["Heading3"])
            )
            flowables.append(Spacer(1, 6))

        # ---------------- PARAGRAPHS ----------------

        elif element.name == "p":
            flowables.append(
                Paragraph(str(element), body_style)
            )
            flowables.append(Spacer(1, 6))

        # ---------------- BULLET LISTS ----------------

        elif element.name == "ul":

            items = []

            for li in element.find_all("li", recursive=False):
                items.append(
                    ListItem(
                        Paragraph(li.get_text(), body_style)
                    )
                )

            flowables.append(
                ListFlowable(
                    items,
                    bulletType="bullet"
                )
            )

            flowables.append(Spacer(1, 6))

        # ---------------- CODE BLOCKS ----------------

        elif element.name == "pre":

            code = element.get_text()

            flowables.append(
                Preformatted(
                    code,
                    styles["CustomCode"]
                )
            )

            flowables.append(Spacer(1, 6))

        # ---------------- TABLES ----------------

        elif element.name == "table":

            table_data = []

            rows = element.find_all("tr")

            for row in rows:

                cols = row.find_all(["th", "td"])

                table_data.append([
                    Paragraph(
                        col.get_text(strip=True),
                        styles["BodyText"]
                    )
                    for col in cols
                ])

            num_cols = len(table_data[0])

            available_width = doc.width

            col_lengths = [0] * num_cols

            for row in table_data:
                for i, cell in enumerate(row):

                    text = cell.text if hasattr(cell, "text") else str(cell)

                    col_lengths[i] += len(text)

            total_length = sum(col_lengths)

            if total_length == 0:
                total_length = 1

            col_widths = [
                available_width * (length / total_length)
                for length in col_lengths
            ]

            min_width = 35 * mm

            col_widths = [
                max(width, min_width)
                for width in col_widths
            ]
            
            table = Table(
                table_data,
                repeatRows=1,
                colWidths=col_widths
            )

            table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ])
            )

            flowables.append(table)
            flowables.append(Spacer(1, 12))

    return flowables

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

    file_path = _generate_file_path()

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # Better body spacing
    styles["BodyText"].spaceAfter = 6
    styles["BodyText"].leading = 18

    # Code style
    styles.add(
        ParagraphStyle(
            name="CustomCode",
            fontName="Courier",
            fontSize=9,
            leading=12,
            backColor=colors.lightgrey,
            leftIndent=6,
            rightIndent=6,
            spaceBefore=6,
            spaceAfter=6,
        )
    )

    story = []

    # Title
    story.append(
        Paragraph("Study Script", styles["Title"])
    )

    story.append(Spacer(1, 20))

    # Convert markdown into flowables
    story.extend(
        markdown_to_flowables(text, styles, doc)
    )

    try:
        doc.build(story)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to build PDF: {exc}"
        ) from exc

    return file_path
