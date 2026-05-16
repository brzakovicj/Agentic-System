import os
import uuid
import markdown
from weasyprint import HTML
from datetime import datetime

def _generate_file_path(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]

    return os.path.join(output_dir, f"study_script_{timestamp}_{unique_id}.pdf")

def create_pdf(markdown_text: str) -> str:
    """
    Converts Markdown -> HTML -> PDF using WeasyPrint.
    """

    file_path = _generate_file_path()

    # -----------------------------
    # Markdown -> HTML
    # -----------------------------

    html_content = markdown.markdown(
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
        ]
    )

    # -----------------------------
    # HTML Template
    # -----------------------------

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">

        <style>

            @page {{
                size: A4;
                margin: 25mm;
            }}

            body {{
                font-family: Arial, sans-serif;
                line-height: 1.7;
                color: #222;
                font-size: 12pt;
            }}

            h1 {{
                font-size: 28px;
                color: #1e3a8a;
                margin-top: 32px;
                margin-bottom: 16px;
                border-bottom: 2px solid #1e3a8a;
                padding-bottom: 8px;
            }}

            h2 {{
                font-size: 22px;
                color: #1e40af;
                margin-top: 28px;
                margin-bottom: 12px;
            }}

            h3 {{
                font-size: 18px;
                margin-top: 24px;
                margin-bottom: 10px;
            }}

            p {{
                margin: 10px 0;
            }}

            ul, ol {{
                margin: 10px 0 10px 24px;
            }}

            li {{
                margin-bottom: 6px;
            }}

            code {{
                background-color: #f4f4f4;
                padding: 2px 5px;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 0.95em;
            }}

            pre {{
                background-color: #f4f4f4;
                padding: 14px;
                border-radius: 8px;
                overflow-x: auto;
                margin: 16px 0;
            }}

            pre code {{
                background: none;
                padding: 0;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 11pt;
            }}

            th {{
                background-color: #1e3a8a;
                color: white;
                padding: 10px;
                border: 1px solid #d1d5db;
                text-align: left;
            }}

            td {{
                padding: 10px;
                border: 1px solid #d1d5db;
                vertical-align: top;
            }}

            tr:nth-child(even) {{
                background-color: #f9fafb;
            }}

            blockquote {{
                border-left: 4px solid #d1d5db;
                padding-left: 16px;
                color: #555;
                margin: 16px 0;
            }}

            img {{
                max-width: 100%;
            }}

            hr {{
                border: none;
                border-top: 1px solid #ccc;
                margin: 30px 0;
            }}

        </style>
    </head>

    <body>

        <h1>Study Script</h1>

        {html_content}

    </body>
    </html>
    """

    # -----------------------------
    # HTML -> PDF
    # -----------------------------

    HTML(string=full_html).write_pdf(file_path)

    return file_path