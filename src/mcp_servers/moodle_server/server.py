import json
import os
import re
from pathlib import Path
from typing import Any
from pydantic import BaseModel
import fitz
from fastmcp import FastMCP
from playwright.async_api import async_playwright

class PdfLink(BaseModel):
    href: str
    label: str

# ── FastMCP inicijalizacija ────────────────────────────────────────────────────

mcp = FastMCP(name="Moodle PDF Scraper")

# ── Pomoćne funkcije ───────────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    name = name.replace(" Datoteka", "").strip()
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name + ".pdf"


def _extract_titles_from_pdf(pdf_path: Path) -> list[str]:
    """
    Extracts all section headings from a PDF by detecting spans with:
    - Font size above the document body threshold
    - Or a distinct color (e.g. blue headings common in academic materials)
    Returns a deduplicated list of heading strings.
    """
    try:
        doc = fitz.open(str(pdf_path))
 
        # Collect all font sizes to determine body text size
        all_sizes = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span["text"].strip():
                            all_sizes.append(span["size"])
 
        if not all_sizes:
            return []
 
        # Body text is the most common font size
        from collections import Counter
        body_size = Counter(round(s) for s in all_sizes).most_common(1)[0][0]
        heading_threshold = body_size * 1.05  # anything noticeably larger
 
        seen = set()
        headings = []
 
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text = " ".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()
                    if not line_text or len(line_text) < 3:
                        continue
 
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        size = span["size"]
 
                        is_large = size >= heading_threshold
 
                        if is_large and len(line_text) > 3:
                            key = line_text.lower()
                            if key not in seen:
                                seen.add(key)
                                headings.append(line_text)
                            break  # one span per line is enough
 
        return headings if headings else []
 
    except Exception as e:
        return [f"Error: {e}"]


# ── Alat 1: Skupljanje PDF linkova ─────────────────────────────────────────────

async def fetch_pdf_links(course_url: str) -> list[dict]:
    """
    Scrapes a Moodle course page and returns the course name and a list of all PDF resource links found on it.

    Use this as the first step when the user wants to retrieve course materials from a Moodle URL.
    It uses a headless browser to load the page and detect PDF resources by their icon.

    Args:
        course_url: Full URL of the Moodle course page
                    (e.g. https://imi.pmf.kg.ac.rs/moodle/course/view.php?id=572)

    Returns:
        A dict with fields:
        - course_name: human-readable name of the course extracted from the page heading
                        (e.g. "Uvod u nauku o podacima")
        - links: list of dicts, each with:
            - label: human-readable name of the resource (e.g. "Lecture 4")
            - href:  Moodle resource URL (e.g. .../mod/resource/view.php?id=...)
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await (await browser.new_context()).new_page()
        await page.goto(course_url, wait_until="networkidle")

        # Naziv predmeta iz h1
        course_name = await page.inner_text("div.page-header-headings h1")
        
        # Fallback: zadnji breadcrumb ako h1 ne postoji
        if not course_name:
            crumbs = await page.query_selector_all("ol.breadcrumb li.breadcrumb-item")
            if crumbs:
                course_name = await crumbs[-1].inner_text()
        
        course_name = course_name.strip()

        links = await page.eval_on_selector_all(
            'a:has(img[src*="pdf-24"])',
            """elements => elements.map(el => ({
                href: el.href,
                label: el.innerText.trim()
            }))""",
        )

        await browser.close()

    # Ukloni " Datoteka" sufiks iz labela
    for item in links:
        item["label"] = item["label"].split()[0].strip()

    return {
        "course_name": course_name,
        "links": links,
    }


# ── Alat 2: Preuzimanje PDF-ova ────────────────────────────────────────────────

async def download_pdfs(
    links: list[PdfLink],
    output_dir: str = "moodle_pdfs",
) -> list[dict]:
    """
    Downloads PDF files from a list of Moodle resource links using a headless browser.

    Moodle resource URLs redirect to the actual file only within an active browser session.
    This tool handles those redirects transparently via Playwright.

    Use this after fetch_pdf_links to download the discovered PDFs to a local folder.

    Args:
        links:      List of dicts returned by fetch_pdf_links (fields: label, href)
        output_dir: Local directory where PDFs will be saved (default: "moodle_pdfs")

    Returns:
        A list of dicts with fields:
          - label:    resource name
          - href:     original Moodle URL
          - filename: saved filename on disk
          - path:     absolute path to the saved file (None if download failed)
          - success:  True if downloaded successfully
          - error:    error message if download failed, otherwise None
    """
    import httpx

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)

        # Korak 1: poseti kurs stranicu da uspostavi sesiju sa Moodle-om
        # Bez ovoga Moodle redirectuje nazad na kurs umesto da servira PDF
        if links:
            first_href = links[0].href if hasattr(links[0], "href") else links[0]["href"]
            # Izvuci base URL kursa iz prvog linka
            import re as _re
            base_match = _re.search(r"(https?://[^/]+/moodle)/", first_href)
            if base_match:
                # Poseti kurs stranicu da inicijalizuje sesiju
                init_page = await context.new_page()
                course_id = _re.search(r"id=(\d+)", first_href)
                # Poseti root Moodle stranicu ako ne znamo kurs ID
                await init_page.goto(base_match.group(1) + "/", wait_until="networkidle")
                await init_page.close()

        for item in links:
            href = item.href if hasattr(item, "href") else item["href"]
            label = item.label if hasattr(item, "label") else item["label"]
            label = label.split()[0].strip()
            filename = _sanitize_filename(label)
            dest = out / filename

            try:
                # Poseti resource stranicu u browseru da pratimo redirect
                tab = await context.new_page()
                final_pdf_url = None

                async def on_response(response):
                    nonlocal final_pdf_url
                    ct = response.headers.get("content-type", "")
                    if "pdf" in ct.lower():
                        final_pdf_url = response.url

                tab.on("response", on_response)

                try:
                    await tab.goto(href, wait_until="networkidle", timeout=15_000)
                except Exception:
                    pass  # Download event prekida navigaciju — to je OK

                await tab.close()

                if not final_pdf_url:
                    raise RuntimeError("No PDF response detected — Moodle did not serve a PDF")

                # Preuzmi PDF direktno sa httpx koristeći kolačiće iz browser sesije
                cookies = await context.cookies()
                cookie_dict = {c["name"]: c["value"] for c in cookies}

                async with httpx.AsyncClient(follow_redirects=True, timeout=60, cookies=cookie_dict) as client:
                    r = await client.get(final_pdf_url)
                    if "pdf" not in r.headers.get("content-type", "").lower():
                        raise RuntimeError(f"Unexpected content-type: {r.headers.get('content-type')}")
                    dest.write_bytes(r.content)

                results.append({
                    "label": label,
                    "href": href,
                    "filename": filename,
                    "path": str(dest.resolve()),
                    "success": True,
                    "error": None,
                })

            except Exception as e:
                results.append({
                    "label": label,
                    "href": href,
                    "filename": filename,
                    "path": None,
                    "success": False,
                    "error": str(e),
                })

        await browser.close()

    return results


# ── Alat 3: Ekstrakcija naslova ────────────────────────────────────────────────

def extract_titles(download_results: list[Any]) -> list[dict]:
    """
    Extracts titles from a list of downloaded PDF files.
 
    Tries the following strategies in order:
      1. PDF metadata title field
      2. Largest font text on the first page (likely a heading)
      3. First non-empty line of text on the first page
 
    Use this after download_pdfs to enrich each entry with its extracted title.
 
    Args:
        download_results: List of dicts returned by download_pdfs
 
    Returns:
        A list of dicts with fields:
          - label:           resource name from Moodle
          - filename:        saved filename on disk
          - path:            absolute path to the PDF
          - href:            original Moodle URL
          - title: title string extracted from the PDF, or None if extraction failed
    """
    output = []
    for item in download_results:
        if not item.get("success") or not item.get("path"):
            output.append({
                "label": item["label"],
                "filename": item["filename"],
                "path": item.get("path"),
                "href": item["href"],
                "title": None,
                "error": item.get("error", "Fajl nije preuzet"),
            })
            continue
 
        titles = _extract_titles_from_pdf(Path(item["path"]))
        output.append({
            "label": item["label"],
            "filename": item["filename"],
            "path": item["path"],
            "href": item["href"],
            "title": titles[0] if titles else "Nepoznat naslov",
            "subtopics": titles,
        })
 
    return output


# ── Alat 4: Čuvanje JSON-a ─────────────────────────────────────────────────────

def save_json(
    data: list[dict],
    course_name: str = "Unknown Course",
) -> dict:
    """
    Saves a list of records to a JSON file on disk.

    Use this as the final step after extract_titles to persist the results.

    Args:
        data:        List of dicts to serialize (typically the output of extract_titles)
        course_name: Name of the course (used for organizing the JSON structure)

    Returns:
        A dict with fields:
          - saved: number of records written
          - path:  absolute path to the output file
    """
    base_dir = Path(os.path.abspath(__file__)).parent
    project_root = base_dir.parent.parent.parent
    syllabi_path = project_root / "course_syllabus" / "syllabus.json"

    # Kreira folder ako ne postoji
    syllabi_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {"courses": {}}
    if syllabi_path.exists():
        with open(syllabi_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {"courses": {}}

    existing["courses"][course_name] = {
        "course": course_name,
        "topics": data,
    }

    with open(syllabi_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return {
        "saved": len(data),
        "path": str(syllabi_path.resolve()),
    }

# ── Alat 5: Kompletan pipeline u jednom pozivu ────────────────────────────────
 
@mcp.tool
async def fetch_and_extract_all(
    course_url: str,
) -> dict:
    """
    Runs the complete Moodle PDF pipeline in a single call:
    fetches all PDF links, downloads them, extracts subtopics/titles, and saves to JSON.
 
    ALWAYS use this tool instead of calling fetch_pdf_links, download_pdfs,
    extract_titles, and save_json individually. This is the preferred and only
    tool you should call when a Moodle course URL is available.
 
    Args:
        course_url:  Full URL of the Moodle course page
                     (e.g. https://imi.pmf.kg.ac.rs/moodle/course/view.php?id=572)
 
    Returns:
        A dict with fields:
          - course_name: human-readable name of the course
          - saved:     number of PDFs processed
          - path:      absolute path to the output JSON file
          - materials: list of dicts, each with:
              - label:           resource name from Moodle
              - filename:        saved PDF filename
              - title:           main title of the PDF
              - subtopics:       list of all section headings found in the PDF
    """
    output_dir = "moodle_materials"
    result = await fetch_pdf_links(course_url)
    course_name = result["course_name"] or "Unknown Course"
    links = [PdfLink(href=item["href"], label=item["label"]) for item in result["links"]]
    download_results = await download_pdfs(links, output_dir)
    titled = extract_titles(download_results)
    saved = save_json(titled, course_name)
 
    return {
        "course_name": course_name,
        "saved": saved["saved"],
        "path": saved["path"],
        "materials": [
            {
                "label": t["label"],
                "filename": t["filename"],
                "title": t.get("title"),
                "subtopics": t.get("subtopics", []),
            }
            for t in titled
        ],
    }

# ── Pokretanje ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        raise