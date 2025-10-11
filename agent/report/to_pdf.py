# agent/report/to_pdf.py
import markdown
from agent.runtime.browser import new_page

CSS = """
  body { font-family: -apple-system, Segoe UI, Roboto, "Noto Sans JP", Arial, sans-serif;
         margin: 24px; line-height: 1.55; font-size: 13.5px; }
  h1,h2,h3 { margin: .6em 0 .3em; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ddd; padding: 6px 8px; }
  code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
"""


def md_to_pdf(md_path: str, pdf_path: str) -> None:
    """Render Markdown -> PDF via headless Chromium; write only if non-empty."""
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html = markdown.markdown(md_text, extensions=["tables"])
    html_doc = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html}</body>
</html>"""

    page = new_page()
    page.set_content(html_doc, wait_until="load")
    page.emulate_media(media="screen")
    # Get bytes first (more reliable), then write ourselves
    pdf_bytes = page.pdf(
        format="A4",
        print_background=True,
        margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"},
    )
    # Only write if looks real
    if pdf_bytes and len(pdf_bytes) > 1024:
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
    else:
        raise RuntimeError("PDF renderer returned empty content")
