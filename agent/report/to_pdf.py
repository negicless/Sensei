import markdown
from weasyprint import HTML

def md_to_pdf(md_path: str, pdf_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    html = markdown.markdown(md_text, extensions=["tables"])
    HTML(string=html).write_pdf(pdf_path)
