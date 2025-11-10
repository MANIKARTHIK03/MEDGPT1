from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import os

def generate_report(content, filename="MedGPT_Report.pdf"):
    """Generate a simple PDF report from given text content."""
    file_path = os.path.join(os.getcwd(), filename)
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 750, "MedGPT Health Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 730, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    text_object = c.beginText(50, 700)
    text_object.setFont("Helvetica", 11)

    lines = content.split("\n")
    for line in lines:
        text_object.textLine(line)

    c.drawText(text_object)
    c.save()
    return file_path
