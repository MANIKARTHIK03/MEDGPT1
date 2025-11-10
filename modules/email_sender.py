import smtplib
import os
from io import BytesIO
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

load_dotenv()

SENDER_EMAIL = os.getenv("EMAIL_USER", "medgptaiassistant@gmail.com")
SENDER_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "dmzd bzdr jvxq cptd")


def generate_pdf_report(detected_text, description, image_bytes=None):
    """Generate a clean, formatted PDF report with medicine info and optional image."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=60,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.textColor = colors.HexColor("#2b6cb0")

    subtitle_style = ParagraphStyle(
        name="Subtitle",
        fontSize=14,
        leading=18,
        spaceAfter=12,
        textColor=colors.HexColor("#4a5568"),
    )

    normal_style = ParagraphStyle(
        name="Normal",
        fontSize=11,
        leading=16,
        spaceAfter=12,
    )

    # Content list
    content = []

    # Title
    content.append(Paragraph("💊 MedGPT - AI Medicine Report", title_style))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    content.append(Spacer(1, 12))

    # Image (if available)
    if image_bytes:
        image_stream = BytesIO(image_bytes)
        img = RLImage(image_stream, width=250, height=250)
        img.hAlign = "CENTER"
        content.append(img)
        content.append(Spacer(1, 20))

    # Detected text and description
    content.append(Paragraph("🧾 <b>Detected Text:</b>", subtitle_style))
    content.append(Paragraph(detected_text.replace("\n", "<br/>"), normal_style))
    content.append(Spacer(1, 8))

    content.append(Paragraph("💬 <b>Description:</b>", subtitle_style))
    content.append(Paragraph(description.replace("\n", "<br/>"), normal_style))
    content.append(Spacer(1, 12))

    # Disclaimer
    data = [["Note:", "This report is for educational purposes only. Consult a qualified doctor for medical advice."]]
    table = Table(data, colWidths=[60, 440])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 1, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    content.append(Spacer(1, 12))
    content.append(table)

    doc.build(content)
    buffer.seek(0)
    return buffer


def send_medicine_email(to_email, detected_text, description, image_bytes=None, image_filename=None):
    """Send an HTML-styled email with inline image and attached PDF report."""
    msg = MIMEMultipart("related")
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email
    msg["Subject"] = "💊 Your MedGPT Medicine Report"

    # Generate PDF with image
    pdf_buffer = generate_pdf_report(detected_text, description, image_bytes=image_bytes)

    # HTML email with inline image
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height:1.6; color:#333;">
        <h2 style="color:#2b6cb0;">💊 MedGPT - AI Medicine Report</h2>
        <p>Here’s the analysis of your uploaded medicine:</p>

        <h3 style="color:#4a5568;">🧾 Detected Text:</h3>
        <p>{detected_text}</p>

        <h3 style="color:#4a5568;">💬 Description:</h3>
        <p>{description}</p>

        {"<h3>📸 Uploaded Image:</h3><img src='cid:image1' style='width:300px;border-radius:10px;'/>" if image_bytes else ""}

        <p style="margin-top:20px;">⚠️ <i>This information is for educational purposes only. Please consult a licensed doctor for medical advice.</i></p>
        <p>— <b>MedGPT Team</b></p>
    </body>
    </html>
    """

    msg.attach(MIMEText(html_content, "html"))

    # Inline image (for email)
    if image_bytes and image_filename:
        img = MIMEImage(image_bytes, name=image_filename)
        img.add_header("Content-ID", "<image1>")
        msg.attach(img)

    # Attach PDF report
    pdf_attach = MIMEApplication(pdf_buffer.read(), _subtype="pdf")
    pdf_attach.add_header("Content-Disposition", "attachment", filename="AI_Medicine_Report.pdf")
    msg.attach(pdf_attach)

    # Send email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        return f"⚠️ Error sending email: {e}"
# ----------------------------------------------------------------------
# ✅ Legacy Compatibility Function
# This allows older parts of app.py that import send_prescription() to still work.
# ----------------------------------------------------------------------
def send_prescription(email, pdf_path):
    """Legacy helper to send an existing PDF prescription file."""
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = email
        msg["Subject"] = "Your AI Health Prescription (Educational Use Only)"

        # Email body
        body = """\
Hello,

Attached is your AI-generated prescription summary.

Please note: This is for educational purposes only. 
Consult a licensed doctor before taking any medicine.

Stay safe!
— MedGPT Team
"""
        msg.attach(MIMEText(body, "plain"))

        # Attach PDF file
        with open(pdf_path, "rb") as f:
            attach = MIMEApplication(f.read(), _subtype="pdf")
            attach.add_header("Content-Disposition", "attachment", filename="AI_Prescription.pdf")
            msg.attach(attach)

        # Send email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        return "✅ Prescription sent successfully!"
    except Exception as e:
        return f"⚠️ Error sending prescription: {e}"
