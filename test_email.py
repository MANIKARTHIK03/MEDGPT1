import smtplib
from email.mime.text import MIMEText

sender_email = "sdfsgfsgfafd@gmail.com"
sender_password = "ibka zwcm woai znok"
receiver_email = "pothulamanikarthik001@gmail.com"

msg = MIMEText("Test message from MedGPT app")
msg["Subject"] = "Testing Gmail SMTP"
msg["From"] = sender_email
msg["To"] = receiver_email

with smtplib.SMTP("smtp.gmail.com", 587) as server:
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(msg)
    print("✅ Sent successfully!")
