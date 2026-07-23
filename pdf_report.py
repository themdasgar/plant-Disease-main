from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(plant, disease, confidence, weather=None):
    filename = "Plant_Disease_Report.pdf"

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Plant Disease Detection Report</b>", styles["Title"]))
    story.append(Paragraph(f"<b>Plant:</b> {plant}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Disease:</b> {disease}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["BodyText"]))

    if weather:
        story.append(Paragraph("<br/><b>Weather Information</b>", styles["Heading2"]))
        story.append(Paragraph(f"Temperature: {weather['temp']} °C", styles["BodyText"]))
        story.append(Paragraph(f"Humidity: {weather['humidity']} %", styles["BodyText"]))
        story.append(Paragraph(f"Weather: {weather['weather']}", styles["BodyText"]))

    doc.build(story)

    return filename