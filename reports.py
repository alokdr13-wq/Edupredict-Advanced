from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def prediction_pdf(result):
    buffer=BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=letter,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=42)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Sub",parent=styles["Normal"],textColor=colors.HexColor("#475569"),leading=16))
    story=[Paragraph("EduPredict",styles["Title"]),Paragraph("Student Performance Prediction Report",styles["Heading2"]),Spacer(1,12)]
    rows=[["Student",result["name"]],["Roll number",result.get("roll","") or "Not provided"],["Predicted mathematics grade",f'{result["grade"]:.2f} / 20'],["Performance outlook",result["category"]],["Report ID",f'EP-{result["id"]:05d}']]
    table=Table(rows,colWidths=[2.2*inch,4.1*inch])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#e8eef7")),("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#0f172a")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#cbd5e1")),("PADDING",(0,0),(-1,-1),8)]))
    story.extend([table,Spacer(1,18),Paragraph("Input details",styles["Heading3"])])
    input_rows=[["Field","Value"]]+[[str(k),str(v)] for k,v in result["features"].items()]
    it=Table(input_rows,colWidths=[2.2*inch,4.1*inch])
    it.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dbeafe")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#cbd5e1")),("PADDING",(0,0),(-1,-1),6)]))
    story.extend([it,Spacer(1,18),Paragraph("Suggested next steps",styles["Heading3"])])
    story.extend([Paragraph("• "+t,styles["Sub"]) for t in result["tips"]])
    story.extend([Spacer(1,16),Paragraph("Important limitation",styles["Heading3"]),Paragraph("This model is trained on a public dataset of students from two Portuguese secondary schools. This estimate is not a guaranteed outcome and may not generalize to other settings. It must not be used as the sole basis for educational decisions.",styles["Sub"])])
    doc.build(story)
    return buffer.getvalue()
