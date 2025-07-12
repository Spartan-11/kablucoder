
import pandas as pd
from fpdf import FPDF

# Load data
df = pd.read_csv("data.csv")

# Analyze data
average_score = df["Score"].mean()
max_score = df["Score"].max()
min_score = df["Score"].min()
topper = df[df["Score"] == max_score]["Name"].values[0]

# Create PDF report
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, "Student Score Report", ln=True, align='C')

# Summary stats
pdf.set_font("Arial", '', 12)
pdf.ln(10)
pdf.cell(200, 10, f"Average Score: {average_score:.2f}", ln=True)
pdf.cell(200, 10, f"Highest Score: {max_score} (by {topper})", ln=True)
pdf.cell(200, 10, f"Lowest Score: {min_score}", ln=True)

# Table heading
pdf.ln(10)
pdf.set_font("Arial", 'B', 12)
pdf.cell(100, 10, "Name", border=1)
pdf.cell(50, 10, "Score", border=1, ln=True)

# Table data
pdf.set_font("Arial", '', 12)
for index, row in df.iterrows():
    pdf.cell(100, 10, row["Name"], border=1)
    pdf.cell(50, 10, str(row["Score"]), border=1, ln=True)

# Save PDF
pdf.output("report.pdf")

print("PDF report generated: report.pdf")
