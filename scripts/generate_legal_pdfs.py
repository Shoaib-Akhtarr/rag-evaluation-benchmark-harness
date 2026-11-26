import os
from fpdf import FPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(ROOT, "documents")
os.makedirs(DOCS_DIR, exist_ok=True)

LEGAL_PDFS = [
    (
        "01_Master_Services_Agreement.pdf",
        "Enterprise Cloud Master Services Agreement (Effective: 2026-01-15)",
        [
            "Section 9.1 Cap on Liability.\n"
            "Except for breaches of confidentiality or IP indemnification obligations, neither party's aggregate cumulative liability arising out of or related to this Agreement shall exceed the total fees paid by Customer to Provider in the twelve (12) months immediately preceding the event giving rise to liability.\n\n"
            "Section 14.2 Governing Law.\n"
            "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflict of laws principles.",
            "Section 11.4 Non-Compete Covenant.\n"
            "During the Term and for a period of twenty-four (24) months following expiration or termination for cause, Contractor shall not directly develop or market a competing multi-tenant vector storage engine within North America or the EEA.\n\n"
            "Section 12.1 Anti-Assignment.\n"
            "Neither party may assign this Agreement without prior written consent, except in connection with a merger, acquisition, or sale of substantially all assets."
        ],
    ),
    (
        "02_Data_Processing_Addendum_GDPR.pdf",
        "Data Processing Addendum - GDPR & Standard Contractual Clauses",
        [
            "Section 4.2 Cross-Border Transfers & Breach Notice.\n"
            "Processor shall process EEA personal data solely in EU-central-2 unless Customer executes the 2021 EU Standard Contractual Clauses (Module 2). Processor shall notify Customer within 48 hours of any confirmed Personal Data Breach.\n\n"
            "Section 4.5 Subprocessors.\n"
            "Provider shall give 30 days advance email notice before onboarding a new subprocessor. Customer may object on reasonable data-protection grounds within 15 days."
        ],
    ),
    (
        "03_SLA_Uptime_and_Termination_Schedule.pdf",
        "Service Level Agreement, Renewal & Termination Schedule",
        [
            "Section 2.1 Monthly Uptime Commitment.\n"
            "Provider guarantees 99.95% Monthly Uptime Percentage for Production APIs. If Monthly Uptime falls below 99.95%, Customer receives a 10% Service Credit; below 99.0%, a 25% Service Credit.\n\n"
            "Section 6.3 Termination for Convenience.\n"
            "Customer may terminate for convenience upon sixty (60) days prior written notice, paying accrued fees plus an early termination fee equal to 25% of remaining committed annual minimum spend.",
            "Section 5.3 Renewal Price Cap.\n"
            "Upon automatic 12-month renewal, Provider may increase recurring subscription fees by no more than the lesser of 5% or the U.S. Consumer Price Index (CPI-U) for the prior year. Non-renewal notice is 45 days."
        ],
    ),
    (
        "04_AI_Governance_and_IP_Rider.pdf",
        "AI Data Governance, Zero-Training & Intellectual Property Rider",
        [
            "Section 8.1 Customer Content Ownership.\n"
            "Customer retains all right, title, and interest in Customer Data and Output. Provider retains all IP rights in the underlying Platform and Background IP.\n\n"
            "Section 8.4 Zero AI Training Guarantee.\n"
            "Provider shall not use Customer Prompts, embeddings, or fine-tuning datasets to train or improve foundation models shared with third parties, enforcing zero-day prompt retention."
        ],
    ),
    (
        "05_Security_Audit_Escrow_Insurance.pdf",
        "Security Audit, Source Code Escrow & Insurance Schedule",
        [
            "Section 10.3 Annual Security Audit.\n"
            "Once per twelve (12) months upon 30 days written notice, Customer may review Provider's independent SOC 2 Type II and ISO 27001 audit reports at no additional charge.\n\n"
            "Section 13.4 Source Code Escrow.\n"
            "Provider deposits source code with Iron Mountain Escrow, released solely upon Chapter 7 bankruptcy or permanent cessation of support.\n\n"
            "Section 18.1 Required Insurance.\n"
            "Provider shall maintain Cyber & Tech E&O insurance of $5,000,000 per occurrence and Commercial General Liability of $2,000,000 per occurrence."
        ],
    ),
    (
        "06_Dispute_Resolution_and_Data_Purge.pdf",
        "Dispute Resolution, Arbitration & Post-Termination Data Purge",
        [
            "Section 14.5 Binding Arbitration.\n"
            "Any dispute exceeding $100,000 not resolved within 30 days shall be settled by binding arbitration administered by JAMS in Wilmington, Delaware.\n\n"
            "Section 6.6 Data Return & Purge.\n"
            "Upon termination, Customer Data is exportable in JSON/Parquet format for 30 days and cryptographically erased within 14 days thereafter."
        ],
    ),
]


def build_pdfs():
    for filename, title, pages in LEGAL_PDFS:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        for idx, body in enumerate(pages, 1):
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 8, title if idx == 1 else f"{title} (Page {idx})")
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, body)
        pdf.output(os.path.join(DOCS_DIR, filename))
    print(f"Generated {len(LEGAL_PDFS)} legal PDFs in {DOCS_DIR}")


if __name__ == "__main__":
    build_pdfs()
