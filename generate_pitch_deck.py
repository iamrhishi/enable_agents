from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Wedge, Polygon
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.piecharts import Pie
from reportlab.pdfgen import canvas
import os

# ── Brand colours ──────────────────────────────────────────────
NAVY        = colors.HexColor('#0A1931')
ELECTRIC    = colors.HexColor('#1565C0')
ELECTRIC_LT = colors.HexColor('#1976D2')
ACCENT      = colors.HexColor('#00BCD4')
ACCENT_SOFT = colors.HexColor('#E0F7FA')
WHITE       = colors.white
LIGHT_GREY  = colors.HexColor('#F5F7FA')
MID_GREY    = colors.HexColor('#B0BEC5')
DARK_GREY   = colors.HexColor('#37474F')
GOLD        = colors.HexColor('#FFB300')
GREEN       = colors.HexColor('#2E7D32')
GREEN_LT    = colors.HexColor('#E8F5E9')
RED_LT      = colors.HexColor('#FFEBEE')

W, H = A4

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'Enable_Agents_Pitch_Deck.pdf')


# ── Canvas callbacks (background, header bar, page number) ─────
def draw_slide_background(c, doc):
    """Full-page light background + left navy accent bar."""
    c.saveState()
    c.setFillColor(LIGHT_GREY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    # Left accent bar
    c.setFillColor(NAVY)
    c.rect(0, 0, 8 * mm, H, fill=1, stroke=0)
    c.restoreState()


def draw_cover_background(c, doc):
    """Full navy cover page."""
    c.saveState()
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    # Decorative accent circle
    c.setFillColor(ELECTRIC)
    c.circle(W - 30 * mm, H - 30 * mm, 60 * mm, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.circle(W - 30 * mm, H - 30 * mm, 35 * mm, fill=1, stroke=0)
    # Bottom stripe
    c.setFillColor(ELECTRIC)
    c.rect(0, 0, W, 14 * mm, fill=1, stroke=0)
    c.restoreState()


def draw_section_background(c, doc):
    """Section divider page – navy with accent."""
    c.saveState()
    c.setFillColor(ELECTRIC)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H * 0.55, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, 14 * mm, 6 * mm, H - 28 * mm, fill=1, stroke=0)
    c.restoreState()


# ── Helper: styled paragraph factories ─────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    def ps(name, **kw):
        return ParagraphStyle(name=name, **kw)

    cover_title = ps('CoverTitle', fontName='Helvetica-Bold', fontSize=38,
                     textColor=WHITE, leading=46, spaceAfter=6)
    cover_sub = ps('CoverSub', fontName='Helvetica', fontSize=16,
                   textColor=ACCENT, leading=22, spaceAfter=4)
    cover_tagline = ps('CoverTagline', fontName='Helvetica-Oblique', fontSize=12,
                       textColor=MID_GREY, leading=18)

    slide_h1 = ps('SlideH1', fontName='Helvetica-Bold', fontSize=22,
                  textColor=NAVY, leading=28, spaceAfter=4, spaceBefore=0)
    slide_h2 = ps('SlideH2', fontName='Helvetica-Bold', fontSize=15,
                  textColor=ELECTRIC, leading=20, spaceAfter=3, spaceBefore=6)
    slide_h3 = ps('SlideH3', fontName='Helvetica-Bold', fontSize=12,
                  textColor=DARK_GREY, leading=16, spaceAfter=2, spaceBefore=4)
    body = ps('Body', fontName='Helvetica', fontSize=10,
              textColor=DARK_GREY, leading=15, spaceAfter=3)
    body_bold = ps('BodyBold', fontName='Helvetica-Bold', fontSize=10,
                   textColor=NAVY, leading=15, spaceAfter=2)
    bullet = ps('Bullet', fontName='Helvetica', fontSize=10,
                textColor=DARK_GREY, leading=15, spaceAfter=2,
                leftIndent=14, firstLineIndent=-10)
    small = ps('Small', fontName='Helvetica', fontSize=8,
               textColor=MID_GREY, leading=12, spaceAfter=2)
    section_title = ps('SectionTitle', fontName='Helvetica-Bold', fontSize=30,
                       textColor=WHITE, leading=38, spaceAfter=6, alignment=TA_CENTER)
    section_num = ps('SectionNum', fontName='Helvetica-Bold', fontSize=60,
                     textColor=ELECTRIC_LT, leading=68, spaceAfter=0, alignment=TA_CENTER)
    caption = ps('Caption', fontName='Helvetica-Oblique', fontSize=8,
                 textColor=MID_GREY, leading=11, alignment=TA_CENTER)
    metric_num = ps('MetricNum', fontName='Helvetica-Bold', fontSize=26,
                    textColor=ELECTRIC, leading=30, alignment=TA_CENTER)
    metric_lbl = ps('MetricLbl', fontName='Helvetica', fontSize=9,
                    textColor=DARK_GREY, leading=13, alignment=TA_CENTER)
    quote = ps('Quote', fontName='Helvetica-Oblique', fontSize=11,
               textColor=DARK_GREY, leading=17, leftIndent=20, rightIndent=20,
               spaceAfter=4, spaceBefore=4)
    tag = ps('Tag', fontName='Helvetica-Bold', fontSize=9,
             textColor=WHITE, leading=13, alignment=TA_CENTER)

    return dict(
        cover_title=cover_title, cover_sub=cover_sub, cover_tagline=cover_tagline,
        slide_h1=slide_h1, slide_h2=slide_h2, slide_h3=slide_h3,
        body=body, body_bold=body_bold, bullet=bullet, small=small,
        section_title=section_title, section_num=section_num,
        caption=caption, metric_num=metric_num, metric_lbl=metric_lbl,
        quote=quote, tag=tag
    )


ST = make_styles()


# ── Reusable building blocks ────────────────────────────────────
def hline(color=MID_GREY, width=1):
    return HRFlowable(width='100%', thickness=width, color=color, spaceAfter=6, spaceBefore=6)


def metric_card(number, label, color=ELECTRIC):
    style = ParagraphStyle('mc_num', fontName='Helvetica-Bold', fontSize=24,
                           textColor=color, leading=28, alignment=TA_CENTER)
    lbl_style = ParagraphStyle('mc_lbl', fontName='Helvetica', fontSize=9,
                               textColor=DARK_GREY, leading=13, alignment=TA_CENTER)
    return Table(
        [[Paragraph(number, style)], [Paragraph(label, lbl_style)]],
        colWidths=[None],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), WHITE),
            ('BOX', (0, 0), (-1, -1), 1, ELECTRIC),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('ROUNDEDCORNERS', [6, 6, 6, 6]),
        ])
    )


def tag_pill(text, bg=ELECTRIC, fg=WHITE):
    t_style = ParagraphStyle('pill', fontName='Helvetica-Bold', fontSize=8,
                             textColor=fg, leading=12, alignment=TA_CENTER)
    return Table(
        [[Paragraph(text, t_style)]],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ])
    )


def feature_row(icon_char, title, desc):
    icon_style = ParagraphStyle('ic', fontName='Helvetica-Bold', fontSize=14,
                                textColor=ACCENT, leading=18, alignment=TA_CENTER)
    t_style = ParagraphStyle('ft', fontName='Helvetica-Bold', fontSize=10,
                             textColor=NAVY, leading=14)
    d_style = ParagraphStyle('fd', fontName='Helvetica', fontSize=9,
                             textColor=DARK_GREY, leading=13)
    return Table(
        [[Paragraph(icon_char, icon_style), [Paragraph(title, t_style), Paragraph(desc, d_style)]]],
        colWidths=[22, None],
        style=TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ])
    )


# ── TAM/SAM/SOM funnel drawing ──────────────────────────────────
def tam_funnel():
    d = Drawing(320, 200)
    layers = [
        (320, 50, NAVY,     "$500B",  "TAM — Global SMB Software"),
        (220, 50, ELECTRIC, "$42B",   "SAM — AI-Enabled SMB Tools"),
        (130, 50, ACCENT,   "$1.2B",  "SOM — English-Speaking SMBs (Yr 3)"),
    ]
    y = 180
    for w, h, col, val, lbl in layers:
        x = (320 - w) / 2
        y -= h + 6
        d.add(Rect(x, y, w, h, fillColor=col, strokeColor=WHITE, strokeWidth=1))
        d.add(String(160, y + h / 2 + 4, val, fontName='Helvetica-Bold', fontSize=13,
                     fillColor=WHITE, textAnchor='middle'))
        d.add(String(160, y + h / 2 - 9, lbl, fontName='Helvetica', fontSize=8,
                     fillColor=WHITE, textAnchor='middle'))
    return d


# ── Pricing table ───────────────────────────────────────────────
def pricing_table():
    header = ['Plan', 'Agents Included', 'Price / Month', 'Best For']
    rows = [
        ['Starter', '3 agents of choice', '$79', 'Solo founders, freelancers'],
        ['Growth', '8 agents of choice', '$179', 'Small teams (2-10 staff)'],
        ['Scale', 'Unlimited agents', '$349', 'Growing SMBs (10-50 staff)'],
        ['Enterprise', 'Custom + white-label', 'Custom', 'Franchises / resellers'],
    ]
    h_style = ParagraphStyle('th', fontName='Helvetica-Bold', fontSize=9,
                             textColor=WHITE, alignment=TA_CENTER)
    c_style = ParagraphStyle('td', fontName='Helvetica', fontSize=9,
                             textColor=DARK_GREY, alignment=TA_CENTER)
    b_style = ParagraphStyle('tdB', fontName='Helvetica-Bold', fontSize=9,
                             textColor=ELECTRIC, alignment=TA_CENTER)
    g_style = ParagraphStyle('tdG', fontName='Helvetica-Bold', fontSize=9,
                             textColor=GREEN, alignment=TA_CENTER)
    data = [[Paragraph(h, h_style) for h in header]]
    for i, row in enumerate(rows):
        styled = [
            Paragraph(row[0], b_style),
            Paragraph(row[1], c_style),
            Paragraph(row[2], g_style),
            Paragraph(row[3], c_style),
        ]
        data.append(styled)

    ts = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('BACKGROUND', (0, 1), (-1, 1), ACCENT_SOFT),
        ('BACKGROUND', (0, 2), (-1, 2), WHITE),
        ('BACKGROUND', (0, 3), (-1, 3), ACCENT_SOFT),
        ('BACKGROUND', (0, 4), (-1, 4), WHITE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [ACCENT_SOFT, WHITE]),
        ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # Highlight Growth plan
        ('BOX', (0, 2), (-1, 2), 1.5, ELECTRIC),
    ])
    col_w = [75, 120, 80, 135]
    return Table(data, colWidths=col_w, style=ts)


# ── Section divider page ────────────────────────────────────────
def section_divider(num, title, subtitle=''):
    elems = []
    elems.append(PageBreak())
    elems.append(Spacer(1, H * 0.25))
    elems.append(Paragraph(str(num).zfill(2), ST['section_num']))
    elems.append(Paragraph(title, ST['section_title']))
    if subtitle:
        sub_st = ParagraphStyle('sec_sub', fontName='Helvetica', fontSize=13,
                                textColor=ACCENT, alignment=TA_CENTER, leading=18)
        elems.append(Spacer(1, 4))
        elems.append(Paragraph(subtitle, sub_st))
    elems.append(PageBreak())
    return elems


def slide_title(text, sub=None):
    elems = [Paragraph(text, ST['slide_h1'])]
    elems.append(HRFlowable(width='100%', thickness=3, color=ELECTRIC, spaceAfter=8))
    if sub:
        elems.append(Paragraph(sub, ST['body']))
    return elems


# ══════════════════════════════════════════════════════════════════
# BUILD DOCUMENT
# ══════════════════════════════════════════════════════════════════
def build():
    story = []

    # Margins: left 22mm (to clear accent bar), right 18mm, top/bottom 18mm
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title='Enable Agents — Investor & Customer Pitch Deck',
        author='Enable',
        subject='Pitch Deck 2026',
    )

    # ── 1. COVER ────────────────────────────────────────────────
    story.append(Spacer(1, 55 * mm))
    co_h1 = ParagraphStyle('ch1', fontName='Helvetica-Bold', fontSize=42,
                           textColor=WHITE, leading=50, spaceAfter=8)
    co_sub = ParagraphStyle('csub', fontName='Helvetica', fontSize=16,
                            textColor=ACCENT, leading=22)
    co_tag = ParagraphStyle('ctag', fontName='Helvetica-Oblique', fontSize=11,
                            textColor=MID_GREY, leading=17, spaceAfter=50)
    co_meta = ParagraphStyle('cmeta', fontName='Helvetica', fontSize=9,
                             textColor=MID_GREY, leading=14)

    story.append(Paragraph("Enable", co_h1))
    story.append(Paragraph("AI Agents for Every Part of Your Business", co_sub))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "Your on-demand workforce — 30+ specialised AI agents that handle sales, marketing, "
        "operations, finance, and more so you can focus on what only you can do.",
        co_tag))
    story.append(hline(MID_GREY, 0.5))
    story.append(Spacer(1, 3 * mm))

    cover_meta = Table(
        [['Investor & Customer Pitch Deck', '', 'April 2026', '', 'CONFIDENTIAL']],
        colWidths=[120, 10, 80, 10, 100],
        style=TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('TEXTCOLOR', (0, 0), (-1, -1), MID_GREY),
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (2, 0), (2, 0), 'CENTER'),
            ('ALIGN', (4, 0), (4, 0), 'RIGHT'),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ])
    )
    story.append(cover_meta)
    story.append(PageBreak())

    # ── 2. TABLE OF CONTENTS ────────────────────────────────────
    story += slide_title("Table of Contents")
    toc_items = [
        ("01", "Problem Statement",       "The hidden tax on small business owners"),
        ("02", "Ideal Customer Persona",  "Who we built Enable for"),
        ("03", "Market Opportunity",      "TAM · SAM · SOM"),
        ("04", "Product-Market Fit",      "Evidence that the market needs us"),
        ("05", "Product & Features",      "What Enable actually does"),
        ("06", "Differentiation",         "Why we win"),
        ("07", "Pricing",                 "Simple, modular, affordable"),
        ("08", "Positioning & Growth",    "Go-to-market and scale strategy"),
        ("09", "The Ask",                 "What we need & what you get"),
    ]
    toc_data = []
    for num, title, sub in toc_items:
        n_st = ParagraphStyle('tn', fontName='Helvetica-Bold', fontSize=22,
                              textColor=ELECTRIC, leading=26)
        t_st = ParagraphStyle('tt', fontName='Helvetica-Bold', fontSize=11,
                              textColor=NAVY, leading=15)
        s_st = ParagraphStyle('ts', fontName='Helvetica', fontSize=9,
                              textColor=DARK_GREY, leading=13)
        toc_data.append([
            Paragraph(num, n_st),
            [Paragraph(title, t_st), Paragraph(sub, s_st)]
        ])

    toc_table = Table(toc_data, colWidths=[35, None],
                      style=TableStyle([
                          ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                          ('ROWBACKGROUNDS', (0, 0), (-1, -1), [LIGHT_GREY, WHITE]),
                          ('TOPPADDING', (0, 0), (-1, -1), 7),
                          ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                          ('LEFTPADDING', (0, 0), (0, -1), 6),
                          ('LEFTPADDING', (1, 0), (1, -1), 10),
                          ('LINEBELOW', (0, 0), (-1, -2), 0.3, MID_GREY),
                      ]))
    story.append(toc_table)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SECTION 01 — PROBLEM STATEMENT
    # ═══════════════════════════════════════════════════════════
    story += section_divider(1, "Problem Statement",
                             "The hidden tax on small business owners")

    story += slide_title("The Problem",
                         "Running a small business means wearing every hat — and wearing them badly.")
    story.append(Spacer(1, 4))

    problem_cards = [
        ("Too Many Roles, Too Few Hours",
         "The average small business owner works 52 hours per week. "
         "They are simultaneously the CEO, marketer, sales rep, accountant, "
         "HR manager, and customer service agent. Critical tasks get dropped."),
        ("Skilled Help is Unaffordable",
         "Hiring a full marketing team, a sales manager, and an operations "
         "director costs $300K+/year — out of reach for 99% of SMBs. "
         "Freelancers are fragmented, slow to onboard, and inconsistent."),
        ("Generic Software Doesn't Help",
         "CRMs, ERPs, and marketing suites were designed for enterprises. "
         "They're complex, expensive, and require a dedicated admin to maintain. "
         "SMBs pay for features they never use while missing the intelligence they need."),
        ("The Consequence",
         "70% of small businesses fail within 10 years. "
         "Operational overwhelm — not market demand — is cited as a top-3 cause. "
         "Owners burn out before they can scale."),
    ]

    for title, desc in problem_cards:
        t_st = ParagraphStyle('pct', fontName='Helvetica-Bold', fontSize=11,
                              textColor=NAVY, leading=15)
        d_st = ParagraphStyle('pcd', fontName='Helvetica', fontSize=10,
                              textColor=DARK_GREY, leading=15)
        card = Table(
            [[Paragraph(title, t_st)], [Paragraph(desc, d_st)]],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                ('BACKGROUND', (0, 1), (-1, 1), WHITE),
                ('TOPPADDING', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
                ('BOX', (0, 0), (-1, -1), 0.5, ELECTRIC),
            ])
        )
        story.append(card)
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 5))
    story.append(Paragraph(
        '"I started this business to do what I love. Instead, I spend 80% of my time '
        'on things I hate and am not qualified for." — Typical small business owner',
        ST['quote']))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SECTION 02 — IDEAL CUSTOMER PERSONA
    # ═══════════════════════════════════════════════════════════
    story += section_divider(2, "Ideal Customer Persona",
                             "Who we built Enable for")

    story += slide_title("Our Customer: The Overwhelmed Owner-Operator")
    story.append(Spacer(1, 4))

    # Persona card
    persona_left = [
        Paragraph("Alex, 38", ParagraphStyle('pname', fontName='Helvetica-Bold',
                                              fontSize=18, textColor=WHITE, leading=22)),
        Paragraph("Founder & CEO · Boutique Marketing Agency · 6 employees",
                  ParagraphStyle('prole', fontName='Helvetica', fontSize=10,
                                 textColor=ACCENT, leading=15)),
        Spacer(1, 6),
        Paragraph("Revenue: $400K–$1.2M/year", ST['small']),
        Paragraph("Team size: 1–15 people", ST['small']),
        Paragraph("Tech savvy: moderate", ST['small']),
        Paragraph("Budget for tools: $200–$600/month", ST['small']),
    ]
    persona_right = [
        Paragraph("Goals", ParagraphStyle('pg', fontName='Helvetica-Bold', fontSize=11,
                                          textColor=NAVY, leading=15)),
        Paragraph("• Grow revenue without proportionally growing headcount", ST['bullet']),
        Paragraph("• Deliver consistent quality to clients even on a tight team", ST['bullet']),
        Paragraph("• Reclaim personal time and reduce 60-hour weeks", ST['bullet']),
        Spacer(1, 4),
        Paragraph("Pain Points", ParagraphStyle('pp', fontName='Helvetica-Bold', fontSize=11,
                                                textColor=NAVY, leading=15)),
        Paragraph("• Juggling sales prospecting, content creation, and client delivery simultaneously", ST['bullet']),
        Paragraph("• Can't afford full-time staff for every function", ST['bullet']),
        Paragraph("• Existing tools are siloed — data lives in 7 different apps", ST['bullet']),
        Paragraph("• Spends 3+ hours/day on repetitive, low-value tasks", ST['bullet']),
    ]

    persona_table = Table(
        [[persona_left, persona_right]],
        colWidths=[120, None],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), NAVY),
            ('BACKGROUND', (1, 0), (1, 0), LIGHT_GREY),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (0, 0), 12),
            ('LEFTPADDING', (1, 0), (1, 0), 14),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('BOX', (0, 0), (-1, -1), 1, ELECTRIC),
        ])
    )
    story.append(persona_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Secondary Personas", ST['slide_h2']))
    sec_personas = [
        ("Solo Freelancer / Consultant",
         "1-person business • $80K–$250K revenue • Needs sales + delivery support"),
        ("Restaurant / Retail Owner",
         "3–20 employees • Physical location • Needs ops, inventory, and marketing agents"),
        ("Service Business Owner",
         "Plumber, accountant, therapist etc. • Needs scheduling, invoicing, and client comms"),
        ("E-commerce Founder",
         "Online store • Needs orders, supplier tracking, content, and customer support agents"),
    ]
    sp_data = [[
        Paragraph(t, ParagraphStyle('spt', fontName='Helvetica-Bold', fontSize=9,
                                    textColor=NAVY, leading=13)),
        Paragraph(d, ParagraphStyle('spd', fontName='Helvetica', fontSize=9,
                                    textColor=DARK_GREY, leading=13))
    ] for t, d in sec_personas]

    sp_table = Table(sp_data, colWidths=[130, None],
                     style=TableStyle([
                         ('ROWBACKGROUNDS', (0, 0), (-1, -1), [ACCENT_SOFT, WHITE]),
                         ('TOPPADDING', (0, 0), (-1, -1), 6),
                         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                         ('LEFTPADDING', (0, 0), (-1, -1), 8),
                         ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                         ('LINEBELOW', (0, 0), (-1, -2), 0.3, MID_GREY),
                         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                     ]))
    story.append(sp_table)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SECTION 03 — MARKET OPPORTUNITY
    # ═══════════════════════════════════════════════════════════
    story += section_divider(3, "Market Opportunity", "TAM · SAM · SOM")

    story += slide_title("Market Opportunity",
                         "One of the largest, most underserved markets on earth.")
    story.append(Spacer(1, 4))

    # Metrics row
    metrics = [
        ("333M+", "SMBs globally\n(World Bank, 2024)"),
        ("70%", "Say they lack\nadequate support tools"),
        ("$500B", "Global SMB software\nmarket by 2027"),
        ("42%", "CAGR of AI-driven\nSMB tools segment"),
    ]
    m_cells = []
    for val, lbl in metrics:
        v_st = ParagraphStyle('mv', fontName='Helvetica-Bold', fontSize=20,
                              textColor=ELECTRIC, leading=24, alignment=TA_CENTER)
        l_st = ParagraphStyle('ml', fontName='Helvetica', fontSize=8,
                              textColor=DARK_GREY, leading=12, alignment=TA_CENTER)
        m_cells.append(Table(
            [[Paragraph(val, v_st)], [Paragraph(lbl, l_st)]],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), WHITE),
                ('BOX', (0, 0), (-1, -1), 1.5, ELECTRIC),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ])
        ))

    m_row = Table([m_cells], colWidths=[None, None, None, None],
                  style=TableStyle([
                      ('LEFTPADDING', (0, 0), (-1, -1), 4),
                      ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                  ]))
    story.append(m_row)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Market Sizing", ST['slide_h2']))
    story.append(Spacer(1, 4))

    funnel = tam_funnel()
    story.append(funnel)
    story.append(Spacer(1, 4))

    sizing_data = [
        ['', 'Size', 'Definition'],
        ['TAM', '$500B', 'All SMB software & services spend worldwide'],
        ['SAM', '$42B', 'AI-powered productivity & automation tools for SMBs'],
        ['SOM (Yr 1)', '$60M', 'English-speaking markets: US, UK, AU, India, ZA'],
        ['SOM (Yr 3)', '$1.2B', 'Expanded to 15 markets, 250K paying businesses'],
    ]
    hd_st = ParagraphStyle('th2', fontName='Helvetica-Bold', fontSize=9,
                           textColor=WHITE, alignment=TA_CENTER)
    td_st = ParagraphStyle('td2', fontName='Helvetica', fontSize=9,
                           textColor=DARK_GREY, alignment=TA_LEFT)
    tb_st = ParagraphStyle('tb2', fontName='Helvetica-Bold', fontSize=9,
                           textColor=ELECTRIC, alignment=TA_LEFT)
    tv_st = ParagraphStyle('tv2', fontName='Helvetica-Bold', fontSize=9,
                           textColor=GREEN, alignment=TA_LEFT)

    s_data = [[Paragraph(c, hd_st) for c in sizing_data[0]]]
    for row in sizing_data[1:]:
        s_data.append([
            Paragraph(row[0], tb_st),
            Paragraph(row[1], tv_st),
            Paragraph(row[2], td_st),
        ])

    s_table = Table(s_data, colWidths=[55, 65, None],
                    style=TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [ACCENT_SOFT, WHITE]),
                        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        ('LEFTPADDING', (0, 0), (-1, -1), 8),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
    story.append(s_table)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SECTION 04 — PRODUCT-MARKET FIT
    # ═══════════════════════════════════════════════════════════
    story += section_divider(4, "Product-Market Fit",
                             "Evidence that the market needs us")

    story += slide_title("Product-Market Fit",
                         "We validated demand before we wrote a line of code.")
    story.append(Spacer(1, 4))

    pmf_signals = [
        ("Customer Discovery",
         "Interviewed 120+ small business owners across retail, food service, "
         "consulting, and e-commerce. 89% said operational overwhelm was their "
         "#1 or #2 business challenge. 76% expressed willingness to pay $150+/month "
         "for an AI assistant that handled 3+ business functions."),
        ("Early Access Validation",
         "Launched a closed beta with 40 businesses. Average session length: 38 min/day. "
         "Week-4 retention: 72%. NPS score: 61 (world-class for B2B SaaS at this stage). "
         "Top 3 used agents: Sales Helper, Content Marketing, and Executive Assistant."),
        ("Trend Alignment",
         "The AI productivity tool market grew 187% YoY in 2024–2025. "
         "Google Trends shows 'AI business assistant' searches up 340% in 18 months. "
         "ChatGPT's explosive adoption proved SMBs will embrace AI when the UX is right."),
        ("Competitive Gap",
         "Zapier, HubSpot, and Monday.com serve enterprises and mid-market. "
         "No dominant player offers a modular, affordable, AI-native agent platform "
         "purpose-built for the 1–50 employee business. That gap is our runway."),
    ]

    for title, desc in pmf_signals:
        t_st = ParagraphStyle('pmft', fontName='Helvetica-Bold', fontSize=11,
                              textColor=ELECTRIC, leading=15)
        d_st = ParagraphStyle('pmfd', fontName='Helvetica', fontSize=10,
                              textColor=DARK_GREY, leading=15)
        card = Table(
            [[Paragraph(title, t_st)], [Paragraph(desc, d_st)]],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), ACCENT_SOFT),
                ('BACKGROUND', (0, 1), (-1, 1), WHITE),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LINEAFTER', (0, 0), (0, -1), 3, ELECTRIC),
                ('BOX', (0, 0), (-1, -1), 0.5, MID_GREY),
            ])
        )
        story.append(card)
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        '"Enable reduced the time I spend on non-billable admin work from 3 hours a day '
        'to under 40 minutes. I reclaimed 10+ hours per week immediately." '
        '— Beta user, events management business',
        ST['quote']))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SECTION 05 — PRODUCT & FEATURES
    # ═══════════════════════════════════════════════════════════
    story += section_divider(5, "Product & Features",
                             "What Enable actually does")

    story += slide_title("The Enable Platform",
                         "30+ specialised AI agents, one unified workspace.")
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "Enable is a modular AI-agent platform. Each agent is a domain expert "
        "trained on business best practices, able to act — not just answer. "
        "Agents collaborate with each other and with the business owner via "
        "natural language chat, automated workflows, and structured dashboards.",
        ST['body']))
    story.append(Spacer(1, 6))

    # Feature grid
    features = [
        # col 1
        [
            ("Sales Helper Agent",
             "Upload your prospect CSV. Get personalised outreach sequences, "
             "objection handling scripts, and pipeline analytics."),
            ("Content Marketing Agent",
             "Upload brand documents. Generate LinkedIn posts, emails, blogs, "
             "and ad copy across every channel in your brand voice."),
            ("Executive Assistant",
             "Manage tasks, stakeholder follow-ups, and WhatsApp communications "
             "— your AI chief of staff."),
            ("Market Research",
             "Instant competitor analysis, customer insight reports, "
             "and market trend summaries on demand."),
        ],
        # col 2
        [
            ("Supply Chain & Inventory",
             "Track stock levels, supplier relationships, and flag "
             "disruption risks before they hit your bottom line."),
            ("Invoices & Finance",
             "Automated invoice generation, payment tracking, "
             "and cash flow reports without an accountant."),
            ("Hiring & Onboarding",
             "Post roles, screen CVs, generate onboarding plans, "
             "and manage HR paperwork end-to-end."),
            ("Data Insights & Reports",
             "Connect your data sources and receive plain-English "
             "business intelligence reports automatically."),
        ],
    ]

    def feature_card(title, desc):
        t_st = ParagraphStyle('fct', fontName='Helvetica-Bold', fontSize=9,
                              textColor=WHITE, leading=13)
        d_st = ParagraphStyle('fcd', fontName='Helvetica', fontSize=8,
                              textColor=DARK_GREY, leading=12)
        return Table(
            [[Paragraph(title, t_st)], [Paragraph(desc, d_st)]],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                ('BACKGROUND', (0, 1), (-1, 1), LIGHT_GREY),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('BOX', (0, 0), (-1, -1), 0.5, ELECTRIC),
            ])
        )

    col1_cells = [[feature_card(t, d)] for t, d in features[0]]
    col2_cells = [[feature_card(t, d)] for t, d in features[1]]

    # Flatten into 2-col table
    grid_rows = []
    for (t1, d1), (t2, d2) in zip(features[0], features[1]):
        grid_rows.append([feature_card(t1, d1), feature_card(t2, d2)])

    feature_grid = Table(grid_rows, colWidths=[None, None],
                         style=TableStyle([
                             ('LEFTPADDING', (0, 0), (-1, -1), 3),
                             ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                             ('TOPPADDING', (0, 0), (-1, -1), 3),
                             ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                         ]))
    story.append(feature_grid)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Plus 22 more agents including: Orders · Notifications · Dashboards · "
                           "AI Chatbot · Community Network · Travel Agent · Team Performance · "
                           "Automation · Integration · Data Security · Monitoring · Analytics",
                           ST['small']))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SECTION 06 — DIFFERENTIATION
    # ═══════════════════════════════════════════════════════════
    story += section_divider(6, "Differentiation", "Why we win")

    story += slide_title("Why Enable Wins",
                         "We are not a chatbot. We are not a CRM. We are a business operating system.")
    story.append(Spacer(1, 4))

    # Competitive comparison table
    comp_header = ['Capability', 'Enable', 'ChatGPT', 'HubSpot', 'Zapier', 'VA / Freelancer']
    comp_rows = [
        ('Modular agent selection', '✓ Full', '✗', 'Partial', '✗', '✗'),
        ('SMB-first design', '✓ Built for SMBs', '✗ Generic', '✗ Mid-market', '✗ Tech-heavy', 'Partial'),
        ('Domain-trained AI agents', '✓ 30+ agents', '✗ One model', '✗ No AI agents', '✗ No AI', '✗'),
        ('Action (not just advice)', '✓', '✗ Text only', 'Partial', 'Partial', 'Partial'),
        ('Cross-agent collaboration', '✓ Native', '✗', '✗', 'Partial', '✗'),
        ('Affordable entry price', '✓ $79/mo', '✗ $20/mo (no agents)', '$800+/mo', '$20–$599/mo', '$500–$5K/mo'),
        ('WhatsApp / mobile native', '✓', '✗', '✗', '✗', 'Partial'),
        ('Knowledge graph memory', '✓ Per-business', '✗ Session only', '✗', '✗', '✗'),
    ]
    h_st = ParagraphStyle('ch', fontName='Helvetica-Bold', fontSize=8,
                          textColor=WHITE, alignment=TA_CENTER)
    e_st = ParagraphStyle('ce', fontName='Helvetica-Bold', fontSize=8,
                          textColor=GREEN, alignment=TA_CENTER)
    x_st = ParagraphStyle('cx', fontName='Helvetica', fontSize=8,
                          textColor=MID_GREY, alignment=TA_CENTER)
    n_st2 = ParagraphStyle('cn', fontName='Helvetica-Bold', fontSize=8,
                           textColor=NAVY, alignment=TA_LEFT)

    c_data = [[Paragraph(h, h_st) for h in comp_header]]
    for row in comp_rows:
        cells = [Paragraph(row[0], n_st2)]
        for i, val in enumerate(row[1:]):
            st_to_use = e_st if (i == 0 and '✓' in val) else x_st
            cells.append(Paragraph(val, st_to_use))
        c_data.append(cells)

    c_table = Table(c_data, colWidths=[105, 60, 52, 52, 52, 75],
                    style=TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                        ('BACKGROUND', (1, 1), (1, -1), GREEN_LT),
                        ('ROWBACKGROUNDS', (0, 1), (0, -1), [LIGHT_GREY, WHITE]),
                        ('ROWBACKGROUNDS', (2, 1), (-1, -1), [LIGHT_GREY, WHITE]),
                        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                        ('LEFTPADDING', (0, 0), (-1, -1), 5),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('BOX', (1, 1), (1, -1), 2, GREEN),
                    ]))
    story.append(c_table)
    story.append(Spacer(1, 8))

    diff_points = [
        ("Modular by design",
         "Pick exactly the agents you need. No bloated feature sets. "
         "Start with 3, scale to 30. Every agent is purpose-built, not retrofitted."),
        ("Action-oriented AI",
         "Enable's agents don't just recommend — they execute. Send the WhatsApp message. "
         "Generate the invoice. Build the content calendar. Close the loop."),
        ("Business memory",
         "Every agent learns your business through a proprietary knowledge graph. "
         "It knows your suppliers, your tone of voice, your pricing, your team — "
         "and gets smarter with every interaction."),
    ]
    for title, desc in diff_points:
        t_st = ParagraphStyle('dpt', fontName='Helvetica-Bold', fontSize=10,
                              textColor=ELECTRIC, leading=14)
        d_st = ParagraphStyle('dpd', fontName='Helvetica', fontSize=9,
                              textColor=DARK_GREY, leading=13)
        row = Table(
            [[Paragraph("→", ParagraphStyle('arr', fontName='Helvetica-Bold', fontSize=14,
                                             textColor=ACCENT, leading=18)),
              [Paragraph(title, t_st), Paragraph(desc, d_st)]]],
            colWidths=[18, None],
            style=TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ])
        )
        story.append(row)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SECTION 07 — PRICING
    # ═══════════════════════════════════════════════════════════
    story += section_divider(7, "Pricing",
                             "Simple, modular, affordable")

    story += slide_title("Pricing",
                         "Transparent monthly plans. No lock-in. Swap agents any time.")
    story.append(Spacer(1, 6))

    story.append(pricing_table())
    story.append(Spacer(1, 8))

    story.append(Paragraph("Pricing Principles", ST['slide_h2']))
    pricing_notes = [
        ("Pay only for what you use",
         "Every plan lets you choose which agents are active. Pause or swap any agent "
         "with 1 click. No wasted spend on unused modules."),
        ("Usage-based add-ons",
         "Heavy users of AI inference (e.g., content generation, data analysis) can "
         "purchase additional token packs. Base plans cover typical SMB usage comfortably."),
        ("Annual discount",
         "2 months free on any annual plan. Enterprise clients receive custom SLAs, "
         "dedicated onboarding, and white-label options."),
        ("Unit economics",
         "Blended CAC target: <$150. LTV target: $4,200 (24-month average tenure × "
         "$175 ARPU). LTV:CAC ratio target: 28:1 at scale."),
    ]
    for title, desc in pricing_notes:
        t_st = ParagraphStyle('pnt', fontName='Helvetica-Bold', fontSize=10,
                              textColor=NAVY, leading=14)
        d_st = ParagraphStyle('pnd', fontName='Helvetica', fontSize=9,
                              textColor=DARK_GREY, leading=13)
        card = Table(
            [[Paragraph(title, t_st), Paragraph(desc, d_st)]],
            colWidths=[110, None],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), ACCENT_SOFT),
                ('BACKGROUND', (1, 0), (1, 0), WHITE),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEBELOW', (0, 0), (-1, 0), 0.3, MID_GREY),
            ])
        )
        story.append(card)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SECTION 08 — POSITIONING & GROWTH STRATEGY
    # ═══════════════════════════════════════════════════════════
    story += section_divider(8, "Positioning & Growth Strategy",
                             "Go-to-market and scale")

    story += slide_title("Positioning",
                         "The AI business operating system built exclusively for small business owners.")
    story.append(Spacer(1, 4))

    pos_box_text = (
        "Enable is the only AI platform that gives a small business owner access to "
        "a full team of expert AI agents — each specialised, all collaborating — "
        "at a fraction of the cost of a single part-time employee."
    )
    pos_box_st = ParagraphStyle('pb', fontName='Helvetica-BoldOblique', fontSize=12,
                                textColor=WHITE, leading=18, alignment=TA_CENTER)
    pos_box = Table(
        [[Paragraph(pos_box_text, pos_box_st)]],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), NAVY),
            ('TOPPADDING', (0, 0), (-1, -1), 16),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ])
    )
    story.append(pos_box)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Go-to-Market Strategy", ST['slide_h2']))

    gtm_phases = [
        ("Phase 1 — Land (Q1–Q2 2026)",
         [
             "Content-led SEO: publish 3 high-quality articles/week targeting 'AI tools for small business' keywords",
             "Partnerships with SMB associations, accountants, and business coaches (referral revenue share)",
             "Free-trial funnel: 14-day full access → convert with Growth plan offer",
             "LinkedIn & Meta paid social: targeting business owners, 25–50, $100K+ revenue signals",
         ]),
        ("Phase 2 — Expand (Q3–Q4 2026)",
         [
             "Product-led growth: viral sharing of agent outputs (reports, content, analyses)",
             "App marketplace listings (Shopify, Xero, QuickBooks) for inbound discovery",
             "Launch Enable Partner Program: accountants and consultants resell at margin",
             "Geographic expansion: UK, Australia, South Africa, UAE",
         ]),
        ("Phase 3 — Scale (2027+)",
         [
             "Agent marketplace: third-party developers build and monetise custom agents",
             "Vertical-specific bundles: 'Restaurant Pack', 'Agency Pack', 'Retail Pack'",
             "White-label licensing for banks, telcos, and SMB-serving platforms",
             "Series A raise to fund international engineering and sales teams",
         ]),
    ]

    for phase_title, bullets in gtm_phases:
        t_st = ParagraphStyle('gtt', fontName='Helvetica-Bold', fontSize=10,
                              textColor=WHITE, leading=14)
        b_st = ParagraphStyle('gtb', fontName='Helvetica', fontSize=9,
                              textColor=DARK_GREY, leading=13, leftIndent=12,
                              firstLineIndent=-10)
        bullet_paras = [Paragraph(f"• {b}", b_st) for b in bullets]
        phase_card = Table(
            [[Paragraph(phase_title, t_st)],
             [bullet_paras]],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), ELECTRIC),
                ('BACKGROUND', (0, 1), (-1, 1), LIGHT_GREY),
                ('TOPPADDING', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('BOX', (0, 0), (-1, -1), 0.5, ELECTRIC),
            ])
        )
        story.append(phase_card)
        story.append(Spacer(1, 5))
    story.append(PageBreak())

    # Growth metrics projection
    story += slide_title("Growth Projections",
                         "Conservative scenario based on validated conversion benchmarks.")
    story.append(Spacer(1, 6))

    proj_header = ['Metric', 'Year 1', 'Year 2', 'Year 3']
    proj_rows = [
        ['Paying businesses', '500', '5,000', '25,000'],
        ['Avg. revenue per user/mo', '$145', '$168', '$192'],
        ['Annual Recurring Revenue', '$870K', '$10.1M', '$57.6M'],
        ['Gross Margin (target)', '72%', '76%', '80%'],
        ['Net Revenue Retention', '105%', '112%', '118%'],
        ['Countries', '2', '6', '15'],
    ]
    h_st2 = ParagraphStyle('ph2', fontName='Helvetica-Bold', fontSize=9,
                           textColor=WHITE, alignment=TA_CENTER)
    r_st = ParagraphStyle('pr', fontName='Helvetica', fontSize=9,
                          textColor=DARK_GREY, alignment=TA_CENTER)
    rb_st = ParagraphStyle('prb', fontName='Helvetica-Bold', fontSize=9,
                           textColor=NAVY, alignment=TA_LEFT)
    rg_st = ParagraphStyle('prg', fontName='Helvetica-Bold', fontSize=9,
                           textColor=GREEN, alignment=TA_CENTER)

    p_data = [[Paragraph(h, h_st2) for h in proj_header]]
    for i, row in enumerate(proj_rows):
        bg_highlight = (i == 2)  # ARR row
        cells = [Paragraph(row[0], rb_st)]
        for val in row[1:]:
            cells.append(Paragraph(val, rg_st if bg_highlight else r_st))
        p_data.append(cells)

    p_table = Table(p_data, colWidths=[130, 75, 75, 75],
                    style=TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [ACCENT_SOFT, WHITE]),
                        ('BACKGROUND', (0, 3), (-1, 3), GREEN_LT),
                        ('GRID', (0, 0), (-1, -1), 0.4, MID_GREY),
                        ('TOPPADDING', (0, 0), (-1, -1), 7),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                        ('LEFTPADDING', (0, 0), (-1, -1), 8),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
    story.append(p_table)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SECTION 09 — THE ASK
    # ═══════════════════════════════════════════════════════════
    story += section_divider(9, "The Ask", "What we need & what you get")

    story += slide_title("Investment Ask",
                         "We are raising a pre-seed round to reach 500 paying businesses.")
    story.append(Spacer(1, 6))

    ask_items = [
        ("Raising", "$750,000 USD", "Pre-seed round"),
        ("Instrument", "SAFE Note", "20% discount, $5M cap"),
        ("Use of funds", "Product 40% · GTM 35% · Ops 25%", "18-month runway"),
        ("Milestone", "500 businesses · $870K ARR", "Series A ready"),
    ]
    for label, value, sub in ask_items:
        l_st = ParagraphStyle('al', fontName='Helvetica', fontSize=10,
                              textColor=DARK_GREY, leading=14)
        v_st = ParagraphStyle('av', fontName='Helvetica-Bold', fontSize=14,
                              textColor=ELECTRIC, leading=18)
        s_st = ParagraphStyle('as', fontName='Helvetica', fontSize=9,
                              textColor=MID_GREY, leading=13)
        row = Table(
            [[Paragraph(label, l_st), [Paragraph(value, v_st), Paragraph(sub, s_st)]]],
            colWidths=[100, None],
            style=TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), ACCENT_SOFT),
                ('BACKGROUND', (1, 0), (1, 0), WHITE),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LINEBELOW', (0, 0), (-1, 0), 0.5, MID_GREY),
            ])
        )
        story.append(row)
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 10))
    story.append(Paragraph("What Investors Get", ST['slide_h2']))
    what_you_get = [
        "First-mover advantage in the AI agent platform for SMBs — "
        "a $42B SAM with no dominant player",
        "A capital-efficient SaaS model with strong unit economics "
        "(LTV:CAC target 28:1) and 80%+ gross margins at scale",
        "A team that has already shipped 30+ working AI agents and "
        "validated PMF with real paying customers",
        "A platform business with network effects — "
        "each new agent type increases platform value for all users",
        "Clear path to Series A at $10M ARR with multiple expansion "
        "vectors (verticals, geographies, marketplace, white-label)",
    ]
    for item in what_you_get:
        story.append(Paragraph(f"• {item}", ST['bullet']))
    story.append(Spacer(1, 10))

    story.append(hline(ELECTRIC, 2))
    cta_st = ParagraphStyle('cta', fontName='Helvetica-Bold', fontSize=14,
                            textColor=NAVY, leading=20, alignment=TA_CENTER, spaceBefore=8)
    contact_st = ParagraphStyle('contact', fontName='Helvetica', fontSize=10,
                                textColor=DARK_GREY, leading=15, alignment=TA_CENTER)
    story.append(Paragraph("Ready to enable every business on earth?", cta_st))
    story.append(Paragraph(
        "Contact: hello@enableyou.co  ·  enableyou.co  ·  April 2026",
        contact_st))

    # ── Page templates ──────────────────────────────────────────
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

    # We need to track which pages use which background.
    # Rebuild as a BaseDocTemplate with onPage callbacks.

    doc.build(story,
              onFirstPage=draw_cover_background,
              onLaterPages=draw_slide_background)

    print(f"✓ Pitch deck saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    build()
