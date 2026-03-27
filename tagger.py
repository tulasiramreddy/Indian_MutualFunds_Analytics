"""Sector/style tag assignment for Indian equity mutual funds."""

import logging
from db import get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag taxonomy
# ---------------------------------------------------------------------------

# SEBI category → style tag (non-sectoral funds)
CATEGORY_TAG = {
    "Equity Scheme - Large Cap Fund":       "Large Cap",
    "Equity Scheme - Mid Cap Fund":         "Mid Cap",
    "Equity Scheme - Small Cap Fund":       "Small Cap",
    "Equity Scheme - Large & Mid Cap Fund": "Large & Mid Cap",
    "Equity Scheme - Multi Cap Fund":       "Multi Cap",
    "Equity Scheme - Flexi Cap Fund":       "Flexi Cap",
    "Equity Scheme - Focused Fund":         "Focused",
    "Equity Scheme - Value Fund":           "Value",
    "Equity Scheme - Contra Fund":          "Contra",
    "Equity Scheme - ELSS":                 "ELSS",
    "ELSS":                                 "ELSS",
}

# For sectoral/thematic funds: ordered list of (tag, [keywords]).
# First matching rule wins — ORDER MATTERS.
# More specific / easily confused rules go first.
THEME_RULES: list[tuple[str, list[str]]] = [

    # ── Banking / Finance ────────────────────────────────────────────────────
    ("Banking & Financial Services", [
        "banking", "financial services", "bfsi", "fin service",
    ]),

    # ── Technology / IT ──────────────────────────────────────────────────────
    ("Technology", [
        "technology", "digital india", "digital bharat",
        "teck fund", "tech fund", " it fund",
        "information technology", "innovation in india",
    ]),

    # ── Healthcare / Pharma ──────────────────────────────────────────────────
    ("Healthcare", [
        "pharma", "healthcare", "health and wellness",
        "health care", "diagnostic", "wellness",
    ]),

    # ── Infrastructure ───────────────────────────────────────────────────────
    ("Infrastructure", [
        "infrastructure", "t.i.g.e.r", "build india",
        "india tiger", "power & infra", "power and infra",
        "roads", "railways",
    ]),

    # ── Consumption / FMCG ──────────────────────────────────────────────────
    ("Consumption & FMCG", [
        "consumption", "consumer trends", "consumer opportunities",
        "consumer value", "consumer durable",
        "fmcg", "india consumer", "great consumer",
    ]),

    # ── Manufacturing ────────────────────────────────────────────────────────
    ("Manufacturing", [
        "manufacturing", "manufacture in india",
        "make in india",
    ]),

    # ── Energy / Resources ───────────────────────────────────────────────────
    ("Energy & Resources", [
        "energy", "natural resources", "resources & energy",
        "resources and energy", "resources and new energy",
        "commodity", "commodities", "agri plan",
        "new energy", "green energy", "clean energy",
    ]),

    # ── PSU ──────────────────────────────────────────────────────────────────
    ("PSU", [
        "psu fund", " psu ", "psu&", "public sector",
    ]),

    # ── Defence ──────────────────────────────────────────────────────────────
    ("Defence", [
        "defence", "defense", "strategic",
    ]),

    # ── MNC ──────────────────────────────────────────────────────────────────
    ("MNC", [
        " mnc fund", "-mnc fund", " mnc-", "multinational",
    ]),

    # ── ESG / Ethical ────────────────────────────────────────────────────────
    ("ESG & Ethical", [
        "esg", "ethical fund", "best-in-class", "best in class",
        "exclusionary strategy", "integration strategy",
    ]),

    # ── Business Cycle — BEFORE Quant to avoid quant AMC misclassification ──
    ("Business Cycle", [
        "business cycle",
    ]),

    # ── Momentum — BEFORE Quant for same reason ──────────────────────────────
    ("Momentum", [
        "momentum fund", "active momentum", "momentum etf",
    ]),

    # ── Dividend Yield ───────────────────────────────────────────────────────
    ("Dividend Yield", [
        "dividend yield", "dividend opportunity",
        "dividends for prosperity",
    ]),

    # ── Quant & Factor ───────────────────────────────────────────────────────
    # NOTE: "quant " (trailing space) removed — it matches the "quant" AMC name
    # and misclassifies quant Business Cycle, quant Momentum, etc.
    # "quant fund" and "quantamental" are sufficient for real quant funds.
    ("Quant & Factor", [
        "quant fund", "quantamental",
        "multi-factor", "multifactor",
        "equity minimum variance", "factor fund",
        "smart beta", "alpha fund", "alpha strategy",
        "low volatility fund", "minimum variance",
    ]),

    # ── International ────────────────────────────────────────────────────────
    ("International", [
        "international equity", "global equity", "global fund",
        "japan equity", "us equity", "taiwan equity",
        "global agri", "asian equity", "us bluechip",
        "emerging market", "overseas fund",
        "china equity", "europe equity",
    ]),

    # ── Housing / Realty ─────────────────────────────────────────────────────
    ("Housing", [
        "housing", "real estate", "realty",
    ]),

    # ── Auto & Logistics ─────────────────────────────────────────────────────
    ("Auto & Logistics", [
        "automotive", "transportation and logistics",
        "transportation & logistics", "mobility",
        "auto and mobility",
    ]),

    # ── Conglomerate ─────────────────────────────────────────────────────────
    ("Conglomerate", [
        "conglomerate", "business conglomerates",
        "india conglomerate",
    ]),

    # ── Innovation ───────────────────────────────────────────────────────────
    ("Innovation", [
        "innovation fund", "innovative opportunities",
        "innovation opportunities", "innovation & opportunities",
        "pioneer fund", "new age",
    ]),

    # ── Quality ──────────────────────────────────────────────────────────────
    ("Quality", [
        "quality fund", "quality equity", "quality value",
        "quality and value",
    ]),

    # ── Rural & Agri ─────────────────────────────────────────────────────────
    ("Rural & Agri", [
        "rural opportunities", "rural and consumption",
        "export opportunities", "agri ", "agriculture",
        "india growth opportunities",
    ]),

    # ── Services ─────────────────────────────────────────────────────────────
    ("Services", [
        "services fund", "services opportunities", " services fund",
        "india services",
    ]),

    # ── IPO ──────────────────────────────────────────────────────────────────
    ("IPO", [
        "recently listed ipo", "ipo fund", "new listings",
    ]),

    # ── Special Situations ───────────────────────────────────────────────────
    # Catch-all for rotation/opportunity funds — keep last among named themes
    ("Special Situations", [
        "special opportunities", "opportunities fund",
        "sector rotation", "special situation",
        "value opportunities",
    ]),
]

SECTORAL_CATEGORY = "Equity Scheme - Sectoral/ Thematic"


def assign_tags(scheme_name: str, scheme_category: str) -> list[str]:
    """Return the list of tags for a fund.

    Most funds get exactly one tag. A fund can appear under multiple tags
    if it genuinely fits more than one (rare).
    """
    tags: list[str] = []

    # Style tag from SEBI category
    style = CATEGORY_TAG.get(scheme_category)
    if style:
        tags.append(style)
        return tags  # non-sectoral: category is sufficient

    # Sectoral/thematic: match on fund name
    if scheme_category == SECTORAL_CATEGORY:
        name_lower = scheme_name.lower()
        for tag, keywords in THEME_RULES:
            if any(kw in name_lower for kw in keywords):
                tags.append(tag)
                break   # one primary theme per fund

    return tags if tags else ["Other"]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

TAG_SCHEMA = """
CREATE TABLE IF NOT EXISTS fund_tags (
    scheme_code INTEGER NOT NULL,
    tag         TEXT    NOT NULL,
    PRIMARY KEY (scheme_code, tag),
    FOREIGN KEY (scheme_code) REFERENCES funds(scheme_code)
);
CREATE INDEX IF NOT EXISTS idx_fund_tags_tag  ON fund_tags(tag);
CREATE INDEX IF NOT EXISTS idx_fund_tags_code ON fund_tags(scheme_code);
"""


def init_tags_table():
    """Create fund_tags table and indexes if they don't exist."""
    with get_connection() as conn:
        conn.executescript(TAG_SCHEMA)


def auto_tag_all_funds() -> dict[str, int]:
    """Assign tags to every fund in the DB. Returns {tag: count}."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT scheme_code, scheme_name, scheme_category FROM funds"
        ).fetchall()

        conn.execute("DELETE FROM fund_tags")

        records = []
        for code, name, cat in rows:
            for tag in assign_tags(name, cat or ""):
                records.append((code, tag))

        conn.executemany(
            "INSERT OR IGNORE INTO fund_tags (scheme_code, tag) VALUES (?, ?)",
            records,
        )

    # Count per tag
    counts: dict[str, int] = {}
    for _, tag in records:
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def get_all_tags() -> list[tuple[str, int]]:
    """Return [(tag, fund_count)] sorted by tag name."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT tag, COUNT(*) as n
               FROM fund_tags
               GROUP BY tag
               ORDER BY tag"""
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_funds_by_tag(tag: str) -> list[dict]:
    """Return list of fund dicts for a given tag."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT f.scheme_code, f.scheme_name, f.scheme_category, f.fund_house
               FROM funds f
               JOIN fund_tags t ON f.scheme_code = t.scheme_code
               WHERE t.tag = ?
               ORDER BY f.scheme_name""",
            (tag,),
        ).fetchall()
    return [
        {"scheme_code": r[0], "scheme_name": r[1],
         "scheme_category": r[2], "fund_house": r[3]}
        for r in rows
    ]


def get_fund_tags(scheme_code: int) -> list[str]:
    """Return all tags for a specific fund."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT tag FROM fund_tags WHERE scheme_code = ? ORDER BY tag",
            (scheme_code,),
        ).fetchall()
    return [r[0] for r in rows]


def set_fund_tags(scheme_code: int, tags: list[str]):
    """Replace all tags for a fund."""
    with get_connection() as conn:
        conn.execute("DELETE FROM fund_tags WHERE scheme_code = ?", (scheme_code,))
        conn.executemany(
            "INSERT OR IGNORE INTO fund_tags (scheme_code, tag) VALUES (?, ?)",
            [(scheme_code, t) for t in tags],
        )
