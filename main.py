"""
====================================================
 Airbnb Host Accounting Tool — FastAPI Backend
 (with SQLite + SQLAlchemy persistence)
====================================================

HOW TO RUN:
-----------
1. Install dependencies:
       pip install fastapi uvicorn python-multipart sqlalchemy httpx reportlab

2. Start the server:
       uvicorn main:app --reload

3. Open the interactive API docs in your browser:
       http://127.0.0.1:8000/docs

4. A file called airbnb_accounting.db will be created automatically
   in the same folder as main.py on first startup.
"""

import csv
import io
import json
import os
import httpx
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── SQLAlchemy imports ────────────────────────────────────────────────────────
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean,
    ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# ── Database setup ────────────────────────────────────────────────────────────
#
# SQLite stores everything in a single file beside main.py.
# connect_args={"check_same_thread": False} is required for SQLite when
# used with FastAPI because requests run on different threads.
DATABASE_URL = "sqlite:///./airbnb_accounting.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# SessionLocal is a factory: call SessionLocal() to get a new DB session.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base is the parent class all ORM models inherit from.
Base = declarative_base()


# ── ORM Models (Database Tables) ──────────────────────────────────────────────

class AirbnbTransaction(Base):
    """
    One row per parsed Airbnb payout CSV record.
    All uploads are cleared and re-inserted so the table always reflects
    the most recently uploaded file.
    """
    __tablename__ = "airbnb_transactions"

    id             = Column(Integer, primary_key=True, index=True)
    row_number     = Column(Integer)          # 1-based position in the CSV
    date           = Column(String)           # ISO-8601 string e.g. "2024-03-01"
    listing        = Column(String)           # Property / listing name
    payout_amount  = Column(Float)            # Net payout (what hits the bank)
    cleaning_fee   = Column(Float)
    airbnb_service_fee = Column(Float)
    taxes_withheld = Column(Float)
    raw_row        = Column(Text)             # JSON-encoded original CSV row

    # One AirbnbTransaction can appear in at most one Match
    match = relationship("Match", back_populates="airbnb_transaction", uselist=False)


class BankTransaction(Base):
    """
    One row per parsed bank-statement CSV record.
    All uploads are cleared and re-inserted on each /upload-bank call.
    """
    __tablename__ = "bank_transactions"

    id          = Column(Integer, primary_key=True, index=True)
    row_number  = Column(Integer)
    date        = Column(String)
    description = Column(String)
    amount      = Column(Float)
    raw_row     = Column(Text)             # JSON-encoded original CSV row

    match = relationship("Match", back_populates="bank_transaction", uselist=False)


class Match(Base):
    """
    Links one AirbnbTransaction to one BankTransaction.
    is_manual=True  → user confirmed via the "Match manually" UI
    is_manual=False → auto-matched by the reconciliation algorithm
    """
    __tablename__ = "matches"

    id             = Column(Integer, primary_key=True, index=True)
    airbnb_id      = Column(Integer, ForeignKey("airbnb_transactions.id"), unique=True)
    bank_id        = Column(Integer, ForeignKey("bank_transactions.id"),   unique=True)
    is_manual      = Column(Boolean, default=False)

    airbnb_transaction = relationship("AirbnbTransaction", back_populates="match")
    bank_transaction   = relationship("BankTransaction",   back_populates="match")


# Create all tables on startup (no-op if they already exist)
Base.metadata.create_all(bind=engine)


# ── Dependency: database session per request ───────────────────────────────────
#
# FastAPI calls get_db() for every request that declares `db: Session = Depends(get_db)`.
# The `finally` block guarantees the session is closed even if an exception occurs.

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Airbnb Host Accounting API",
    description="Upload Airbnb earnings or bank-statement CSVs and get structured JSON back.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory mirrors (kept for backward-compat with non-DB helpers) ──────────
#
# The AI insights, expense categorisation, and Schedule E endpoints were built
# against in-memory lists.  We keep these mirrors populated after each upload
# so those endpoints continue to work without any changes.
airbnb_data: list[dict] = []
bank_data:   list[dict] = []


# ── CSV parsing helpers ───────────────────────────────────────────────────────

def parse_amount(value: str) -> Optional[float]:
    if not value or value.strip() == "":
        return None
    cleaned = value.strip().replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_date(value: str) -> Optional[str]:
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value.strip()


def read_csv(file_bytes: bytes) -> list[dict]:
    text   = file_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows   = []
    for row in reader:
        normalised = {k.strip().lower(): v.strip() for k, v in row.items()}
        rows.append(normalised)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty or has no data rows.")
    return rows


def find_column(row: dict, candidates: list[str]) -> Optional[str]:
    for key in row:
        for candidate in candidates:
            if candidate in key:
                return key
    return None


# ── Helper: ORM row → plain dict (for JSON responses) ────────────────────────

def airbnb_row_to_dict(row: AirbnbTransaction) -> dict:
    """Convert an AirbnbTransaction ORM object to the same dict shape the
    frontend and downstream helpers already expect."""
    return {
        "row_number":         row.row_number,
        "date":               row.date,
        "listing":            row.listing,
        "payout_amount":      row.payout_amount,
        "cleaning_fee":       row.cleaning_fee,
        "airbnb_service_fee": row.airbnb_service_fee,
        "taxes_withheld":     row.taxes_withheld,
        "raw_row":            json.loads(row.raw_row) if row.raw_row else {},
    }


def bank_row_to_dict(row: BankTransaction) -> dict:
    """Convert a BankTransaction ORM object to the same dict shape used
    throughout the existing reconciliation and categorisation logic."""
    return {
        "row_number":  row.row_number,
        "date":        row.date,
        "description": row.description,
        "amount":      row.amount,
        "raw_row":     json.loads(row.raw_row) if row.raw_row else {},
    }


# ── Endpoint 1: Upload Airbnb CSV ─────────────────────────────────────────────

@app.post("/upload-airbnb", summary="Upload an Airbnb earnings CSV", tags=["Airbnb"])
async def upload_airbnb(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Parse an Airbnb earnings CSV and persist each row to AirbnbTransaction.
    Previous rows and their associated matches are deleted first so the table
    always reflects the latest upload.
    """
    global airbnb_data

    raw_bytes = await file.read()
    rows = read_csv(raw_bytes)

    # ── Clear old data ────────────────────────────────────────────────────────
    # Delete matches first (foreign key constraint), then transactions.
    db.query(Match).filter(
        Match.airbnb_id.in_(
            db.query(AirbnbTransaction.id)
        )
    ).delete(synchronize_session=False)
    db.query(AirbnbTransaction).delete()
    db.commit()

    parsed_records = []

    for i, row in enumerate(rows):
        date_col     = find_column(row, ["date"])
        listing_col  = find_column(row, ["listing", "property", "description"])
        payout_col   = find_column(row, ["payout", "amount", "earnings", "gross"])
        cleaning_col = find_column(row, ["cleaning"])
        service_col  = find_column(row, ["service fee", "host fee", "airbnb fee"])
        tax_col      = find_column(row, ["tax", "withheld"])

        record_dict = {
            "row_number":         i + 1,
            "date":               parse_date(row[date_col])         if date_col     else None,
            "listing":            row[listing_col]                   if listing_col  else None,
            "payout_amount":      parse_amount(row[payout_col])      if payout_col   else None,
            "cleaning_fee":       parse_amount(row[cleaning_col])    if cleaning_col else None,
            "airbnb_service_fee": parse_amount(row[service_col])     if service_col  else None,
            "taxes_withheld":     parse_amount(row[tax_col])         if tax_col      else None,
            "raw_row":            row,
        }
        parsed_records.append(record_dict)

        # Persist to DB
        db_row = AirbnbTransaction(
            row_number         = record_dict["row_number"],
            date               = record_dict["date"],
            listing            = record_dict["listing"],
            payout_amount      = record_dict["payout_amount"],
            cleaning_fee       = record_dict["cleaning_fee"],
            airbnb_service_fee = record_dict["airbnb_service_fee"],
            taxes_withheld     = record_dict["taxes_withheld"],
            raw_row            = json.dumps(row),
        )
        db.add(db_row)

    db.commit()

    # Keep in-memory mirror in sync
    airbnb_data = parsed_records

    total_payout   = sum(r["payout_amount"]     or 0 for r in parsed_records)
    total_cleaning = sum(r["cleaning_fee"]      or 0 for r in parsed_records)
    total_fees     = sum(r["airbnb_service_fee"] or 0 for r in parsed_records)
    total_taxes    = sum(r["taxes_withheld"]    or 0 for r in parsed_records)

    return {
        "status":  "success",
        "message": f"Parsed and saved {len(parsed_records)} Airbnb transaction(s).",
        "summary": {
            "total_rows":               len(parsed_records),
            "total_payout_amount":      round(total_payout,   2),
            "total_cleaning_fees":      round(total_cleaning, 2),
            "total_airbnb_service_fees":round(total_fees,     2),
            "total_taxes_withheld":     round(total_taxes,    2),
            "net_income_estimate":      round(total_payout - total_fees - total_taxes, 2),
        },
        "transactions": parsed_records,
    }


# ── Endpoint 2: Upload bank CSV ───────────────────────────────────────────────

@app.post("/upload-bank", summary="Upload a bank transactions CSV", tags=["Bank"])
async def upload_bank(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Parse a bank-statement CSV and persist each row to BankTransaction.
    Previous rows and their matches are deleted first.
    """
    global bank_data

    raw_bytes = await file.read()
    rows = read_csv(raw_bytes)

    # Clear old bank data and associated matches
    db.query(Match).filter(
        Match.bank_id.in_(
            db.query(BankTransaction.id)
        )
    ).delete(synchronize_session=False)
    db.query(BankTransaction).delete()
    db.commit()

    parsed_records = []

    for i, row in enumerate(rows):
        date_col   = find_column(row, ["date"])
        desc_col   = find_column(row, ["description", "memo", "narration", "details", "particulars"])
        amount_col = find_column(row, ["amount", "credit", "debit", "transaction"])

        record_dict = {
            "row_number":  i + 1,
            "date":        parse_date(row[date_col])     if date_col   else None,
            "description": row[desc_col]                 if desc_col   else None,
            "amount":      parse_amount(row[amount_col]) if amount_col else None,
            "raw_row":     row,
        }
        parsed_records.append(record_dict)

        db_row = BankTransaction(
            row_number  = record_dict["row_number"],
            date        = record_dict["date"],
            description = record_dict["description"],
            amount      = record_dict["amount"],
            raw_row     = json.dumps(row),
        )
        db.add(db_row)

    db.commit()

    # Keep in-memory mirror in sync
    bank_data = parsed_records

    credits = [r["amount"] for r in parsed_records if (r["amount"] or 0) > 0]
    debits  = [r["amount"] for r in parsed_records if (r["amount"] or 0) < 0]

    return {
        "status":  "success",
        "message": f"Parsed and saved {len(parsed_records)} bank transaction(s).",
        "summary": {
            "total_rows":    len(parsed_records),
            "total_credits": round(sum(credits), 2),
            "total_debits":  round(sum(debits),  2),
            "net_cash_flow": round(sum(credits) + sum(debits), 2),
        },
        "transactions": parsed_records,
    }


# ── Reconciliation helpers ─────────────────────────────────────────────────────

DATE_TOLERANCE_DAYS = 3
AMOUNT_TOLERANCE    = 0.01


def to_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def dates_within_tolerance(date_a: Optional[str], date_b: Optional[str]) -> bool:
    dt_a, dt_b = to_date(date_a), to_date(date_b)
    if dt_a is None or dt_b is None:
        return False
    return abs((dt_a - dt_b).days) <= DATE_TOLERANCE_DAYS


def amounts_match(airbnb_amount: Optional[float], bank_amount: Optional[float]) -> bool:
    if airbnb_amount is None or bank_amount is None:
        return False
    return abs(abs(airbnb_amount) - abs(bank_amount)) <= AMOUNT_TOLERANCE


def find_best_bank_match(airbnb_record: dict, available_bank_records: list[dict]) -> Optional[dict]:
    airbnb_date = to_date(airbnb_record.get("date"))
    candidates  = []
    for bank_record in available_bank_records:
        if not amounts_match(airbnb_record.get("payout_amount"), bank_record.get("amount")):
            continue
        if not dates_within_tolerance(airbnb_record.get("date"), bank_record.get("date")):
            continue
        bank_date = to_date(bank_record.get("date"))
        day_gap   = abs((airbnb_date - bank_date).days) if (airbnb_date and bank_date) else DATE_TOLERANCE_DAYS
        candidates.append((day_gap, bank_record))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ── Shared reconciliation logic ───────────────────────────────────────────────
#
# run_reconciliation() contains ALL matching logic so it can be called from
# both the /reconcile endpoint and /match-manual (which re-runs reconciliation
# immediately after saving a manual match, so the UI updates in one round-trip).

def run_reconciliation(db: Session) -> dict:
    """
    Core reconciliation function. Reads all Airbnb and bank rows from the DB,
    honours manual matches first, then auto-matches the remainder by amount +
    date proximity. Persists auto matches to the Match table and returns the
    full response dict (identical shape to the old /reconcile response).

    Raises HTTPException(400) if either dataset is missing.
    """

    # ── Load all rows from DB as plain dicts ──────────────────────────────────
    airbnb_rows = [airbnb_row_to_dict(r) for r in db.query(AirbnbTransaction).all()]
    bank_rows   = [bank_row_to_dict(r)   for r in db.query(BankTransaction).all()]

    if not airbnb_rows:
        raise HTTPException(status_code=400, detail="No Airbnb data found. Please call /upload-airbnb first.")
    if not bank_rows:
        raise HTTPException(status_code=400, detail="No bank data found. Please call /upload-bank first.")

    # ── Load existing manual matches from DB ──────────────────────────────────
    # Manual matches survive server restarts because they live in the DB, not RAM.
    manual_db_matches = db.query(Match).filter(Match.is_manual == True).all()

    # Build lookup: airbnb row_number → bank row_number
    manually_matched_airbnb_rows = {
        m.airbnb_transaction.row_number: m.bank_transaction.row_number
        for m in manual_db_matches
    }
    manually_matched_bank_rows = set(manually_matched_airbnb_rows.values())

    # ── Delete old AUTO matches — manual matches are untouched ────────────────
    db.query(Match).filter(Match.is_manual == False).delete()
    db.commit()

    # Bank rows available for auto-matching (manual claims excluded)
    remaining_bank = [r for r in bank_rows if r["row_number"] not in manually_matched_bank_rows]

    # DB id lookups needed to create new Match rows
    airbnb_id_by_row = {r.row_number: r.id for r in db.query(AirbnbTransaction).all()}
    bank_id_by_row   = {r.row_number: r.id for r in db.query(BankTransaction).all()}

    matches          = []
    unmatched_airbnb = []

    for airbnb_record in airbnb_rows:
        if airbnb_record.get("payout_amount") is None:
            continue

        airbnb_row_num = airbnb_record["row_number"]

        # ── Step 1: honour manual match if one exists ─────────────────────────
        if airbnb_row_num in manually_matched_airbnb_rows:
            bank_row_num = manually_matched_airbnb_rows[airbnb_row_num]
            bank_record  = next((r for r in bank_rows if r["row_number"] == bank_row_num), None)
            matches.append({
                "airbnb_payout":           airbnb_record,
                "matched_bank_transaction": bank_record,
                "status":                  "matched",
                "match_type":              "manual",
                "date_gap_days":           None,
            })
            continue

        # ── Step 2: auto-match by amount + date proximity ─────────────────────
        best_match = find_best_bank_match(airbnb_record, remaining_bank)

        if best_match:
            dt_a    = to_date(airbnb_record.get("date"))
            dt_b    = to_date(best_match.get("date"))
            day_gap = abs((dt_a - dt_b).days) if (dt_a and dt_b) else None

            matches.append({
                "airbnb_payout":           airbnb_record,
                "matched_bank_transaction": best_match,
                "status":                  "matched",
                "match_type":              "auto",
                "date_gap_days":           day_gap,
            })

            db.add(Match(
                airbnb_id = airbnb_id_by_row[airbnb_row_num],
                bank_id   = bank_id_by_row[best_match["row_number"]],
                is_manual = False,
            ))
            remaining_bank.remove(best_match)

        else:
            unmatched_airbnb.append({
                "airbnb_payout":           airbnb_record,
                "matched_bank_transaction": None,
                "status":                  "unmatched",
            })

    db.commit()

    # ── Unmatched bank credits ────────────────────────────────────────────────
    matched_bank_row_nums = (
        manually_matched_bank_rows |
        {m["matched_bank_transaction"]["row_number"]
         for m in matches if m["match_type"] == "auto" and m["matched_bank_transaction"]}
    )
    unmatched_bank = [
        {
            "bank_transaction": r,
            "status":           "unmatched_bank",
            "note":             "Credit not matched to any Airbnb payout on record.",
        }
        for r in bank_rows
        if (r.get("amount") or 0) > 0 and r["row_number"] not in matched_bank_row_nums
    ]

    # ── Build and return summary ───────────────────────────────────────────────
    total_matched_value   = sum(m["airbnb_payout"].get("payout_amount") or 0 for m in matches)
    total_unmatched_value = sum(u["airbnb_payout"].get("payout_amount") or 0 for u in unmatched_airbnb)
    manual_count          = sum(1 for m in matches if m["match_type"] == "manual")

    # ── Tax Confidence Score ───────────────────────────────────────────────────
    # Measures how "audit-ready" the reconciliation is on a 0–100 scale.
    # Each unmatched Airbnb payout is a bigger concern (5 pts) than an
    # unmatched bank credit (3 pts) because a missing payout means potential
    # unreported income — the more serious tax risk.
    n_unmatched_airbnb = len(unmatched_airbnb)
    n_unmatched_bank   = len(unmatched_bank)

    raw_score = 100 - (n_unmatched_airbnb * 5) - (n_unmatched_bank * 3)
    score     = max(0, raw_score)   # floor at 0

    # Status thresholds
    if score >= 90:
        status = "Tax Ready"
    elif score >= 70:
        status = "Needs Review"
    else:
        status = "High Risk"

    # Human-readable issue messages — only added when the count is non-zero
    # so a clean reconciliation returns an empty issues list.
    issues = []
    if n_unmatched_airbnb > 0:
        noun = "payout" if n_unmatched_airbnb == 1 else "payouts"
        issues.append(
            f"{n_unmatched_airbnb} Airbnb {noun} are not matched to bank deposits"
        )
    if n_unmatched_bank > 0:
        noun = "transaction" if n_unmatched_bank == 1 else "transactions"
        issues.append(
            f"{n_unmatched_bank} bank {noun} are not linked to any Airbnb payout"
        )

    return {
        "status": "success",
        "summary": {
            "total_airbnb_payouts":   len(airbnb_rows),
            "matched_count":          len(matches),
            "manual_match_count":     manual_count,
            "auto_match_count":       len(matches) - manual_count,
            "unmatched_airbnb_count": n_unmatched_airbnb,
            "unmatched_bank_count":   n_unmatched_bank,
            "total_matched_value":    round(total_matched_value,   2),
            "total_unmatched_value":  round(total_unmatched_value, 2),
            "match_rate_pct":         round(100 * len(matches) / len(airbnb_rows), 1) if airbnb_rows else 0,
        },
        # ── NEW: confidence object ────────────────────────────────────────────
        # Appended after summary so existing frontend code that reads summary
        # fields is completely unaffected. Frontend can opt-in to displaying
        # this block without any required changes.
        "confidence": {
            "score":                   score,
            "status":                  status,
            "matched_transactions":    len(matches),
            "unmatched_transactions":  n_unmatched_airbnb + n_unmatched_bank,
            "issues":                  issues,
        },
        "matches":          matches,
        "unmatched_airbnb": unmatched_airbnb,
        "unmatched_bank":   unmatched_bank,
    }


# ── Endpoint 3: Reconcile ─────────────────────────────────────────────────────

@app.post(
    "/reconcile",
    summary="Match Airbnb payouts to bank transactions",
    tags=["Reconciliation"],
)
def reconcile(db: Session = Depends(get_db)):
    """
    Delegates entirely to run_reconciliation() so the logic lives in one place.
    Response format is unchanged — the frontend sees exactly the same shape.
    """
    return run_reconciliation(db)


# ── Endpoint: Manual match ────────────────────────────────────────────────────

class ManualMatchRequest(BaseModel):
    airbnb_row: int   # row_number of the Airbnb payout
    bank_row:   int   # row_number of the bank transaction


@app.post(
    "/match-manual",
    summary="Save a user-confirmed manual match and return updated reconciliation",
    tags=["Reconciliation"],
)
def match_manual(body: ManualMatchRequest, db: Session = Depends(get_db)):
    """
    Persist a manual match to the Match table with is_manual=True, then
    IMMEDIATELY re-runs reconciliation and returns the full reconciliation
    response so the frontend can update match rate, unmatched lists, and
    totals in a single round-trip — no separate /reconcile call needed.

    If this Airbnb row was already matched (manually or automatically),
    the old match is replaced (last-write-wins).
    """

    # ── Validate: both row numbers must exist in the DB ───────────────────────
    airbnb_db = db.query(AirbnbTransaction).filter(
        AirbnbTransaction.row_number == body.airbnb_row
    ).first()
    bank_db = db.query(BankTransaction).filter(
        BankTransaction.row_number == body.bank_row
    ).first()

    if not airbnb_db:
        raise HTTPException(status_code=404, detail=f"No Airbnb record with row_number={body.airbnb_row}.")
    if not bank_db:
        raise HTTPException(status_code=404, detail=f"No bank record with row_number={body.bank_row}.")

    # ── Remove any pre-existing match on either side (last-write-wins) ────────
    db.query(Match).filter(Match.airbnb_id == airbnb_db.id).delete()
    db.query(Match).filter(Match.bank_id   == bank_db.id).delete()
    db.commit()

    # ── Persist the new manual match ──────────────────────────────────────────
    db.add(Match(
        airbnb_id = airbnb_db.id,
        bank_id   = bank_db.id,
        is_manual = True,
    ))
    db.commit()

    # ── Re-run full reconciliation and return its result ──────────────────────
    # This means the frontend gets an updated match rate, unmatched lists, and
    # totals immediately — no need to call /reconcile as a second request.
    reconciliation_result = run_reconciliation(db)

    # Inject a confirmation message so the caller knows the save succeeded
    reconciliation_result["manual_match_saved"] = {
        "airbnb_row": body.airbnb_row,
        "bank_row":   body.bank_row,
        "message":    f"Manual match saved: Airbnb row {body.airbnb_row} → bank row {body.bank_row}.",
    }

    return reconciliation_result


# ── Financial breakdown helper ────────────────────────────────────────────────

def calculate_breakdown(record: dict) -> dict:
    """
    Gross income = net_payout + airbnb_service_fee + taxes_withheld.
    This reconstructs the full booking value the guest paid before Airbnb
    deductions — the figure that must be reported as taxable rental income.
    """
    net_payout     = record.get("payout_amount")      or 0.0
    airbnb_fees    = record.get("airbnb_service_fee") or 0.0
    taxes_withheld = record.get("taxes_withheld")     or 0.0
    gross_income   = net_payout + airbnb_fees + taxes_withheld
    return {
        "date":           record.get("date"),
        "listing":        record.get("listing"),
        "gross_income":   round(gross_income,   2),
        "airbnb_fees":    round(airbnb_fees,    2),
        "taxes_withheld": round(taxes_withheld, 2),
        "net_payout":     round(net_payout,     2),
    }


# ── Endpoint 4: Airbnb breakdown ──────────────────────────────────────────────

@app.get("/airbnb-breakdown", summary="Financial breakdown of all stored Airbnb payouts", tags=["Airbnb"])
def airbnb_breakdown(db: Session = Depends(get_db)):
    rows = [airbnb_row_to_dict(r) for r in db.query(AirbnbTransaction).all()]
    if not rows:
        raise HTTPException(status_code=400, detail="No Airbnb data found. Please call /upload-airbnb first.")

    breakdowns  = [calculate_breakdown(r) for r in rows]
    total_gross = sum(b["gross_income"]   for b in breakdowns)
    total_fees  = sum(b["airbnb_fees"]    for b in breakdowns)
    total_taxes = sum(b["taxes_withheld"] for b in breakdowns)
    total_net   = sum(b["net_payout"]     for b in breakdowns)

    return {
        "status":  "success",
        "message": f"Breakdown calculated for {len(breakdowns)} Airbnb payout(s).",
        "totals": {
            "total_gross_income":              round(total_gross,  2),
            "total_fees":                      round(total_fees,   2),
            "total_taxes_withheld":            round(total_taxes,  2),
            "total_net_payout":                round(total_net,    2),
            "gross_equals_net_plus_deductions": round(total_gross, 2) == round(total_net + total_fees + total_taxes, 2),
        },
        "payouts": breakdowns,
    }


# ── Expense categorisation ────────────────────────────────────────────────────

CATEGORY_RULES: list[tuple[str, str]] = [
    ("AIRBNB",      "Income"),
    ("HOME DEPOT",  "Repairs"),
    ("LOWES",       "Repairs"),
    ("CLEANING",    "Cleaning"),
    ("UTILITY",     "Utilities"),
    ("ELECTRIC",    "Utilities"),
    ("WATER",       "Utilities"),
    ("INTERNET",    "Utilities"),
    ("UBER",        "Travel"),
    ("GAS",         "Travel"),
    ("SHELL",       "Travel"),
    ("EXXON",       "Travel"),
    ("GROCERY",     "Supplies"),
    ("GROCERIES",   "Supplies"),
    ("WALMART",     "Supplies"),
    ("RENT",        "Other"),
]

DEFAULT_CATEGORY    = "Other"
VALID_AI_CATEGORIES = {"Repairs", "Cleaning", "Utilities", "Supplies", "Travel", "Other"}


def categorize_with_ai(description: str) -> str:
    """Call GPT-4o-mini to categorise a transaction when no rule matches."""
    print(f"AI CALLED: {description}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Defaulting to 'Other'.")
        return DEFAULT_CATEGORY

    system_prompt = (
        "You are an accountant for an Airbnb host. "
        "Categorize bank transactions into ONLY one of these categories: "
        "Repairs, Cleaning, Utilities, Supplies, Travel, Other. "
        "Return ONLY the category name — no explanation, no punctuation."
    )
    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model":       "gpt-4o-mini",
                "max_tokens":  10,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": description},
                ],
            },
            timeout=10,
        )
        response.raise_for_status()
        ai_category = response.json()["choices"][0]["message"]["content"].strip()
        if ai_category in VALID_AI_CATEGORIES:
            return ai_category
        print(f"WARNING: AI returned unexpected category '{ai_category}'. Defaulting to 'Other'.")
        return DEFAULT_CATEGORY
    except Exception as exc:
        print(f"WARNING: AI categorisation failed ({exc}). Defaulting to 'Other'.")
        return DEFAULT_CATEGORY


def categorize_transaction(description: Optional[str]) -> str:
    """Stage 1: rule-based. Stage 2: AI fallback if nothing matches."""
    if not description:
        return DEFAULT_CATEGORY
    desc_upper = description.upper()
    for keyword, category in CATEGORY_RULES:
        if keyword in desc_upper:
            return category
    return categorize_with_ai(description)


# ── Endpoint 5: Categorise expenses ──────────────────────────────────────────

@app.get("/categorize-expenses", summary="Categorise stored bank transactions by expense type", tags=["Bank"])
def categorize_expenses(db: Session = Depends(get_db)):
    rows = [bank_row_to_dict(r) for r in db.query(BankTransaction).all()]
    if not rows:
        raise HTTPException(status_code=400, detail="No bank data found. Please call /upload-bank first.")

    categorized = []
    for record in rows:
        categorized.append({
            "date":        record.get("date"),
            "description": record.get("description"),
            "amount":      record.get("amount"),
            "category":    categorize_transaction(record.get("description")),
        })

    summary: dict[str, float] = {
        "income": 0.0, "repairs": 0.0, "cleaning": 0.0,
        "utilities": 0.0, "supplies": 0.0, "travel": 0.0, "other": 0.0,
    }
    for txn in categorized:
        cat_key = txn["category"].lower()
        amount  = abs(txn["amount"] or 0.0)
        if cat_key in summary:
            summary[cat_key] += amount
        else:
            summary["other"] += amount

    summary = {k: round(v, 2) for k, v in summary.items()}

    category_counts: dict[str, int] = {}
    for txn in categorized:
        key = txn["category"]
        category_counts[key] = category_counts.get(key, 0) + 1

    return {
        "status":          "success",
        "message":         f"Categorised {len(categorized)} bank transaction(s).",
        "summary":         summary,
        "category_counts": category_counts,
        "transactions":    categorized,
    }


# ── Schedule E Summary ────────────────────────────────────────────────────────

@app.get("/schedule-e-summary", summary="IRS Schedule E rental income & expense summary", tags=["Tax & Reporting"])
def schedule_e_summary(db: Session = Depends(get_db)):
    """
    Combines Airbnb gross income and categorised bank expenses into
    a Schedule E–style summary.  Both /upload-airbnb and /upload-bank
    should be called first; bank data is optional.
    """
    airbnb_rows = [airbnb_row_to_dict(r) for r in db.query(AirbnbTransaction).all()]
    if not airbnb_rows:
        raise HTTPException(status_code=400, detail="No Airbnb data found. Please call /upload-airbnb first.")

    # Section A — Gross income
    total_gross_income = sum(calculate_breakdown(r)["gross_income"] for r in airbnb_rows)

    # Section B — Expenses
    expenses: dict[str, float] = {
        "repairs": 0.0, "cleaning": 0.0, "utilities": 0.0,
        "supplies": 0.0, "travel": 0.0, "other": 0.0,
    }
    bank_rows = [bank_row_to_dict(r) for r in db.query(BankTransaction).all()]
    for record in bank_rows:
        category = categorize_transaction(record.get("description"))
        if category.lower() == "income":
            continue
        cat_key = category.lower()
        amount  = abs(record.get("amount") or 0.0)
        if cat_key in expenses:
            expenses[cat_key] += amount
        else:
            expenses["other"] += amount

    expenses       = {k: round(v, 2) for k, v in expenses.items()}
    total_expenses = sum(expenses.values())
    net_income     = round(total_gross_income - total_expenses, 2)

    return {
        "status": "success",
        "schedule_e_note": (
            "This summary maps to IRS Schedule E Part I. "
            "Report 'total_gross_income' on Line 3 (Rents received). "
            "Report individual expense categories on Lines 11–19. "
            "Report 'net_income' on Line 22 (Net rental income or loss). "
            "Consult a CPA before filing."
        ),
        "income":     {"total_gross_income": round(total_gross_income, 2)},
        "expenses":   {**expenses, "total_expenses": round(total_expenses, 2)},
        "net_income": net_income,
    }


# ── AI Insights ───────────────────────────────────────────────────────────────

@app.get("/ai-insights", summary="Generate AI-powered financial insights", tags=["AI & Insights"])
def ai_insights(db: Session = Depends(get_db)):
    """
    Generate 3 structured financial insights (PROBLEM / BREAKDOWN / ACTION)
    from the current Schedule E data using GPT-4o-mini.
    """
    airbnb_rows = [airbnb_row_to_dict(r) for r in db.query(AirbnbTransaction).all()]
    if not airbnb_rows:
        raise HTTPException(status_code=400, detail="No Airbnb data found. Please call /upload-airbnb first.")

    total_gross_income = sum(calculate_breakdown(r)["gross_income"] for r in airbnb_rows)

    expenses: dict[str, float] = {
        "repairs": 0.0, "cleaning": 0.0, "utilities": 0.0,
        "supplies": 0.0, "travel": 0.0, "other": 0.0,
    }
    bank_rows = [bank_row_to_dict(r) for r in db.query(BankTransaction).all()]
    for record in bank_rows:
        category = categorize_transaction(record.get("description"))
        if category.lower() == "income":
            continue
        cat_key = category.lower()
        amount  = abs(record.get("amount") or 0.0)
        if cat_key in expenses:
            expenses[cat_key] += amount
        else:
            expenses["other"] += amount

    expenses       = {k: round(v, 2) for k, v in expenses.items()}
    total_expenses = round(sum(expenses.values()), 2)
    net_income     = round(total_gross_income - total_expenses, 2)

    expense_ratio         = round((total_expenses / total_gross_income) * 100, 1) if total_gross_income > 0 else 0.0
    biggest_expense_key   = max(expenses, key=expenses.get)
    biggest_expense_value = expenses[biggest_expense_key]

    financial_summary = (
        f"Gross rental income: ${total_gross_income:,.2f}\n"
        f"Total expenses: ${total_expenses:,.2f}\n"
        f"  - repairs: ${expenses['repairs']:,.2f}\n"
        f"  - cleaning: ${expenses['cleaning']:,.2f}\n"
        f"  - utilities: ${expenses['utilities']:,.2f}\n"
        f"  - supplies: ${expenses['supplies']:,.2f}\n"
        f"  - travel: ${expenses['travel']:,.2f}\n"
        f"  - other: ${expenses['other']:,.2f}\n"
        f"Net income: ${net_income:,.2f}\n"
        f"Expense ratio: {expense_ratio}%\n"
        f"Largest expense category: {biggest_expense_key} (${biggest_expense_value:,.2f})"
    )

    FALLBACK_INSIGHTS = [
        "Unable to generate insights at this time. Please check your OpenAI API key.",
        "Tip: ensure OPENAI_API_KEY is set in your environment and try again.",
        "In the meantime, review your expense ratio manually: high ratios (>60%) may indicate overspending.",
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "fallback", "insights": FALLBACK_INSIGHTS}

    # ── AI prompt: neutral, analytical, business-focused ─────────────────────
    # "host" and person-references are explicitly banned so the output reads
    # as a financial report about a business, not advice to a specific person.
    system_prompt = (
        "You are a financial analyst producing a report on an Airbnb rental business. "
        "Given the financial dataset below, write exactly 3 insights that follow this structure:\n"
        "Insight 1 — PROBLEM: State the key financial issue for this Airbnb business "
        "(e.g. net loss, high expense ratio). Must include a dollar amount or percentage.\n"
        "Insight 2 — BREAKDOWN: Identify the single largest or most significant expense category. "
        "State its dollar amount and its share of total expenses.\n"
        "Insight 3 — ACTION: State one specific, numbers-based improvement opportunity "
        "(e.g. 'Reducing cleaning costs by 20% would save $X and bring net income to $Y').\n"
        "Strict rules for ALL 3 insights: "
        "(a) Every insight MUST include at least one specific number (dollar amount, %, or ratio). "
        "(b) Each insight covers a different idea — no repeating the same point. "
        "(c) 1-2 sentences max per insight. "
        "(d) Do NOT use the word 'host' or refer to a person — speak about 'this Airbnb business' or 'this dataset'. "
        "(e) Do NOT use vague openers like 'consider', 'you may want to', or 'it might be worth'. "
        "(f) Write in a neutral, analytical tone — like a financial report, not personal advice. "
        "(g) Return ONLY a JSON array of exactly 3 strings — no keys, no markdown, no preamble."
    )

    user_prompt = (
        "Here is the financial dataset for this Airbnb business:\n\n"
        + financial_summary
        + "\n\nWrite exactly 3 insights following the PROBLEM / BREAKDOWN / ACTION structure. "
        + "Each insight must cite at least one specific number from the dataset above. "
        + "Do not use the word 'host' — refer to 'this Airbnb business' instead."
    )

    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model":       "gpt-4o-mini",
                "max_tokens":  300,
                "temperature": 0.4,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            },
            timeout=15,
        )
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"].strip()
        insights = json.loads(raw_text)
        if isinstance(insights, list) and len(insights) == 3 and all(isinstance(i, str) and i.strip() for i in insights):
            return {"status": "success", "insights": insights}
        print(f"WARNING: AI returned unexpected insight shape: {insights}")
        return {"status": "fallback", "insights": FALLBACK_INSIGHTS}
    except Exception as exc:
        print(f"WARNING: AI insights failed ({exc}). Returning fallback.")
        return {"status": "fallback", "insights": FALLBACK_INSIGHTS}


# ── Schedule E PDF Export ────────────────────────────────────────────────────
#
# Uses reportlab to build a clean, readable PDF in memory (no temp file needed).
# The PDF is streamed directly to the caller with Content-Disposition headers
# so browsers offer it as a download named "schedule_e.pdf".
#
# reportlab docs: https://docs.reportlab.com/reportlab/userguide/

@app.get(
    "/export-schedule-e",
    summary="Download a Schedule E summary as a PDF",
    tags=["Tax & Reporting"],
    response_class=StreamingResponse,
)
def export_schedule_e(db: Session = Depends(get_db)):
    """
    Generate and stream a Schedule E–style PDF using data already in the DB.

    The PDF contains:
    - Report title and generation date
    - Section A: Rental income (gross income per payout, total)
    - Section B: Expense breakdown by category (Cleaning, Repairs, etc.)
    - Section C: Net income / loss
    - Footer disclaimer

    Call /upload-airbnb (and optionally /upload-bank) before this endpoint.
    """
    # Late import — only needed for this endpoint so the server still starts
    # even if reportlab isn't installed yet (it will fail at request time with
    # a clear ImportError rather than silently at boot).
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="reportlab is not installed. Run: pip install reportlab",
        )

    # ── Pull data from DB ──────────────────────────────────────────────────────
    airbnb_rows = [airbnb_row_to_dict(r) for r in db.query(AirbnbTransaction).all()]
    bank_rows   = [bank_row_to_dict(r)   for r in db.query(BankTransaction).all()]

    if not airbnb_rows:
        raise HTTPException(
            status_code=400,
            detail="No Airbnb data found. Please call /upload-airbnb first.",
        )

    # ── Section A: Rental income ───────────────────────────────────────────────
    # We report gross income (payout + fees + taxes) not just the net payout,
    # because that is the IRS-required figure for Schedule E Line 3.
    income_rows = []   # list of (date, listing, gross_income) tuples for the table
    total_gross_income = 0.0

    for r in airbnb_rows:
        bd = calculate_breakdown(r)
        income_rows.append((
            r.get("date") or "—",
            r.get("listing") or "—",
            bd["gross_income"],
        ))
        total_gross_income += bd["gross_income"]

    total_gross_income = round(total_gross_income, 2)

    # ── Section B: Expense breakdown ───────────────────────────────────────────
    # Bucket names mirror the Schedule E line items we track.
    # Using the existing categorize_transaction() keeps logic consistent with
    # /schedule-e-summary and /categorize-expenses.
    expense_buckets: dict[str, float] = {
        "Cleaning":    0.0,
        "Repairs":     0.0,
        "Utilities":   0.0,
        "Supplies":    0.0,
        "Travel":      0.0,
        "Other":       0.0,
    }

    for record in bank_rows:
        category = categorize_transaction(record.get("description"))
        if category == "Income":
            continue                       # Airbnb deposits are income, not expenses
        amount = abs(record.get("amount") or 0.0)
        if category in expense_buckets:
            expense_buckets[category] += amount
        else:
            expense_buckets["Other"] += amount

    expense_buckets  = {k: round(v, 2) for k, v in expense_buckets.items()}
    total_expenses   = round(sum(expense_buckets.values()), 2)
    net_income       = round(total_gross_income - total_expenses, 2)
    generated_date   = datetime.now().strftime("%B %d, %Y")

    # ── Build PDF in memory ────────────────────────────────────────────────────
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
        title="Schedule E Summary",
    )

    styles = getSampleStyleSheet()

    # ── Custom styles ──────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=4,
        textColor=colors.HexColor("#1a1a2e"),
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=16,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=18,
        spaceAfter=6,
        textColor=colors.HexColor("#1f2937"),
        borderPad=0,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#374151"),
        spaceAfter=4,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.HexColor("#9ca3af"),
        spaceBefore=20,
    )

    # ── Helper: format a dollar amount ────────────────────────────────────────
    def fmt(amount: float) -> str:
        """Format a float as a dollar string, with parentheses for negatives."""
        if amount < 0:
            return f"(${abs(amount):,.2f})"
        return f"${amount:,.2f}"

    # ── Table style shared by income and expense tables ───────────────────────
    def base_table_style(header_bg=colors.HexColor("#f3f4f6")) -> TableStyle:
        return TableStyle([
            # Header row
            ("BACKGROUND",   (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING",(0, 0), (-1, 0), 6),
            ("TOPPADDING",   (0, 0), (-1, 0), 6),
            # Data rows
            ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",     (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ("BOTTOMPADDING",(0, 1), (-1, -1), 5),
            ("TOPPADDING",   (0, 1), (-1, -1), 5),
            # Grid
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
            ("LINEBELOW",    (0, 0), (-1, 0), 1, colors.HexColor("#d1d5db")),
            # Right-align the amount column (last column)
            ("ALIGN",        (-1, 0), (-1, -1), "RIGHT"),
        ])

    # ── Assemble flowable elements ────────────────────────────────────────────
    story = []

    # Title block
    story.append(Paragraph("Schedule E Summary", title_style))
    story.append(Paragraph(f"Airbnb Rental Income &amp; Expenses · Generated {generated_date}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")))

    # ── SECTION A — Rental Income ─────────────────────────────────────────────
    story.append(Paragraph("Section A — Rental Income", section_heading))
    story.append(Paragraph(
        "Gross income is the full booking value paid by guests, before Airbnb fees "
        "and tax withholdings are deducted. Report this figure on Schedule E Line 3.",
        body_style,
    ))

    # Income table: Date | Listing | Gross Income
    income_table_data = [["Date", "Listing / Property", "Gross Income"]]
    for date_val, listing_val, gross in income_rows:
        income_table_data.append([
            date_val,
            listing_val[:45] + "…" if len(str(listing_val)) > 45 else listing_val,
            fmt(gross),
        ])
    # Totals row
    income_table_data.append(["", "TOTAL RENTAL INCOME", fmt(total_gross_income)])

    income_table = Table(
        income_table_data,
        colWidths=[1.1 * inch, 3.9 * inch, 1.5 * inch],
        repeatRows=1,
    )
    ts = base_table_style()
    # Bold + top border on the totals row
    ts.add("FONTNAME",  (0, -1), (-1, -1), "Helvetica-Bold")
    ts.add("LINEABOVE", (0, -1), (-1, -1), 1, colors.HexColor("#6b7280"))
    ts.add("BACKGROUND",(0, -1), (-1, -1), colors.HexColor("#ecfdf5"))
    income_table.setStyle(ts)
    story.append(income_table)

    # ── SECTION B — Expenses ──────────────────────────────────────────────────
    story.append(Paragraph("Section B — Expense Breakdown", section_heading))
    story.append(Paragraph(
        "Expenses are categorised from bank transaction descriptions using keyword "
        "matching. Report each category on the corresponding Schedule E line (11–19).",
        body_style,
    ))

    # Map our internal category names to the nearest Schedule E line number
    sch_e_lines = {
        "Cleaning":  "Line 14 – Cleaning / Management",
        "Repairs":   "Line 11 – Repairs & Maintenance",
        "Utilities": "Line 18 – Utilities",
        "Supplies":  "Line 19 – Supplies",
        "Travel":    "Line 19 – Travel",
        "Other":     "Line 19 – Other",
    }

    expense_table_data = [["Category", "Schedule E Line", "Amount"]]
    for category, amount in expense_buckets.items():
        expense_table_data.append([
            category,
            sch_e_lines.get(category, "Line 19 – Other"),
            fmt(amount),
        ])
    expense_table_data.append(["", "TOTAL EXPENSES", fmt(total_expenses)])

    expense_table = Table(
        expense_table_data,
        colWidths=[1.3 * inch, 3.7 * inch, 1.5 * inch],
        repeatRows=1,
    )
    ts2 = base_table_style(header_bg=colors.HexColor("#fef9c3"))
    ts2.add("FONTNAME",  (0, -1), (-1, -1), "Helvetica-Bold")
    ts2.add("LINEABOVE", (0, -1), (-1, -1), 1, colors.HexColor("#6b7280"))
    ts2.add("BACKGROUND",(0, -1), (-1, -1), colors.HexColor("#fef3c7"))
    expense_table.setStyle(ts2)
    story.append(expense_table)

    # ── SECTION C — Net Income ────────────────────────────────────────────────
    story.append(Paragraph("Section C — Net Income / Loss", section_heading))

    net_color = colors.HexColor("#ecfdf5") if net_income >= 0 else colors.HexColor("#fef2f2")
    net_label = "NET RENTAL INCOME" if net_income >= 0 else "NET RENTAL LOSS"

    net_table = Table(
        [
            ["Gross Rental Income",  fmt(total_gross_income)],
            ["Total Expenses",      f"({fmt(total_expenses)})"],
            [net_label,              fmt(net_income)],
        ],
        colWidths=[5 * inch, 1.5 * inch],
    )
    net_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (-1, 1), "Helvetica"),
        ("FONTNAME",    (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ALIGN",       (1, 0), (1, -1), "RIGHT"),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LINEABOVE",   (0, 2), (-1, 2), 1, colors.HexColor("#6b7280")),
        ("BACKGROUND",  (0, 2), (-1, 2), net_color),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
    ]))
    story.append(net_table)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "⚠ This document is generated for informational purposes only and does not "
        "constitute tax advice. Figures are based on data you uploaded and may not "
        "reflect all income or deductions. Consult a licensed CPA or tax professional "
        "before filing IRS Schedule E.",
        disclaimer_style,
    ))

    # ── Build and stream the PDF ──────────────────────────────────────────────
    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="schedule_e.pdf"',
        },
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root(db: Session = Depends(get_db)):
    """Quick health-check — confirms the API and database are running."""
    return {
        "status":              "ok",
        "message":             "Airbnb Accounting API is running. Visit /docs for the full API reference.",
        "stored_airbnb_rows":  db.query(AirbnbTransaction).count(),
        "stored_bank_rows":    db.query(BankTransaction).count(),
        "stored_matches":      db.query(Match).count(),
        "stored_manual_matches": db.query(Match).filter(Match.is_manual == True).count(),
    }