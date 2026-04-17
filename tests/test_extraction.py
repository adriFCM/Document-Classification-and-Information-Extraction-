from src.date_utils import find_dates, DATE_RE


def test_find_dates_basic():
    assert find_dates("Invoice date: 15/03/2024") == ["15/03/2024"]


def test_find_dates_named_month():
    assert find_dates("Issued March 15, 2024") == ["March 15, 2024"]


def test_find_dates_multiple():
    got = find_dates("From 01/01/2024 to 31/12/2024")
    assert got == ["01/01/2024", "31/12/2024"]


def test_date_re_exported():
    assert DATE_RE.search("15/03/2024") is not None


def test_us_date_format():
    # MM/DD/YYYY — day > 12 makes this unambiguous US-style
    assert find_dates("Date of issue: 07/29/2011") == ["07/29/2011"]


def test_day_first_still_preferred_on_ambiguous():
    # 03/04/2024 is ambiguous — day-first should win (SROIE behavior)
    assert find_dates("Date: 03/04/2024") == ["03/04/2024"]


from src.information_extraction import extract_invoice_fields


def _extract(text):
    return extract_invoice_fields(text)


def test_client_label_as_recipient():
    text = (
        "Some Header Line\n"
        "Client:\n"
        "Thompson PLC\n"
        "810 Adkins Canyon\n"
    )
    assert _extract(text)["recipient"] == "Thompson PLC"


def test_seller_label_as_issuer():
    text = (
        "Date of issue: 07/29/2011\n"
        "\n"
        "Seller:\n"
        "Wood-Kim\n"
        "8881 Nicholas Grove\n"
        "\n"
        "Client:\n"
        "Thompson PLC\n"
    )
    assert _extract(text)["issuer"] == "Wood-Kim"


def test_date_on_separate_line_from_label():
    text = "Date of issue:\n07/29/2011\nSeller:\nFoo Co\n"
    assert _extract(text)["invoice_date"] == "07/29/2011"


def test_due_date_on_separate_line():
    text = "Due date:\n15/04/2024\n"
    assert _extract(text)["due_date"] == "15/04/2024"


def test_european_decimal_total():
    text = "TOTAL: $ 164,97\n"
    assert _extract(text)["total"] == "164,97"


def test_net_vat_gross_row_picks_rightmost():
    text = (
        "           VAT [%]   Net worth   VAT    Gross worth\n"
        "              10%     149,97    15,00    164,97\n"
        "Total                 $ 149,97  $ 15,00  $ 164,97\n"
    )
    assert _extract(text)["total"] == "164,97"


def test_extract_invoice_fields_accepts_pdf_bytes_kwarg():
    # Passing pdf_bytes=None must not raise; result should match the text-only call.
    text = "Invoice date: 15/03/2024\nTOTAL: $100.00\n"
    assert extract_invoice_fields(text, pdf_bytes=None) == extract_invoice_fields(text)


from src.layout_extractor import extract as layout_extract


def test_layout_extract_empty_bytes_returns_all_none():
    # Zero-byte PDF — pdfplumber will fail gracefully; we must return all-None.
    result = layout_extract(b"")
    assert result == {
        "invoice_number": None,
        "invoice_date":   None,
        "due_date":       None,
        "issuer":         None,
        "recipient":      None,
        "total":          None,
    }


from src.layout_extractor import find_value_for_label


def test_find_value_right_of_label():
    # Label "Date of issue:" at x=100..200 on line y=50.
    # Value "07/29/2011" at x=400 on same line.
    words = [
        {"text": "Date",   "x0": 100, "x1": 130, "top": 50, "bottom": 60},
        {"text": "of",     "x0": 135, "x1": 150, "top": 50, "bottom": 60},
        {"text": "issue:", "x0": 155, "x1": 200, "top": 50, "bottom": 60},
        {"text": "07/29/2011", "x0": 400, "x1": 470, "top": 50, "bottom": 60},
    ]
    got = find_value_for_label([r"date of issue"], words)
    assert got == "07/29/2011"


def test_find_value_below_label():
    # Label "Seller:" at x=100..150 on line y=50.
    # Value "Wood-Kim" at x=100..160 on line y=80 (same column).
    words = [
        {"text": "Seller:",  "x0": 100, "x1": 150, "top": 50, "bottom": 60},
        {"text": "Wood-Kim", "x0": 100, "x1": 160, "top": 80, "bottom": 90},
    ]
    got = find_value_for_label([r"seller"], words)
    assert got == "Wood-Kim"


def test_find_value_returns_none_when_label_absent():
    words = [{"text": "Hello", "x0": 0, "x1": 50, "top": 0, "bottom": 10}]
    assert find_value_for_label([r"date"], words) is None


def test_total_ignores_bare_year():
    # "Total" line has a year hanging around (OCR noise) plus a real amount.
    # Must pick the real amount, not the year.
    text = "Total  5,00  2013\n"
    assert _extract(text)["total"] == "5,00"


def test_total_none_when_only_a_year():
    # If the only thing on the total line looks like a year, we should return None.
    text = "Total 2013\n"
    assert _extract(text)["total"] is None


def test_total_ignores_date_on_next_line():
    # OCR quirk: "Total" alone, then a date (no real amount near the label).
    # Must NOT return a day or month from the date.
    text = "Total\n04/13/2013\nUM\n"
    assert _extract(text)["total"] is None


def test_compact_date_rejects_invalid_year():
    # 8-digit invoice number 22083742 must NOT parse as DDMMYYYY (year 3742).
    assert find_dates("Invoice number: 22083742") == []


def test_compact_date_accepts_valid_year():
    # 25122024 → 25/12/2024 (valid 20xx year).
    assert find_dates("Paid 25122024") == ["25122024"]


def test_extract_from_words_end_to_end():
    # Two-column invoice simulated with word bboxes (mimics tesseract output).
    from src.layout_extractor import extract_from_words
    words = [
        {"text": "Invoice", "x0": 100, "x1": 140, "top": 30, "bottom": 40},
        {"text": "no:",     "x0": 145, "x1": 165, "top": 30, "bottom": 40},
        {"text": "INV-42",  "x0": 200, "x1": 250, "top": 30, "bottom": 40},
        {"text": "Date",    "x0": 100, "x1": 130, "top": 50, "bottom": 60},
        {"text": "of",      "x0": 135, "x1": 150, "top": 50, "bottom": 60},
        {"text": "issue:",  "x0": 155, "x1": 200, "top": 50, "bottom": 60},
        {"text": "07/29/2011", "x0": 400, "x1": 470, "top": 50, "bottom": 60},
        {"text": "Seller:", "x0": 100, "x1": 150, "top": 100, "bottom": 110},
        {"text": "Wood-Kim","x0": 100, "x1": 160, "top": 120, "bottom": 130},
        {"text": "Client:", "x0": 400, "x1": 450, "top": 100, "bottom": 110},
        {"text": "Thompson","x0": 400, "x1": 470, "top": 120, "bottom": 130},
        {"text": "PLC",     "x0": 475, "x1": 500, "top": 120, "bottom": 130},
        {"text": "Total",   "x0": 100, "x1": 140, "top": 300, "bottom": 310},
        {"text": "$164.97", "x0": 500, "x1": 560, "top": 300, "bottom": 310},
    ]
    got = extract_from_words(words)
    assert got["invoice_number"] == "INV-42"
    assert got["invoice_date"] == "07/29/2011"
    assert got["issuer"] == "Wood-Kim"
    assert got["recipient"] == "Thompson PLC"
    assert got["total"] == "164.97"
