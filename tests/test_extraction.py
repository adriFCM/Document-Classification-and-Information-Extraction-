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
