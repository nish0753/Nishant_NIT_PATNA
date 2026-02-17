"""Unit tests for core utilities and logic."""
import pytest
import json

# ---- Tests for clean_json ----
from app.extractor import clean_json


class TestCleanJson:
    def test_strips_markdown_code_block(self):
        raw = '```json\n{"key": "value"}\n```'
        assert json.loads(clean_json(raw)) == {"key": "value"}

    def test_extracts_json_without_markdown(self):
        raw = 'Some preamble {"key": 1} trailing text'
        assert json.loads(clean_json(raw)) == {"key": 1}

    def test_fixes_missing_comma_between_objects(self):
        raw = '[{"a":1} {"b":2}]'
        result = clean_json(raw)
        assert '}, {' in result

    def test_removes_trailing_comma_in_object(self):
        raw = '{"a": 1, "b": 2,}'
        parsed = json.loads(clean_json(raw))
        assert parsed == {"a": 1, "b": 2}

    def test_removes_trailing_comma_in_array(self):
        raw = '[1, 2, 3,]'
        parsed = json.loads(clean_json(raw))
        assert parsed == [1, 2, 3]


# ---- Tests for safe_float ----
from app.main import safe_float


class TestSafeFloat:
    def test_normal_float(self):
        assert safe_float(3.14) == 3.14

    def test_string_with_commas(self):
        assert safe_float("1,234.56") == 1234.56

    def test_none_returns_zero(self):
        assert safe_float(None) == 0.0

    def test_invalid_string_returns_zero(self):
        assert safe_float("abc") == 0.0

    def test_integer(self):
        assert safe_float(42) == 42.0

    def test_empty_string(self):
        assert safe_float("") == 0.0


# ---- Tests for build_response ----
from app.main import build_response


class TestBuildResponse:
    def test_valid_input(self):
        result = {
            "pagewise_line_items": [
                {
                    "page_no": "1",
                    "page_type": "Bill Detail",
                    "bill_items": [
                        {"item_name": "Aspirin", "item_amount": 10.0, "item_rate": 5.0, "item_quantity": 2.0}
                    ],
                }
            ],
            "total_item_count": 1,
        }
        resp = build_response(result)
        assert resp["is_success"] is True
        assert resp["data"]["total_item_count"] == 1
        assert resp["data"]["pagewise_line_items"][0]["bill_items"][0]["item_name"] == "Aspirin"

    def test_empty_input(self):
        resp = build_response({})
        assert resp["is_success"] is True
        assert resp["data"]["total_item_count"] == 0

    def test_non_dict_input(self):
        resp = build_response("garbage")
        assert resp["is_success"] is True
        assert resp["data"]["total_item_count"] == 0

    def test_malformed_items_skipped(self):
        result = {
            "pagewise_line_items": [
                {
                    "page_no": "1",
                    "page_type": "Bill Detail",
                    "bill_items": ["not_a_dict", {"item_name": "Valid", "item_amount": 5}],
                }
            ]
        }
        resp = build_response(result)
        assert resp["data"]["total_item_count"] == 1


# ---- Tests for preprocess_image ----
from app.extractor import preprocess_image
from PIL import Image
import io


class TestPreprocessImage:
    def _make_test_image(self) -> bytes:
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_returns_bytes(self):
        result = preprocess_image(self._make_test_image())
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_invalid_bytes_returns_original(self):
        garbage = b"not_an_image"
        result = preprocess_image(garbage)
        assert result == garbage


# ---- Tests for FastAPI endpoints ----
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestEndpoints:
    def test_root_returns_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_extract_bill_invalid_url_scheme(self):
        resp = client.post("/extract-bill-data", json={"document": "ftp://evil.com/file.pdf"})
        assert resp.status_code == 422  # Validation error

    def test_extract_bill_missing_document(self):
        resp = client.post("/extract-bill-data", json={})
        assert resp.status_code == 422

    def test_extract_bill_empty_url(self):
        resp = client.post("/extract-bill-data", json={"document": ""})
        assert resp.status_code == 422
