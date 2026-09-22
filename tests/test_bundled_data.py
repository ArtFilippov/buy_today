from email.message import Message
from io import BytesIO
import stat
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, ProxyHandler, build_opener
from urllib.response import addinfourl
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from buy_today import bundled_data


RAW_FILES = (
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
)


def make_zip(entries=None, *, compression=ZIP_DEFLATED):
    if entries is None:
        entries = [(name, b"id,value\n1,example\n") for name in RAW_FILES]
    output = BytesIO()
    with ZipFile(output, "w", compression=compression) as archive:
        for name, contents in entries:
            archive.writestr(name, contents)
    return output.getvalue()


@pytest.fixture
def serve_zip(monkeypatch):
    """Mock the HTTP boundary, while exercising real ZIP decoding and writes."""
    def serve(payload):
        response = BytesIO(payload)
        requests = []

        def open_url(request, *, timeout):
            requests.append((request, timeout))
            return response

        monkeypatch.setattr(bundled_data, "urlopen", open_url)
        return response, requests

    return serve


def test_downloads_exact_raw_tables_and_removes_archive(tmp_path, serve_zip):
    entries = [(name, f'id,value\n1,"São Paulo {name}"\n'.encode()) for name in RAW_FILES]
    response, requests = serve_zip(make_zip(entries))
    destination = tmp_path / "raw Olist"

    result = bundled_data.download_olist(str(destination))

    assert bundled_data.OLIST_FILES == RAW_FILES
    assert result == destination
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == dict(entries)
    assert all(path.is_file() and not path.is_symlink() for path in destination.iterdir())
    assert list(tmp_path.iterdir()) == [destination]
    assert response.closed
    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url == (
        "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
    )
    assert request.get_method() == "GET"
    assert not request.has_header("Authorization")
    assert timeout > 0


def test_streams_download_in_bounded_chunks(tmp_path, monkeypatch):
    entries = [(name, b"id,value\n" + b"1,example\n" * 250_000) for name in RAW_FILES[:1]]
    entries.extend((name, b"id\n1\n") for name in RAW_FILES[1:])

    class ChunkedResponse(BytesIO):
        read_sizes = []

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            self.read_sizes.append(size)
            # A response may return less than the requested chunk size.
            return super().read(min(size, 65_536))

    response = ChunkedResponse(make_zip(entries, compression=ZIP_STORED))
    monkeypatch.setattr(bundled_data, "urlopen", lambda *args, **kwargs: response)

    destination = bundled_data.download_olist(tmp_path / "olist")

    assert (destination / RAW_FILES[0]).read_bytes() == entries[0][1]
    assert len(response.read_sizes) > 3
    assert response.closed
    assert list(tmp_path.iterdir()) == [destination]


def test_follows_kaggle_redirect_to_zip_without_credentials(tmp_path, monkeypatch):
    endpoint = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
    storage_url = "https://storage.googleapis.com/olist.zip?signature=example"
    requests = []
    responses = []

    class MockHTTPSHandler(HTTPSHandler):
        def https_open(self, request):
            requests.append(request)
            headers = Message()
            if request.full_url == endpoint:
                headers["Location"] = storage_url
                response = addinfourl(BytesIO(b""), headers, endpoint, 302)
                response.msg = "Found"
            else:
                assert request.full_url == storage_url
                headers["Content-Type"] = "application/zip"
                response = addinfourl(BytesIO(make_zip()), headers, storage_url, 200)
                response.msg = "OK"
            responses.append(response)
            return response

    # Keep urllib's real redirect handling, replacing only the HTTPS transport.
    opener = build_opener(ProxyHandler({}), MockHTTPSHandler())
    monkeypatch.setattr(bundled_data, "urlopen", opener.open)

    destination = bundled_data.download_olist(tmp_path / "olist")

    assert {path.name for path in destination.iterdir()} == set(RAW_FILES)
    assert [request.full_url for request in requests] == [endpoint, storage_url]
    assert all(request.get_method() == "GET" for request in requests)
    assert all(not request.has_header("Authorization") for request in requests)
    assert all(response.closed for response in responses)
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("problem", [
    "missing", "unexpected", "prepared", "duplicate", "empty", "nested", "traversal",
    "directory", "symlink", "device", "dos-directory",
])
def test_rejects_invalid_zip_members_without_publishing(tmp_path, serve_zip, problem):
    entries = [(name, b"id\n1\n") for name in RAW_FILES]
    if problem == "missing":
        entries.pop()
    elif problem == "unexpected":
        entries.append(("extra.csv", b"id\n1\n"))
    elif problem == "prepared":
        entries[-1] = ("olist_prepared_dataset.csv", b"id\n1\n")
    elif problem == "duplicate":
        entries[-1] = entries[0]
    elif problem == "empty":
        entries[-1] = (RAW_FILES[-1], b"")
    elif problem in {"nested", "traversal"}:
        prefix = "nested/" if problem == "nested" else "../"
        entries[-1] = (prefix + RAW_FILES[-1], b"id\n1\n")
    else:
        member = ZipInfo(RAW_FILES[-1])
        member.create_system = 3
        if problem == "dos-directory":
            member.create_system = 0
            member.external_attr = 0x10
        else:
            file_type = {
                "directory": stat.S_IFDIR,
                "symlink": stat.S_IFLNK,
                "device": stat.S_IFCHR,
            }[problem]
            member.external_attr = (file_type | 0o644) << 16
        entries[-1] = (member, b"id\n1\n")

    if problem == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            payload = make_zip(entries)
    else:
        payload = make_zip(entries)
    response, _ = serve_zip(payload)

    with pytest.raises(ValueError, match="Olist ZIP"):
        bundled_data.download_olist(tmp_path / "olist")

    assert list(tmp_path.iterdir()) == []
    assert response.closed


@pytest.mark.parametrize("payload", [b"", b"<html>Login required</html>", b"PK\x03\x04truncated"])
def test_rejects_non_zip_responses_and_cleans_download(tmp_path, serve_zip, payload):
    response, _ = serve_zip(payload)

    with pytest.raises(ValueError, match="not a valid ZIP"):
        bundled_data.download_olist(tmp_path / "olist")

    assert list(tmp_path.iterdir()) == []
    assert response.closed


def test_corrupt_member_crc_leaves_no_partial_bundle(tmp_path, serve_zip):
    entries = [(name, b"id\n1\n") for name in RAW_FILES]
    entries[-1] = (RAW_FILES[-1], b"id\nlast-member\n")
    payload = make_zip(entries, compression=ZIP_STORED)
    # Preserve ZIP metadata but corrupt the last member, after earlier members
    # have already been extracted. Merely inspecting the directory is not enough.
    payload = payload.replace(b"last-member", b"bad--member", 1)
    response, _ = serve_zip(payload)

    with pytest.raises(ValueError, match="CRC"):
        bundled_data.download_olist(tmp_path / "olist")

    assert list(tmp_path.iterdir()) == []
    assert response.closed


@pytest.mark.parametrize("failure", [
    URLError("connection unavailable"),
    HTTPError("https://www.kaggle.com/", 503, "unavailable", {}, None),
    TimeoutError("download timed out"),
])
def test_connection_failures_clean_temporary_directory(tmp_path, monkeypatch, failure):
    def open_url(*args, **kwargs):
        raise failure

    monkeypatch.setattr(bundled_data, "urlopen", open_url)

    with pytest.raises(type(failure)) as caught:
        bundled_data.download_olist(tmp_path / "olist")

    assert caught.value is failure
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_closes_response_and_removes_partial_zip(tmp_path, monkeypatch):
    class InterruptedResponse(BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise ConnectionResetError("connection lost during download")
            return super().read(32)

    response = InterruptedResponse(make_zip())
    monkeypatch.setattr(bundled_data, "urlopen", lambda *args, **kwargs: response)

    with pytest.raises(ConnectionResetError, match="connection lost"):
        bundled_data.download_olist(tmp_path / "olist")

    assert response.closed
    assert list(tmp_path.iterdir()) == []


def test_existing_bundle_is_not_overwritten(tmp_path, monkeypatch):
    destination = tmp_path / "olist"
    destination.mkdir()
    existing = destination / RAW_FILES[0]
    existing.write_bytes(b"existing data")

    def unexpected_download(*args, **kwargs):
        pytest.fail("an existing destination must be rejected before downloading")

    monkeypatch.setattr(bundled_data, "urlopen", unexpected_download)

    with pytest.raises(FileExistsError, match="already exists"):
        bundled_data.download_olist(destination)

    assert existing.read_bytes() == b"existing data"
    assert list(destination.iterdir()) == [existing]
    assert list(tmp_path.iterdir()) == [destination]


def test_build_entrypoint_accepts_destination(tmp_path, serve_zip):
    response, _ = serve_zip(make_zip())
    destination = tmp_path / "image" / "opt" / "buy_today" / "olist"

    bundled_data.main([str(destination)])

    assert {path.name for path in destination.iterdir()} == set(RAW_FILES)
    assert list(destination.parent.iterdir()) == [destination]
    assert response.closed


def test_direct_build_script_needs_only_standard_library(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-S", bundled_data.__file__, "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "destination" in result.stdout
