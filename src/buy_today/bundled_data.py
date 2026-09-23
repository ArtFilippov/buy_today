"""Download the original Olist tables while building the container image.

This module deliberately uses only the standard library so Docker can execute
the file directly, independently of the application's scientific dependencies.
"""

import argparse
from pathlib import Path
import shutil
import stat
import tempfile
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile, ZipInfo


OLIST_FILES = (
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

_OLIST_URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
_CHUNK_SIZE = 1024 * 1024
_UNIX_CREATE_SYSTEM = 3
_DOS_DIRECTORY = 0x10
_ENCRYPTED_MEMBER = 1


def _validate_members(members: list[ZipInfo]) -> None:
    names = [member.filename for member in members]
    if len(names) != len(OLIST_FILES) or set(names) != set(OLIST_FILES):
        raise ValueError(
            f"Olist ZIP must contain exactly the nine original CSV files; found: {names!r}"
        )
    for member in members:
        _validate_member(member)


def _validate_member(member: ZipInfo) -> None:
    # Some ZIP producers omit Unix file-type bits. Those entries are copied
    # into ordinary files; explicit links, directories and devices are not.
    file_type = (
        stat.S_IFMT(member.external_attr >> 16)
        if member.create_system == _UNIX_CREATE_SYSTEM
        else 0
    )
    invalid_type = (
        member.is_dir()
        or member.external_attr & _DOS_DIRECTORY
        or file_type not in {0, stat.S_IFREG}
    )
    if (
        member.orig_filename != member.filename
        or invalid_type
        or member.file_size <= 0
        or member.flag_bits & _ENCRYPTED_MEMBER
    ):
        raise ValueError(f"Olist ZIP member must be a nonempty regular CSV: {member.filename}")


def _occupied(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _download_archive(path: Path) -> None:
    request = Request(
        _OLIST_URL,
        headers={"Accept": "application/zip", "User-Agent": "buy_today-container-build"},
    )
    # urlopen uses GET and follows Kaggle's redirect to signed ZIP storage.
    with urlopen(request, timeout=120) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output, length=_CHUNK_SIZE)


def _extract_member(archive: ZipFile, member: ZipInfo, extracted: Path) -> None:
    path = extracted / member.filename
    with archive.open(member) as source, path.open("xb") as output:
        # Reading through EOF verifies the member's CRC.
        shutil.copyfileobj(source, output, length=_CHUNK_SIZE)
    if path.stat().st_size != member.file_size:
        raise ValueError(f"Incomplete Olist CSV: {member.filename}")


def _extract_archive(path: Path, extracted: Path) -> None:
    with ZipFile(path) as archive:
        members = archive.infolist()
        _validate_members(members)
        for member in members:
            _extract_member(archive, member, extracted)


def download_olist(destination: Path | str) -> Path:
    """Download the latest official ZIP and publish its nine raw CSVs.

    The destination must not already exist. Return its absolute path only after
    all members have been extracted and their ZIP CRCs verified. Network, ZIP
    and filesystem failures propagate; temporary downloads and partial extracts
    are removed on failure as well as on success. No credentials are required.

    Args:
        destination (Path | str): New directory for the nine original CSV files.

    Returns:
        Path: Absolute destination after the complete bundle has been published.

    Raises:
        FileExistsError: The destination already exists or is a symlink.
        ValueError: The downloaded archive is invalid or violates the bundle contract.
    """
    destination = Path(destination).absolute()
    if _occupied(destination):
        raise FileExistsError(f"Olist destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".olist-", dir=destination.parent) as temporary_name:
        temporary = Path(temporary_name)
        archive_path = temporary / "olist.zip"
        _download_archive(archive_path)

        extracted = temporary / "data"
        extracted.mkdir()
        try:
            _extract_archive(archive_path, extracted)
        except BadZipFile as exc:
            raise ValueError(f"Olist download is not a valid ZIP archive: {exc}") from exc

        _ = extracted.rename(destination)

    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download raw Olist CSVs for the image build.")
    _ = parser.add_argument(
        "destination", type=Path, help="new directory for the nine original CSVs"
    )
    args = parser.parse_args(argv)
    _ = download_olist(args.destination)


if __name__ == "__main__":
    main()
