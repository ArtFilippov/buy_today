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


def _validate_members(members: list[ZipInfo]) -> None:
    names = [member.filename for member in members]
    if len(names) != len(OLIST_FILES) or set(names) != set(OLIST_FILES):
        raise ValueError(
            "Olist ZIP must contain exactly the nine original CSV files; "
            f"found: {names!r}"
        )
    for member in members:
        # Some ZIP producers omit Unix file-type bits. Those entries are copied
        # into ordinary files; explicit links, directories and devices are not.
        file_type = (
            stat.S_IFMT(member.external_attr >> 16) if member.create_system == 3 else 0
        )
        if (
            member.orig_filename != member.filename
            or member.is_dir()
            or member.external_attr & 0x10
            or file_type not in (0, stat.S_IFREG)
            or member.file_size <= 0
            or member.flag_bits & 1
        ):
            raise ValueError(f"Olist ZIP member must be a nonempty regular CSV: {member.filename}")


def download_olist(destination: Path | str) -> Path:
    """Download the latest official ZIP and publish its nine raw CSVs.

    The destination must not already exist. Return its absolute path only after
    all members have been extracted and their ZIP CRCs verified. Network, ZIP
    and filesystem failures propagate; temporary downloads and partial extracts
    are removed on failure as well as on success. No credentials are required.
    """
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Olist destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".olist-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / "olist.zip"
        request = Request(
            _OLIST_URL,
            headers={"Accept": "application/zip", "User-Agent": "buy_today-container-build"},
        )
        # urlopen uses GET and follows Kaggle's redirect to signed ZIP storage.
        with urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=_CHUNK_SIZE)

        extracted = temporary / "data"
        extracted.mkdir()
        try:
            with ZipFile(archive_path) as archive:
                members = archive.infolist()
                _validate_members(members)
                for member in members:
                    path = extracted / member.filename
                    with archive.open(member) as source, path.open("xb") as output:
                        # Reading through EOF verifies the member's CRC.
                        shutil.copyfileobj(source, output, length=_CHUNK_SIZE)
                    if path.stat().st_size != member.file_size:
                        raise ValueError(f"Incomplete Olist CSV: {member.filename}")
        except BadZipFile as exc:
            raise ValueError(f"Olist download is not a valid ZIP archive: {exc}") from exc

        extracted.rename(destination)

    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download raw Olist CSVs for the image build.")
    parser.add_argument("destination", type=Path, help="new directory for the nine original CSVs")
    args = parser.parse_args(argv)
    download_olist(args.destination)


if __name__ == "__main__":
    main()
