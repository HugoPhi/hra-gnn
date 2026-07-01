from __future__ import annotations

import argparse

import olefile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract saved TeX annotations from MathType objects in a .doc file"
    )
    parser.add_argument("document")
    arguments = parser.parse_args()

    marker = b"TeX Input Language\x00"
    with olefile.OleFileIO(arguments.document) as document:
        for path in document.listdir():
            if path[-1] != "Equation Native":
                continue
            data = document.openstream(path).read()
            start = data.find(marker)
            tex = ""
            if start >= 0:
                start += len(marker)
                tex = (
                    data[start:].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                )
            print(f"{path[-2]}\t{tex}")


if __name__ == "__main__":
    main()
