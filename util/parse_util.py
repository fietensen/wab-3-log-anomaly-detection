import re
import string


def process_line(text: str, regex: re.Pattern, sel_groups: list[int] | None = None) -> tuple[str] | None:
    re_match = regex.match(text.strip())

    if not re_match:
        return None

    groups = re_match.groups()

    if not sel_groups:
        return tuple(groups)

    return tuple([groups[i] for i in sel_groups])


def format_text(text: str, allowed_specials: list[str] = []) -> str:
    res_string: str = ""

    for char in text.lower().strip():
        if not char in list(" " + string.ascii_lowercase + string.digits) + allowed_specials:
            continue

        res_string += char

    return res_string


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
