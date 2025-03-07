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


def format_text(text: str, allowed_specials: list[str] = [], include_digits: bool = True, to_lowercase=True) -> str:
    res_string: str = ""

    for char in text.lower().strip() if to_lowercase else text.strip():
        if not char in list(" " + string.ascii_letters + (string.digits if include_digits else "")) + allowed_specials:
            res_string += " "
            continue

        res_string += char

    return " ".join(filter(lambda v:v, res_string.split(" ")))


if __name__ == '__main__':
    print("This file is not meant to be run as a script.")
