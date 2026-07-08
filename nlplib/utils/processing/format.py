from typing import List, Dict, Any


def json_to_markdown_table(
    data: List[Dict[str, Any]],
    fix_format = False,
    sep_row_full = False
) -> str:
    """
    Converts JSON (a dict) into a Markdown table.
    - all top-level keys encountered are converted into columns
    - non-string values are converted to their representation (repr)

    Parameters
    ----------
    fix_format:
        whether to escape characters that break Markdown tables ("|" -> "\\|", "\n" -> "<br>")
    sep_row_full:
        the separator row cell must contain 3 or more dashes --- per column (per the
        GitHub Flavored Markdown (GFM) spec), i.e. it doesn't have to match the column 
        name's length.
        sep_row_full = True - the length will match the column name (but no shorter than ---), 
        otherwise ---
    """
    assert isinstance(data, list) and isinstance(data[-1], dict)

    # сбор имен полей в порядке появления
    fields = []
    seen = set()
    for row in data:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)

    def format_cell(val: Any) -> str:
        if val is None:
            return ''
        # optional: json.dumps(val, ensure_ascii=False) вместо repr
        s = repr(val) if isinstance(val, (dict, list)) else str(val)      
        if fix_format:
            return s.replace('|', '\\|').replace('\n', '<br>')
        return s

    fields_row = '| ' + ' | '.join(format_cell(f) for f in fields) + ' |'
    sep_row  = (
        '|' + '|'.join('-' * max(3, len(str(field))) for field in fields) + '|' if sep_row_full
        else '|' + '|'.join('---' for _ in fields) + '|'
    )
    
    body_rows = []
    for row in data:
        cells = [format_cell(row.get(f, '')) for f in fields]
        body_rows.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join([fields_row, sep_row] + body_rows)