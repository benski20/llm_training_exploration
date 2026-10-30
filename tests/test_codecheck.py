"""vt/codecheck.py: Python and JSON in a reply are parsed, never run."""
from vt.codecheck import check_code


def test_parses_python_and_json_blocks():
    good = "Here:\n```python\ndef f(x):\n    return x * 2\n```\nand\n```json\n{\"a\": 1}\n```"
    assert check_code(good) == [{"language": "python", "ok": True}, {"language": "json", "ok": True}]
    bad = check_code("```python\ndef f(x)\n    return x\n```")
    assert not bad[0]["ok"] and "line 1" in bad[0]["error"]
    assert not check_code("```json\n{'a': 1}\n```")[0]["ok"]


def test_other_languages_and_prose_are_skipped():
    assert check_code("```js\nconst x = ;\n```") == []
    assert check_code("No code here, just `inline` text.") == []


def test_never_runs_the_code(tmp_path):
    marker = tmp_path / "ran"
    check_code(f"```python\nopen({str(marker)!r}, 'w').write('x')\n```")
    assert not marker.exists()
