"""LLM-Beginner 各任务自检脚本共用的运行壳。

每个 task 的 eval/run.py 只负责定义若干 test_* 函数（返回 dict，至少含 "test"
名称与三态 "pass"：True 通过 / None 跳过 / False 失败），最后调用 run_tests()。
run_tests 统一处理：UTF-8 控制台、异常兜底、三态打印、写 eval/result.json。
"""
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

_TAGS = {True: "[通过]", None: "[跳过]", False: "[失败]"}


def setup_console() -> None:
    """Windows 控制台默认 CP936 会让中文输出乱码；result.json 仍是 UTF-8。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_tests(tests: list[Callable[[], dict]], root: Path) -> list[dict]:
    """依次运行 tests，打印三态结果并写入 root/eval/result.json，返回结果列表。"""
    setup_console()
    results: list[dict] = []
    for fn in tests:
        try:
            r = fn()
        except Exception as e:
            r = {"test": fn.__name__.removeprefix("test_"),
                 "pass": False, "error": str(e),
                 "trace": traceback.format_exc().splitlines()[-3:]}
        results.append(r)
        tag = _TAGS.get(r.get("pass"), "[失败]")
        print(f"{tag} {r['test']}: {json.dumps(r, ensure_ascii=False)}")

    out = root / "eval" / "result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(root)}")
    return results
