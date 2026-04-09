import asyncio
import json
import logging
import re
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _extract_code(solution_str: str) -> tuple[str | None, bool]:
    text = solution_str or ""

    # Never extract code inside <think>...</think>. If think exists, only
    # allow extraction from the content after the final closing </think>.
    if re.search(r"<think\b", text, flags=re.IGNORECASE):
        think_ends = list(re.finditer(r"</think\s*>", text, flags=re.IGNORECASE))
        if not think_ends:
            return None, False
        text = text[think_ends[-1].end() :]

    blocks = re.findall(
        r"```([^\n`]*)\n(.*?)```",
        text,
        flags=re.DOTALL,
    )
    if blocks:
        cpp_blocks = [code for lang, code in blocks if (lang or "").strip().lower() == "cpp"]
        if cpp_blocks:
            return cpp_blocks[-1].strip(), True
        return blocks[-1][1].strip(), True
    return None, False


def _resolve_problem_fields(ground_truth: Any) -> tuple[str | None, Any]:
    gt_dict = _to_dict(ground_truth)
    problem_id = gt_dict.get("problem_id") or gt_dict.get("oj_problem_id")
    case_ids = gt_dict.get("case_ids")
    if case_ids is None:
        case_ids = gt_dict.get("oj_case_ids")
    if case_ids is None:
        case_ids = "all"
    return problem_id, case_ids


async def _post_with_retry(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    last_exc: Exception | None = None

    for i in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if 200 <= resp.status < 300:
                        return await resp.json()
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
        except Exception as exc:
            last_exc = exc
            if i < max_retries - 1:
                await asyncio.sleep(retry_backoff_s * (i + 1))

    raise RuntimeError(f"sandbox request failed after retries: {last_exc}")


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    *,
    sandbox_url: str,
    data_dir: str | None = None,
    compile_timeout: float = 30.0,
    run_timeout: float | None = None,
    time_limit_multiplier: float = 2.0,
    memory_limit_MB: int | None = None,
    enable_msvc_i64_compat: bool = True,
    include_details: bool = False,
    request_timeout_s: float = 120.0,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
) -> float:
    # Keep default behavior for non-code datasets to avoid breaking mixed-data training.
    if data_source not in {"codecontests", "apps", "codeforces", "taco"}:
        return 0.0

    problem_id, case_ids = _resolve_problem_fields(ground_truth)
    if not problem_id:
        return 0.0

    code, extracted = _extract_code(solution_str)
    if not extracted or code is None:
        return 0.0

    payload: dict[str, Any] = {
        "problem_id": str(problem_id),
        "case_ids": case_ids,
        "code": code,
        "language": "cpp",
        "compile_timeout": compile_timeout,
        "time_limit_multiplier": time_limit_multiplier,
        "enable_msvc_i64_compat": enable_msvc_i64_compat,
        "include_details": include_details,
    }
    if data_dir:
        payload["data_dir"] = data_dir
    if run_timeout is not None:
        payload["run_timeout"] = run_timeout
    if memory_limit_MB is not None:
        payload["memory_limit_MB"] = memory_limit_MB

    try:
        resp = await _post_with_retry(
            url=sandbox_url,
            payload=payload,
            timeout_s=request_timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
    except Exception as exc:
        logger.warning(
            "mounted_oj request failed after retries, fallback score=0. "
            "problem_id=%s, case_ids=%s, err=%r",
            problem_id,
            case_ids,
            exc,
        )
        return 0.0

    total_score = float(resp.get("total_score", 0.0))
    max_score = float(resp.get("max_score", 0.0))
    normalized_score = total_score / max_score if max_score > 0 else 0.0
    return normalized_score


def compute_score_sync(*args, **kwargs):
    """Synchronous wrapper for compute_score used by default router paths."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(compute_score(*args, **kwargs))
    else:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(compute_score(*args, **kwargs))).result()
