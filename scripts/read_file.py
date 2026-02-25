"""
dialogue + 요약 파일 묶음(dialouge_*.txt + *_{timestamp}.md|xml|jsonl|html)을
sudo로 읽고, instruction.txt 파일을 파싱해 summary_type+확장자에 맞는
instruction을 system 메시지로 넣어 ChatLM 스타일로 PyArrow 테이블에 저장하는 모듈.
messages = [system(instruction), user(dialogue), assistant(요약)].
"""
import logging
import re
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import pyarrow as pa

LOG = logging.getLogger(__name__)


SUMMARY_COLUMNS = [
    "soap_assessment",
    "soap_object",
    "soap_plan",
    "soap_report",
    "soap_subject",
    "summary",
    "intents",
    "report",
    "treatment",
    "cost_estimation",
]

# merged_data: dialouge(오타), final_data: dialogue(정자) 둘 다 dialogue 파일로 인식
DIALOGUE_PREFIXES = ("dialouge", "dialogue")
DEFAULT_INSTRUCTION_FILE = (
    "/home/glory/summary_format/grpo/data/instructions_dataset/instruction.txt"
)
# 타임스탬프: 공백/밑줄/T 허용. 확장자 앞에 _table, _txt 붙는 경우도 허용
# 예: soap_assessment_2025-03-24 15:44:57.md, soap_assessment_2025-03-24 15:44:57_table.html
TIMESTAMP_PATTERN = re.compile(
    r"^(.+)_(\d{4}-\d{2}-\d{2}[ \t_T]\d{2}:\d{2}:\d{2})(?:_(table|txt))?\.(txt|md|xml|jsonl|html)$"
)

CHATLM_MESSAGES_TYPE = pa.list_(
    pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
    ])
)


# ---------------------------------------------------------------------------
# sudo 유틸
# ---------------------------------------------------------------------------

def _run_sudo_cmd(cmd_args, capture_stdout=True):
    proc = subprocess.run(
        ["sudo"] + cmd_args,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0 and proc.stderr:
        raise RuntimeError(
            f"명령 실패 (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout if capture_stdout else None


def list_dir_with_sudo(dirpath: str) -> list[str]:
    out = _run_sudo_cmd(["ls", "-1", dirpath])
    if not out:
        LOG.warning("[list_dir] dirpath=%r → 빈 결과 (권한/경로 확인)", dirpath)
        return []
    names = [line.strip() for line in out.splitlines() if line.strip()]
    LOG.debug("[list_dir] dirpath=%r → %d개 항목", dirpath, len(names))
    return names


def read_file_with_sudo(filepath: str) -> str:
    out = _run_sudo_cmd(["cat", "--", filepath])
    return out or ""


# ---------------------------------------------------------------------------
# instruction.txt 파싱
# ---------------------------------------------------------------------------

def parse_instruction_file(filepath: str) -> dict[str, str]:
    """
    instruction.txt를 파싱해 {key: instruction_text} 딕셔너리 반환.

    파일 포맷:
        intents_table.md
        ### Instruction:
        <여러 줄 instruction 본문>

        intents_txt.md
        ### Instruction:
        ...

    key 예: 'intents_table.md', 'soap_assessment.jsonl' 등
    """
    content = read_file_with_sudo(filepath)
    lines = content.split("\n")
    sections: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if (
            line
            and not line.startswith("#")
            and not line.startswith("-")
            and "." in line
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("### Instruction:")
        ):
            key = line
            after_header = lines[i + 1].strip()[len("### Instruction:"):].strip()
            body_parts = []
            if after_header:
                body_parts.append(after_header)
            j = i + 2
            while j < len(lines):
                peek = lines[j].strip()
                if (
                    peek
                    and not peek.startswith("#")
                    and not peek.startswith("-")
                    and "." in peek
                    and j + 1 < len(lines)
                    and lines[j + 1].strip().startswith("### Instruction:")
                ):
                    break
                body_parts.append(lines[j])
                j += 1
            sections[key] = "\n".join(body_parts).strip()
            i = j
            continue
        i += 1
    return sections


# instruction/파일명 형식 접미사: _txt, _table
FORMAT_SUFFIXES = ("_txt", "_table")


def _normalize_prefix(prefix: str) -> tuple[str | None, str]:
    """
    파일명 prefix에서 (summary_type, format_suffix) 추출.
    예: 'soap_assessment_txt' → ('soap_assessment', 'txt')
        'soap_assessment_table' → ('soap_assessment', 'table')
        'soap_assessment' → ('soap_assessment', '')
    """
    for suf in FORMAT_SUFFIXES:
        if prefix.endswith(suf):
            stype = prefix[: -len(suf)]
            if stype in SUMMARY_COLUMNS:
                return stype, suf.lstrip("_")  # 'txt', 'table'
            return None, ""
    if prefix in SUMMARY_COLUMNS:
        return prefix, ""
    return None, ""


def _extract_summary_type_and_format_from_key(key: str) -> tuple[str | None, str, str | None]:
    """instruction key에서 (summary_type, format_suffix, ext) 추출.
    예: 'soap_assessment_txt.md' → ('soap_assessment', 'txt', 'md')
        'soap_assessment_table.html' → ('soap_assessment', 'table', 'html')
    """
    if "." not in key:
        return None, "", None
    stem, ext = key.rsplit(".", 1)
    format_suffix = ""
    for suf in FORMAT_SUFFIXES:
        if stem.endswith(suf):
            stype = stem[: -len(suf)]
            if stype in SUMMARY_COLUMNS:
                return stype, suf.lstrip("_"), ext
            return None, "", ext
    if stem in SUMMARY_COLUMNS:
        return stem, "", ext
    for col in sorted(SUMMARY_COLUMNS, key=len, reverse=True):
        if stem == col or stem.startswith(col + "_"):
            return col, "", ext
    return None, "", ext


def build_instruction_map(
    sections: dict[str, str],
) -> dict[tuple[str, str, str], list[str]]:
    """
    파싱된 sections → {(summary_type, ext, format_suffix): [instruction_text, ...]} 매핑.
    format_suffix: '' | 'txt' | 'table'
    """
    mapping: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for key, text in sections.items():
        stype, format_suffix, ext = _extract_summary_type_and_format_from_key(key)
        if stype and ext and text:
            mapping[(stype, ext, format_suffix)].append(text)
    return dict(mapping)


def find_instruction_for_summary(
    inst_map: dict[tuple[str, str, str], list[str]],
    summary_type: str,
    ext: str,
    format_suffix: str = "",
) -> str:
    """
    summary_type + ext + format_suffix(_txt, _table 등)로 instruction 반환.
    """
    key = (summary_type, ext, format_suffix)
    if key in inst_map and inst_map[key]:
        return inst_map[key][0]
    if not format_suffix:
        for (stype, e, fmt), texts in inst_map.items():
            if stype == summary_type and texts:
                return texts[0]
    return ""


# ---------------------------------------------------------------------------
# merged_data / final_data 파일 수집 (매칭 키 = 시간 타임스탬프)
# ---------------------------------------------------------------------------

def _normalize_ts(ts: str) -> str:
    """그룹 키용 타임스탬프 정규화. 공백/탭을 _로 통일해 디렉터리 구조와 무관하게 같은 시간이면 같은 키."""
    return re.sub(r"[\s\t]+", "_", ts.strip())


def parse_timestamp_and_type(
    filename: str,
) -> tuple[str | None, str | None, str | None, str]:
    """(prefix, ts, ext, ext_suffix). ext_suffix는 확장자 앞 _table/_txt 또는 ''."""
    m = TIMESTAMP_PATTERN.match(filename.strip())
    if not m:
        return None, None, None, ""
    prefix, ts, ext_suffix, ext = m.groups()
    return prefix, ts, ext, (ext_suffix or "")


def _collect_from_dir(
    base: Path,
    names: list[str],
) -> dict[str, dict[str, list[tuple[str, str, str]]]]:
    """한 디렉터리 내 파일명 목록에서 타임스탬프별 그룹 반환.
    값: dialogue → [(path, ext, '')], summary_type → [(path, ext, format_suffix), ...]
    """
    groups: dict[str, dict[str, list[tuple[str, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    no_match: list[str] = []
    skipped_prefix: list[str] = []
    for name in names:
        parsed = parse_timestamp_and_type(name)
        if parsed[1] is None:
            no_match.append(name)
            continue
        prefix, ts, ext, ext_suffix = parsed  # ext_suffix = 확장자 앞 _table/_txt 또는 ''
        key = _normalize_ts(ts)
        full_path = str(base / name)
        if prefix in DIALOGUE_PREFIXES and ext == "txt":
            groups[key]["dialogue"].append((full_path, ext, ""))
            LOG.debug("[collect] dialogue: prefix=%r ts=%r key=%r ext=%r", prefix, ts, key, ext)
        else:
            stype, prefix_fmt = _normalize_prefix(prefix)
            format_suffix = ext_suffix or prefix_fmt  # 확장자 앞 _table/_txt 우선, 없으면 prefix에서
            if stype is not None:
                groups[key][stype].append((full_path, ext, format_suffix))
                LOG.debug(
                    "[collect] summary: prefix=%r → stype=%r fmt=%r key=%r ext=%r",
                    prefix, stype, format_suffix, key, ext,
                )
            else:
                skipped_prefix.append(prefix)
    if no_match and LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "[collect] %s 타임스탬프 패턴 미매칭 (처음 20개): %s",
            base, no_match[:20],
        )
    if skipped_prefix and LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "[collect] %s SUMMARY_COLUMNS 미포함 prefix (처음 20개): %s",
            base, skipped_prefix[:20],
        )
    return dict(groups)


def collect_dialogue_summary_groups(
    dirpath: str,
) -> dict[str, dict[str, list[tuple[str, str, str]]]]:
    """
    매칭 키 = 시간(타임스탬프)만 사용. 디렉터리 구조와 무관하게 같은 시간이면 한 그룹.
    최상위 + 모든 직하위 디렉터리에서 수집 후 정규화된 타임스탬프로 merge.
    반환: { normalized_ts: { 'dialogue': [(path,ext,'')], 'soap_assessment': [...], ... } }
    """
    base = Path(dirpath).resolve()
    LOG.info("[collect_groups] dirpath=%r (매칭=시간 타임스탬프)", dirpath)
    names = list_dir_with_sudo(str(base))
    LOG.info("[collect_groups] 최상위 항목 수: %d", len(names))
    if names and LOG.isEnabledFor(logging.DEBUG):
        LOG.debug("[collect_groups] 최상위 샘플(25개): %s", names[:25])

    all_groups: dict[str, dict[str, list[tuple[str, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    def merge_into(target: dict, source: dict) -> None:
        for key, paths in source.items():
            for k, entries in paths.items():
                target[key][k].extend(entries)

    # 1) 최상위 디렉터리에서 수집 (파일명에 타임스탬프 있으면 정규화 키로 저장)
    top_groups = _collect_from_dir(base, names)
    merge_into(all_groups, top_groups)
    LOG.info("[collect_groups] 최상위: 그룹 %d개 (키=정규화 타임스탬프)", len(top_groups))

    # 2) 직하위 디렉터리 모두 스캔 → 같은 시간(키)이면 merge (디렉터리만 다르면 한 그룹으로)
    for name in names:
        subpath = base / name
        try:
            sub_names = list_dir_with_sudo(str(subpath))
        except Exception as e:
            LOG.debug("[collect_groups] 하위 읽기 스킵: %s → %s", subpath, e)
            continue
        sub_groups = _collect_from_dir(subpath, sub_names)
        if sub_groups:
            merge_into(all_groups, sub_groups)
            LOG.info(
                "[collect_groups] 하위 %s: 항목 %d개, 그룹 %d개 (merge됨)",
                name, len(sub_names), len(sub_groups),
            )

    result = {
        key: dict(paths)
        for key, paths in all_groups.items()
        if paths.get("dialogue")
    }
    LOG.info("[collect_groups] 최종: 그룹 %d개 (dialogue 있는 것만, 키=시간)", len(result))
    if not result and names:
        LOG.warning(
            "[collect_groups] 그룹 0개. 파일명 타임스탬프 패턴·dialogue/dialouge_*.txt·SUMMARY_COLUMNS 확인. 샘플: %s",
            names[:15],
        )
    return result


# ---------------------------------------------------------------------------
# ChatLM 메시지 빌드
# ---------------------------------------------------------------------------

def build_chatlm_messages(
    instruction: str,
    dialogue_text: str,
    assistant_content: str,
) -> list[dict]:
    messages = []
    if (instruction or "").strip():
        messages.append({"role": "system", "content": instruction.strip()})
    messages.append({"role": "user", "content": dialogue_text})
    messages.append({"role": "assistant", "content": assistant_content})
    return messages


# ---------------------------------------------------------------------------
# 테이블 생성 / 저장
# ---------------------------------------------------------------------------

def load_dialogue_summary_chatlm_table(
    dirpath: str,
    instruction_file: str | None = None,
    summary_columns: list[str] | None = None,
) -> pa.Table:
    """
    dirpath의 dialogue+요약 → ChatLM 스타일 PyArrow 테이블.
    instruction.txt를 파싱해 summary_type+ext에 맞는 instruction을 system 메시지로 넣음.
    """
    cols = summary_columns or SUMMARY_COLUMNS
    inst_path = instruction_file or DEFAULT_INSTRUCTION_FILE

    sections = parse_instruction_file(inst_path)
    inst_map = build_instruction_map(sections)
    LOG.info("[load_table] instruction 섹션 %d개, inst_map 키 %d개", len(sections), len(inst_map))

    groups = collect_dialogue_summary_groups(dirpath)
    LOG.info("[load_table] dirpath=%r → 그룹 %d개", dirpath, len(groups))
    rows = []

    for dialogue_id, paths in sorted(groups.items()):
        dialogue_list = paths.get("dialogue") or []
        if not dialogue_list:
            LOG.debug("[load_table] dialogue_id=%r dialogue 목록 없음, 스킵", dialogue_id)
            continue
        dialogue_path, _, _ = dialogue_list[0]
        try:
            dialogue_text = read_file_with_sudo(dialogue_path)
        except Exception as e:
            LOG.warning("[load_table] dialogue_id=%r 파일 읽기 실패: %s → %s", dialogue_id, dialogue_path, e)
            continue
        n_rows_this = 0
        for summary_type in cols:
            for summary_path, summary_ext, format_suffix in paths.get(
                summary_type, []
            ):
                try:
                    content = read_file_with_sudo(summary_path).strip()
                except Exception as e:
                    LOG.debug("[load_table] 요약 읽기 실패: %s → %s", summary_path, e)
                    continue
                if not content:
                    continue
                instruction = find_instruction_for_summary(
                    inst_map, summary_type, summary_ext, format_suffix
                )
                if not (instruction or "").strip() and LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug(
                        "[load_table] instruction 없음: stype=%r ext=%r fmt=%r",
                        summary_type, summary_ext, format_suffix,
                    )
                messages = build_chatlm_messages(
                    instruction, dialogue_text, content
                )
                rows.append({
                    "dialogue_id": dialogue_id,
                    "summary_type": summary_type,
                    "messages": messages,
                })
                n_rows_this += 1
        LOG.debug(
            "[load_table] dialogue_id=%r dialogue_path=%s → 추가 행 %d개",
            dialogue_id, dialogue_path, n_rows_this,
        )

    if not rows:
        LOG.warning("[load_table] 생성된 행 0개 (그룹은 %d개였음). 요약 파일 읽기 실패 또는 내용 없음 확인.", len(groups))
        return pa.table({
            "dialogue_id": pa.array([], type=pa.string()),
            "summary_type": pa.array([], type=pa.string()),
            "messages": pa.array([], type=CHATLM_MESSAGES_TYPE),
        })
    LOG.info("[load_table] 총 행 수: %d", len(rows))
    return pa.Table.from_pylist(rows)


def save_dialogue_summary_arrow(
    dirpath: str,
    out_path: str,
    instruction_file: str | None = None,
    summary_columns: list[str] | None = None,
) -> None:
    table = load_dialogue_summary_chatlm_table(
        dirpath,
        instruction_file=instruction_file,
        summary_columns=summary_columns,
    )
    with pa.OSFile(out_path, "wb") as f:
        with pa.ipc.RecordBatchFileWriter(f, table.schema) as writer:
            writer.write_table(table)


# ---------------------------------------------------------------------------
# 진단: instruction.txt 파싱 결과 확인
# ---------------------------------------------------------------------------

def print_instruction_map(instruction_file: str | None = None) -> None:
    inst_path = instruction_file or DEFAULT_INSTRUCTION_FILE
    sections = parse_instruction_file(inst_path)
    inst_map = build_instruction_map(sections)
    print(f"instruction.txt 파싱 결과: {len(sections)} 섹션")
    print(f"(summary_type, ext, format_suffix) 매핑: {len(inst_map)} 개\n")
    for (stype, ext, fmt), texts in sorted(inst_map.items()):
        preview = texts[0][:80] + "..." if len(texts[0]) > 80 else texts[0]
        print(f"  ({stype!r}, {ext!r}, {fmt!r}) [{len(texts)}개] → {preview!r}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    debug = "--debug" in sys.argv
    if debug:
        sys.argv = [a for a in sys.argv if a != "--debug"]
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s [read_file] %(message)s",
    )
    if debug:
        LOG.info("디버그 로그 활성화 (--debug)")

    dirpath = "/home/glory/summary_format/grpo/data/sft_data/sft_train_data"
    instruction_file = DEFAULT_INSTRUCTION_FILE
    out_path = "dialogue_summary_train_sft.arrow"

    if "--parse-instructions" in sys.argv:
        inst = sys.argv[sys.argv.index("--parse-instructions") + 1] \
            if len(sys.argv) > sys.argv.index("--parse-instructions") + 1 \
            else instruction_file
        print_instruction_map(inst)
        sys.exit(0)

    if len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
        dirpath = sys.argv[1]
    if len(sys.argv) >= 3 and not sys.argv[2].startswith("--"):
        out_path = sys.argv[2]
    if len(sys.argv) >= 4 and not sys.argv[3].startswith("--"):
        instruction_file = sys.argv[3]

    save_dialogue_summary_arrow(
        dirpath, out_path, instruction_file=instruction_file
    )
    table = load_dialogue_summary_chatlm_table(
        dirpath, instruction_file=instruction_file
    )
    print("rows:", table.num_rows, "columns:", table.column_names)

    n_head = min(2, table.num_rows)
    if n_head:
        print("\n--- head (first", n_head, "rows) ---")
        for i in range(n_head):
            row = table.slice(i, 1)
            did = row.column("dialogue_id")[0].as_py()
            stype = row.column("summary_type")[0].as_py()
            msgs = row.column("messages")[0].as_py()
            print(f"[{i}] dialogue_id={did!r} summary_type={stype!r}")
            for j, m in enumerate(msgs):
                role = m.get("role", "")
                raw = m.get("content") or ""
                preview = raw[:80] + "..." if len(raw) > 80 else raw
                print(f"    messages[{j}] role={role!r} content={preview!r}")
            print()
