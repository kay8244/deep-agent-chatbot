"""
Deep Agent 리서치 챗봇 - Streamlit 구현

웹 검색, 요약, 서브에이전트 위임 기능을 포함한 리서치 에이전트의 대화형 인터페이스
기존 deep_agents_from_scratch 패키지의 모듈을 재사용하여 코드 중복을 제거합니다.

기능:
- 일반 대화: LLM 직접 응답 (빠른 답변)
- 딥 리서치: 웹 검색 + 서브에이전트 위임 (심층 조사, 출처 포함)
"""

import json
import os
import re
from pathlib import Path
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# 환경 변수 로드: .env (로컬) → Streamlit Cloud secrets (배포) 순서로 시도
load_dotenv(override=True)

# Streamlit Cloud secrets → os.environ 으로 강제 전달 (override)
for key in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
    try:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
    except Exception:
        pass

from tavily import TavilyClient
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from deep_agents_from_scratch.state import DeepAgentState
from deep_agents_from_scratch.file_tools import ls, read_file, write_file
from deep_agents_from_scratch.todo_tools import write_todos, read_todos
from deep_agents_from_scratch.research_tools import tavily_search, think_tool
import deep_agents_from_scratch.research_tools as _research_tools
from deep_agents_from_scratch.task_tool import _create_task_tool

# 패키지 내부의 모듈 레벨 클라이언트를 올바른 API 키로 재초기화
_research_tools.tavily_client = TavilyClient(
    api_key=os.environ.get("TAVILY_API_KEY", "")
)
_research_tools.summarization_model = init_chat_model(
    model="anthropic:claude-3-5-haiku-20241022",
    temperature=0.0,
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
from deep_agents_from_scratch.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)

# 페이지 설정
st.set_page_config(
    page_title="Deep Agent 리서치 챗봇",
    page_icon="🧠",
    layout="wide",
)

# ── 세션 상태 초기화 ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, sources?, mode?}]
if "files" not in st.session_state:
    st.session_state.files = {}
if "research_stage" not in st.session_state:
    st.session_state.research_stage = "idle"  # "idle" | "plan_pending" | "follow_up"
if "pending_plan" not in st.session_state:
    st.session_state.pending_plan = ""
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""


# ── 캐시된 리소스 ─────────────────────────────────────────────
@st.cache_resource
def _init_model():
    """메인 LLM을 한 번만 초기화합니다."""
    return init_chat_model(
        model="anthropic:claude-haiku-4-5-20251001",
        temperature=0.0,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


@st.cache_resource
def _create_agent():
    """리서치 에이전트를 한 번만 생성합니다."""
    model = _init_model()
    now = datetime.now()

    sub_agent_tools = [tavily_search, think_tool]
    built_in_tools = [ls, read_file, write_file, write_todos, read_todos, think_tool]

    research_sub_agent = {
        "name": "research-agent",
        "description": (
            "Delegate research to the sub-agent researcher. "
            "Only give this researcher one topic at a time."
        ),
        "prompt": RESEARCHER_INSTRUCTIONS.format(
            date=now.strftime("%b %-d, %Y %H:%M:%S (%A)")
        ),
        "tools": ["tavily_search", "think_tool"],
    }

    task_tool = _create_task_tool(
        sub_agent_tools, [research_sub_agent], model, DeepAgentState
    )

    all_tools = sub_agent_tools + built_in_tools + [task_tool]

    subagent_instructions = SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=1,
        max_researcher_iterations=1,
        date=now.strftime("%a %b %-d, %Y"),
    )

    citation_instructions = (
        "# CITATION RULES\n"
        "When writing the final report, you MUST follow these citation rules:\n"
        "- After each factual claim, add an inline citation linking to the source URL.\n"
        "- Use markdown link format: `문장 내용 ([출처제목](URL))`\n"
        "- Example: 2024년 AI 시장 규모는 1조 달러에 달했다 ([Forbes](https://forbes.com/...)).\n"
        "- Every fact must have at least one citation. Do not omit citations.\n"
        "- At the end of the report, include a numbered '## 참고 문헌' section listing all sources.\n"
    )

    system_prompt = "\n\n".join(
        [
            "# TODO MANAGEMENT",
            TODO_USAGE_INSTRUCTIONS,
            "=" * 80,
            "# FILE SYSTEM USAGE",
            FILE_USAGE_INSTRUCTIONS,
            "=" * 80,
            "# SUB-AGENT DELEGATION",
            subagent_instructions,
            "=" * 80,
            citation_instructions,
        ]
    )

    return create_agent(
        model, all_tools, system_prompt=system_prompt, state_schema=DeepAgentState
    )


# ── 리서치 계획 생성 ──────────────────────────────────────────
def _generate_plan(query: str) -> str:
    """사용자 질문을 받아 리서치 계획만 생성합니다 (실제 리서치는 수행하지 않음)."""
    model = _init_model()
    today = datetime.now().strftime("%Y년 %m월 %d일")
    plan_prompt = (
        f"오늘 날짜: {today}\n\n"
        "당신은 리서치 플래너입니다. 아래 질문에 대해 리서치 계획만 작성하세요.\n"
        "실제 리서치는 수행하지 마세요.\n\n"
        "다음 형식으로 번호 매긴 단계별 리스트를 작성하세요:\n"
        "1. [단계 설명]\n"
        "2. [단계 설명]\n"
        "...\n\n"
        f"질문: {query}"
    )
    with st.spinner("📋 리서치 계획 생성 중..."):
        response = model.invoke([HumanMessage(content=plan_prompt)])
    if isinstance(response.content, str):
        return response.content
    parts = [
        item["text"]
        for item in response.content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(parts) if parts else str(response.content)


# ── 유틸리티 ──────────────────────────────────────────────────
def _extract_ai_response(messages: list) -> str:
    """메시지 리스트에서 마지막 AI 응답 텍스트를 추출합니다."""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage) or not msg.content:
            continue
        if isinstance(msg.content, str):
            return msg.content
        parts = [
            item["text"]
            for item in msg.content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if parts:
            return "\n".join(parts)
    return "리서치가 완료되었습니다. 사이드바에서 저장된 파일을 확인해주세요."


def _to_langchain_messages(history: list[dict]) -> list:
    """Streamlit 채팅 히스토리를 LangChain 메시지로 변환합니다."""
    return [
        HumanMessage(content=m["content"])
        if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in history
    ]


def _extract_sources(files: dict) -> list[dict]:
    """파일들에서 출처(URL, 제목) 정보를 추출합니다."""
    sources = []
    seen_urls = set()
    for content in files.values():
        url_match = re.search(r"\*\*URL:\*\*\s*(https?://\S+)", content)
        title_match = re.search(r"# Search Result:\s*(.+)", content)
        if url_match:
            url = url_match.group(1)
            if url not in seen_urls:
                seen_urls.add(url)
                title = title_match.group(1).strip() if title_match else url
                sources.append({"title": title, "url": url})
    return sources


LOCAL_SAVE_DIR = Path("research_outputs")
TEST_CACHE_FILE = LOCAL_SAVE_DIR / "_last_research_cache.json"


def _save_research_cache(response: str, files: dict, sources: list[dict]):
    """마지막 리서치 결과를 JSON 캐시 파일에 저장합니다."""
    LOCAL_SAVE_DIR.mkdir(exist_ok=True)
    cache = {"response": response, "files": files, "sources": sources}
    TEST_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_research_cache() -> tuple[str, dict, list[dict]] | None:
    """캐시된 리서치 결과를 로드합니다. 없으면 None 반환."""
    if not TEST_CACHE_FILE.exists():
        return None
    cache = json.loads(TEST_CACHE_FILE.read_text(encoding="utf-8"))
    return cache["response"], cache["files"], cache["sources"]


def _sanitize_folder_name(query: str) -> str:
    """질문 텍스트를 폴더명으로 사용 가능한 형태로 변환합니다."""
    # 파일시스템에 안전하지 않은 문자 제거
    safe = re.sub(r'[\\/:*?"<>|]', "", query)
    # 공백 정리 및 길이 제한
    safe = safe.strip()[:50].strip()
    return safe or "research"


def _save_files_to_disk(files: dict, query: str = ""):
    """가상 파일시스템의 파일들을 로컬 디스크에 자동 저장합니다.

    research_outputs/<질문요약>/ 하위에 번호 매긴 파일로 저장합니다.
    """
    if not files:
        return
    folder_name = _sanitize_folder_name(query) if query else "research"
    save_dir = LOCAL_SAVE_DIR / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    for idx, (fname, content) in enumerate(files.items(), 1):
        safe_name = Path(fname).name
        numbered_name = f"{idx:02d}_{safe_name}"
        filepath = save_dir / numbered_name
        filepath.write_text(content, encoding="utf-8")


def _render_sources(sources: list[dict]):
    """출처 목록을 렌더링합니다."""
    if not sources:
        return
    with st.expander(f"📚 출처 ({len(sources)}건)", expanded=False):
        for i, src in enumerate(sources, 1):
            st.markdown(f"{i}. [{src['title']}]({src['url']})")


# ── 사이드바 ──────────────────────────────────────────────────
def _render_sidebar() -> tuple[str, bool]:
    """사이드바를 렌더링하고 (모드, 테스트모드 여부)를 반환합니다."""
    with st.sidebar:
        st.header("⚙️ 설정")

        # 모드 선택
        mode = st.radio(
            "대화 모드",
            options=["일반 대화", "딥 리서치"],
            index=0,
            help="일반 대화: 빠른 LLM 직접 응답\n딥 리서치: 웹 검색 + 서브에이전트 심층 조사",
        )

        test_mode = st.toggle(
            "🧪 테스트 모드",
            value=False,
            help="켜면 API 호출 없이 마지막 캐시된 리서치 결과를 재사용합니다.",
        )
        if test_mode:
            has_cache = TEST_CACHE_FILE.exists()
            if has_cache:
                st.caption("✅ 캐시 파일 있음 — API 호출 없이 테스트 가능")
            else:
                st.caption("⚠️ 캐시 없음 — 먼저 딥 리서치를 1회 실행하세요")

        st.divider()

        if st.button("🗑️ 채팅 기록 삭제", use_container_width=True):
            st.session_state.messages = []
            st.session_state.files = {}
            st.session_state.research_stage = "idle"
            st.session_state.pending_plan = ""
            st.session_state.pending_query = ""
            st.rerun()

        st.divider()
        st.header("📁 저장된 파일")

        if st.session_state.files:
            for fname, content in st.session_state.files.items():
                with st.expander(fname):
                    st.code(content, language="markdown")
                    st.download_button(
                        label=f"⬇️ {fname} 다운로드",
                        data=content,
                        file_name=Path(fname).name,
                        mime="text/markdown",
                        key=f"dl_{fname}",
                    )
            st.caption(f"📂 자동 저장 경로: `{LOCAL_SAVE_DIR.resolve()}`")
        else:
            st.info("아직 저장된 파일이 없습니다.")

    return mode, test_mode


# ── 일반 대화 실행 ────────────────────────────────────────────
def _run_normal_chat(history: list[dict]) -> str:
    """LLM에 직접 질문하여 빠른 응답을 받습니다."""
    model = _init_model()
    lc_messages = _to_langchain_messages(history)
    with st.spinner("💬 답변 생성 중..."):
        response = model.invoke(lc_messages)
    if isinstance(response.content, str):
        return response.content
    parts = [
        item["text"]
        for item in response.content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(parts) if parts else str(response.content)


# ── 후속 대화 (리서치 결과 기반) ──────────────────────────────
def _extract_all_urls(content: str) -> list[tuple[str, str]]:
    """파일 내용에서 모든 (제목, URL) 쌍을 추출합니다."""
    urls = []
    seen = set()
    # **URL:** 패턴
    for m in re.finditer(r"\*\*URL:\*\*\s*(https?://\S+)", content):
        url = m.group(1)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    # markdown 링크 패턴 [title](url)
    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^\)]+)\)", content):
        url = m.group(2)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _build_source_map(files: dict) -> str:
    """모든 파일에서 URL을 추출하여 출처 매핑 테이블을 생성합니다."""
    all_urls = {}  # url -> set of file names
    for fname, content in files.items():
        for url in _extract_all_urls(content):
            all_urls.setdefault(url, set()).add(fname)

    if not all_urls:
        return ""

    lines = ["## 출처 URL 목록 (인라인 출처에 반드시 이 URL을 사용하세요)"]
    for i, (url, fnames) in enumerate(all_urls.items(), 1):
        lines.append(f"{i}. {url} (관련 파일: {', '.join(fnames)})")
    return "\n".join(lines)


def _build_file_context(files: dict, max_chars: int = 50000) -> str:
    """리서치 파일 내용을 LLM 컨텍스트 문자열로 변환합니다.

    final/report/findings 파일을 우선 포함하고, 나머지는 공간이 남으면 추가합니다.
    """
    if not files:
        return ""

    # 출처 매핑 테이블을 먼저 포함
    source_map = _build_source_map(files)

    # 우선순위 파일 분류
    priority_keywords = ("final", "report", "findings", "comprehensive")
    priority_files = {}
    other_files = {}
    for fname, content in files.items():
        fname_lower = fname.lower()
        if any(kw in fname_lower for kw in priority_keywords):
            priority_files[fname] = content
        else:
            other_files[fname] = content

    context_parts = [source_map] if source_map else []
    total_chars = len(source_map)

    for group in [priority_files, other_files]:
        for fname, content in group.items():
            urls = _extract_all_urls(content)
            url_line = "출처 URLs: " + ", ".join(urls) if urls else "출처 URL: 없음 (에이전트 생성 요약)"
            entry = f"### 파일: {fname}\n{url_line}\n{content}\n"
            if total_chars + len(entry) > max_chars:
                break
            context_parts.append(entry)
            total_chars += len(entry)

    return "\n".join(context_parts)


def _run_follow_up_chat(history: list[dict], files: dict) -> str:
    """리서치 결과 파일을 컨텍스트로 포함하여 후속 질문에 답변합니다."""
    model = _init_model()
    file_context = _build_file_context(files)

    today = datetime.now().strftime("%Y년 %m월 %d일")
    system_msg = (
        f"오늘 날짜: {today}\n\n"
        "당신은 리서치 결과를 바탕으로 후속 질문에 답변하는 어시스턴트입니다.\n"
        "아래에 리서치에서 수집된 파일 내용이 제공됩니다. "
        "이 자료를 근거로 정확하게 답변하세요.\n\n"
        "## 출처 표기 규칙 (필수)\n"
        "- 모든 사실, 수치, 통계에는 반드시 인라인 출처를 달아야 합니다.\n"
        "- 형식: 문장 내용 ([출처제목](URL))\n"
        "- 예시: DRAM 가격이 15% 상승했다 ([TrendForce](https://trendforce.com/...)).\n"
        "- 반드시 '출처 URL 목록'에 있는 실제 URL을 사용하세요. 파일명을 출처로 쓰지 마세요.\n"
        "- 서로 다른 사실에는 해당 내용이 포함된 서로 다른 출처 URL을 매칭하세요.\n"
        "- 답변 마지막에 '## 참고 문헌' 섹션을 추가하여 사용한 출처를 번호 매겨 나열하세요.\n\n"
        f"## 리서치 자료\n\n{file_context}"
    )

    lc_messages = [HumanMessage(content=system_msg)] + _to_langchain_messages(history)
    with st.spinner("💬 답변 생성 중..."):
        response = model.invoke(lc_messages)
    if isinstance(response.content, str):
        return response.content
    parts = [
        item["text"]
        for item in response.content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(parts) if parts else str(response.content)


# ── 딥 리서치 실행 (스트리밍) ──────────────────────────────────
def _run_deep_research(agent, state: dict) -> tuple[str, dict, list[dict]]:
    """에이전트를 스트리밍 모드로 실행하고 진행 상황을 표시합니다.

    Returns:
        (응답 텍스트, 최종 파일 dict, 출처 리스트)
    """
    final_state = None
    tool_calls_shown = set()
    files_before = set(state.get("files", {}).keys())

    with st.status("🔍 딥 리서치 진행 중...", expanded=True) as status:
        for event in agent.stream(state, stream_mode="values"):
            final_state = event

            for msg in event.get("messages", []):
                if not isinstance(msg, AIMessage):
                    continue
                for tc in getattr(msg, "tool_calls", []) or []:
                    tc_id = tc.get("id", "")
                    if tc_id not in tool_calls_shown:
                        tool_calls_shown.add(tc_id)
                        name = tc.get("name", "unknown")
                        args = tc.get("args", {})
                        detail = ""
                        if "query" in args:
                            detail = f' → "{args["query"]}"'
                        elif "description" in args:
                            desc = args["description"]
                            detail = (
                                f" → {desc[:60]}..."
                                if len(desc) > 60
                                else f" → {desc}"
                            )
                        st.write(f"🔧 `{name}`{detail}")

        status.update(label="✅ 리서치 완료", state="complete")

    if final_state is None:
        return "응답을 생성할 수 없습니다.", state.get("files", {}), []

    response = _extract_ai_response(final_state.get("messages", []))
    files = final_state.get("files", state.get("files", {}))

    # 이번 리서치에서 새로 생성된 파일에서만 출처 추출
    new_files = {k: v for k, v in files.items() if k not in files_before}
    sources = _extract_sources(new_files) if new_files else _extract_sources(files)

    return response, files, sources


# ── 메시지 렌더링 ─────────────────────────────────────────────
def _render_message(msg: dict):
    """메시지 하나를 렌더링합니다 (출처 포함)."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])
        if msg.get("mode") == "딥 리서치":
            st.caption("🔬 딥 리서치")


# ── 메인 앱 ───────────────────────────────────────────────────
def main():
    st.title("🧠 Deep Agent 리서치 챗봇")
    st.caption("웹 검색 · 요약 · 서브에이전트 위임 기능을 갖춘 리서치 에이전트")

    mode, test_mode = _render_sidebar()

    # 채팅 히스토리 표시
    for msg in st.session_state.messages:
        _render_message(msg)

    # 사용자 입력 처리
    if mode == "딥 리서치" and st.session_state.research_stage == "plan_pending":
        placeholder = "승인(진행/네/ok) 또는 수정 내용을 입력하세요..."
    elif mode == "딥 리서치" and st.session_state.research_stage == "follow_up":
        placeholder = "후속 질문을 입력하세요... (새 주제는 '새 리서치'를 입력)"
    elif mode == "딥 리서치":
        placeholder = "리서치할 주제를 입력하세요..."
    else:
        placeholder = "질문을 입력하세요..."

    _needs_rerun = False

    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                if mode == "일반 대화":
                    response = _run_normal_chat(st.session_state.messages)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                elif st.session_state.research_stage == "follow_up":
                    # 딥 리서치: 후속 대화 단계
                    new_research_keywords = {"새 리서치", "새리서치", "new research", "새로운 리서치"}
                    if prompt.strip().lower() in new_research_keywords:
                        st.session_state.research_stage = "idle"
                        st.session_state.files = {}
                        msg = "새로운 리서치를 시작합니다. 리서치할 주제를 입력해주세요."
                        st.markdown(msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": msg}
                        )
                    else:
                        response = _run_follow_up_chat(
                            st.session_state.messages, st.session_state.files
                        )
                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                elif st.session_state.research_stage == "idle" and test_mode:
                    # 테스트 모드: 캐시에서 바로 결과 로드 (API 호출 없음)
                    cached = _load_research_cache()
                    if cached:
                        response, files, sources = cached
                        st.session_state.files = files
                        st.info("🧪 테스트 모드: 캐시된 리서치 결과를 표시합니다.")
                        st.markdown(response)
                        if sources:
                            _render_sources(sources)
                        st.caption("🔬 딥 리서치")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response,
                                "sources": sources,
                                "mode": "딥 리서치",
                            }
                        )
                        st.session_state.research_stage = "follow_up"
                        _needs_rerun = True
                    else:
                        msg = "⚠️ 캐시된 결과가 없습니다. 테스트 모드를 끄고 딥 리서치를 1회 실행해주세요."
                        st.warning(msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": msg}
                        )

                elif st.session_state.research_stage == "idle":
                    # 딥 리서치: 계획 생성 단계
                    plan = _generate_plan(prompt)
                    plan_message = (
                        f"**📋 리서치 계획**\n\n{plan}\n\n---\n"
                        "이 계획대로 진행할까요? "
                        "승인하려면 **진행/네/ok** 등을 입력하고, "
                        "수정이 필요하면 수정 내용을 입력해주세요."
                    )
                    st.markdown(plan_message)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": plan_message}
                    )
                    st.session_state.research_stage = "plan_pending"
                    st.session_state.pending_plan = plan
                    st.session_state.pending_query = prompt

                else:
                    # 딥 리서치: 승인/수정 처리 단계
                    approval_keywords = {
                        "진행", "네", "좋아", "ㅇㅇ", "ok", "yes",
                        "응", "좋아요", "확인", "ㅇ", "고", "시작",
                    }
                    user_input = prompt.strip().lower()

                    if user_input in approval_keywords:
                        plan = st.session_state.pending_plan
                    else:
                        plan = prompt  # 수정 내용을 새 계획으로 사용

                    # 테스트 모드 + 캐시 있음: 캐시된 결과 사용
                    cached = _load_research_cache() if test_mode else None
                    if cached:
                        response, files, sources = cached
                        st.info("🧪 테스트 모드: 캐시된 결과를 표시합니다.")
                    else:
                        # 실제 리서치 실행 (테스트 모드여도 캐시 없으면 fallback)
                        if test_mode:
                            st.warning("🧪 캐시 없음 — 실제 리서치를 실행합니다.")
                        research_prompt = (
                            f"사용자 질문: {st.session_state.pending_query}\n\n"
                            f"리서치 계획:\n{plan}\n\n"
                            "위 계획에 따라 리서치를 수행하세요."
                        )

                        agent = _create_agent()
                        agent_state = {
                            "messages": [HumanMessage(content=research_prompt)],
                            "files": st.session_state.files,
                        }

                        response, files, sources = _run_deep_research(
                            agent, agent_state
                        )
                        _save_files_to_disk(files, st.session_state.pending_query)
                        _save_research_cache(response, files, sources)

                    st.session_state.files = files

                    st.markdown(response)
                    if sources:
                        _render_sources(sources)
                    st.caption("🔬 딥 리서치")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "sources": sources,
                            "mode": "딥 리서치",
                        }
                    )

                    # 리서치 완료 → 후속 대화 모드로 전환
                    st.session_state.research_stage = "follow_up"
                    st.session_state.pending_plan = ""
                    st.session_state.pending_query = ""
                    _needs_rerun = True

            except Exception as e:
                error_msg = f"오류가 발생했습니다: {e}"
                st.error(error_msg)
                st.exception(e)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

    # try/except 밖에서 rerun하여 RerunException이 잡히지 않도록 함
    if _needs_rerun:
        st.rerun()


if __name__ == "__main__":
    main()
