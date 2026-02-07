"""
Deep Agent 리서치 챗봇 - Streamlit 구현

웹 검색, 요약, 서브에이전트 위임 기능을 포함한 리서치 에이전트의 대화형 인터페이스
기존 deep_agents_from_scratch 패키지의 모듈을 재사용하여 코드 중복을 제거합니다.

기능:
- 일반 대화: LLM 직접 응답 (빠른 답변)
- 딥 리서치: 웹 검색 + 서브에이전트 위임 (심층 조사, 출처 포함)
"""

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
    st.session_state.research_stage = "idle"  # "idle" | "plan_pending"
if "pending_plan" not in st.session_state:
    st.session_state.pending_plan = ""
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""


# ── 캐시된 리소스 ─────────────────────────────────────────────
@st.cache_resource
def _init_model():
    """메인 LLM을 한 번만 초기화합니다."""
    return init_chat_model(
        model="anthropic:claude-sonnet-4-5",
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
        max_concurrent_research_units=3,
        max_researcher_iterations=3,
        date=now.strftime("%a %b %-d, %Y"),
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
        ]
    )

    return create_agent(
        model, all_tools, system_prompt=system_prompt, state_schema=DeepAgentState
    )


# ── 리서치 계획 생성 ──────────────────────────────────────────
def _generate_plan(query: str) -> str:
    """사용자 질문을 받아 리서치 계획만 생성합니다 (실제 리서치는 수행하지 않음)."""
    model = _init_model()
    plan_prompt = (
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


def _save_files_to_disk(files: dict):
    """가상 파일시스템의 파일들을 로컬 디스크에 자동 저장합니다."""
    if not files:
        return
    LOCAL_SAVE_DIR.mkdir(exist_ok=True)
    for fname, content in files.items():
        safe_name = Path(fname).name  # 경로 트래버설 방지
        filepath = LOCAL_SAVE_DIR / safe_name
        filepath.write_text(content, encoding="utf-8")


def _render_sources(sources: list[dict]):
    """출처 목록을 렌더링합니다."""
    if not sources:
        return
    with st.expander(f"📚 출처 ({len(sources)}건)", expanded=False):
        for i, src in enumerate(sources, 1):
            st.markdown(f"{i}. [{src['title']}]({src['url']})")


# ── 사이드바 ──────────────────────────────────────────────────
def _render_sidebar() -> str:
    """사이드바를 렌더링하고 선택된 모드를 반환합니다."""
    with st.sidebar:
        st.header("⚙️ 설정")

        # 모드 선택
        mode = st.radio(
            "대화 모드",
            options=["일반 대화", "딥 리서치"],
            index=0,
            help="일반 대화: 빠른 LLM 직접 응답\n딥 리서치: 웹 검색 + 서브에이전트 심층 조사",
        )

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

    return mode


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

    mode = _render_sidebar()

    # 채팅 히스토리 표시
    for msg in st.session_state.messages:
        _render_message(msg)

    # 사용자 입력 처리
    if mode == "딥 리서치" and st.session_state.research_stage == "plan_pending":
        placeholder = "승인(진행/네/ok) 또는 수정 내용을 입력하세요..."
    elif mode == "딥 리서치":
        placeholder = "리서치할 주제를 입력하세요..."
    else:
        placeholder = "질문을 입력하세요..."

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

                    # 원본 질문 + 확정된 계획을 에이전트에 전달
                    research_prompt = (
                        f"사용자 질문: {st.session_state.pending_query}\n\n"
                        f"리서치 계획:\n{plan}\n\n"
                        "위 계획에 따라 리서치를 수행하세요."
                    )

                    agent = _create_agent()
                    # 계획 승인 과정의 대화는 제외하고 리서치 프롬프트만 전달
                    agent_state = {
                        "messages": [HumanMessage(content=research_prompt)],
                        "files": st.session_state.files,
                    }

                    response, files, sources = _run_deep_research(
                        agent, agent_state
                    )
                    st.session_state.files = files
                    _save_files_to_disk(files)

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

                    # 상태 초기화
                    st.session_state.research_stage = "idle"
                    st.session_state.pending_plan = ""
                    st.session_state.pending_query = ""

            except Exception as e:
                error_msg = f"오류가 발생했습니다: {e}"
                st.error(error_msg)
                st.exception(e)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


if __name__ == "__main__":
    main()
