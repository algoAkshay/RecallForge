import asyncio
import datetime
import logging
import time
import uuid

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from streamlit import dialog as st_dialog

from agents.agent import synthesis_agent
from agents.fallback import SynthesisResult, synthesize_with_single_fallback
from features import ENABLE_PDF_FEATURES
from storage.chat_history import add_message, create_thread, delete_thread, initialize_database, list_threads, load_messages, rename_thread
from tools.presentation import render_authoritative_sources, sanitize_answer_markdown
from tools.provenance import citation_collection, evidence_from_document, format_evidence, register_evidence, remove_invalid_citations, validate_citations
from tools.relevance_search import is_document_context_query, retrieve_active_document_candidates, retrieve_memory_candidates
from tools.routing import choose_route
from tools.web_scraper import _fetch_sites
from ui.chat_export import export_chat_markdown, safe_export_filename
from ui.components import append_assistant_message, clear_deleted_active_thread, complete_assistant_message, render_response, render_sidebar
from ui.debug_panel import parse_web_diagnostics
from ui.research_control import consume_search_fresh, forced_web_decision
from ui.styles import workspace_css


logger = logging.getLogger(__name__)
st.set_option("client.showErrorDetails", True)

PROMPT_TEMPLATE = """
Give a detailed answer to the question and explain it clearly. If the supplied evidence is insufficient, return exactly this sentence and nothing else: "I couldn't find enough relevant evidence to answer this confidently."
Use inline [S#] citations for factual claims when supplied. Do not generate Sources, References, or Bibliography sections; the application renders sources separately.
Current Date and Time: {current_time}
question: {question}
"""


@st_dialog("Error Recovery")
def error_fallback(_error):
    st.error("Research could not be completed. Please try again.")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()


@st_dialog("Delete this chat?")
def delete_thread_dialog(thread_id: str):
    st.write("This removes this chat history only. Research memory is unchanged.")
    with st.container(horizontal=True):
        if st.button("Cancel", key=f"cancel_delete_{thread_id}"):
            st.rerun()
        if st.button("Delete", type="primary", key=f"confirm_delete_{thread_id}"):
            try:
                delete_thread(thread_id)
            except KeyError:
                st.info("This chat was already removed.")
            except Exception:
                st.error("Could not delete this chat. Please try again.")
                return
            clear_deleted_active_thread(st.session_state, thread_id)
            st.rerun()


@st_dialog("Chat options", width="small")
def thread_actions_dialog(thread: dict) -> None:
    """Keep thread actions in a modal, never in the sidebar's layout flow."""
    thread_id = thread["id"]
    st.markdown("Rename chat")
    new_title = st.text_input("Rename thread", value=thread["title"], key=f"rename_thread_{thread_id}")
    if st.button("Save name", key=f"save_thread_{thread_id}", width="stretch"):
        try:
            rename_thread(thread_id, new_title)
        except ValueError as error:
            st.error(str(error))
        except KeyError:
            st.error("This chat no longer exists.")
        else:
            st.rerun()
    st.download_button(
        "Export Markdown", data=export_chat_markdown(thread["title"], load_messages(thread_id)),
        file_name=safe_export_filename(thread["title"]), mime="text/markdown",
        key=f"export_thread_{thread_id}", icon=":material/download:", on_click="ignore", width="stretch",
    )
    if st.button("Delete chat", key=f"delete_thread_{thread_id}", type="secondary", width="stretch"):
        st.session_state["pending_delete_thread_id"] = thread_id
        st.rerun()


def render_recent_threads() -> None:
    st.caption("RECENTS")
    threads = list_threads()
    if not threads:
        st.caption("No research threads yet.")
        return
    for thread in threads:
        thread_id = thread["id"]
        row_key = f"recent_thread_row_active_{thread_id}" if thread_id == st.session_state.active_thread_id else f"recent_thread_row_{thread_id}"
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", gap="xxsmall", key=row_key):
            if st.button(thread["title"], key=f"thread_title_{thread_id}", type="tertiary", disabled=thread_id == st.session_state.active_thread_id, width="stretch"):
                st.session_state.active_thread_id = thread_id
                st.session_state.messages = load_messages(thread_id)
                st.session_state.thread_id = thread_id
                st.rerun()
            if st.button("⋯", key=f"thread_menu_{thread_id}", type="tertiary", help="Chat options", width="content"):
                thread_actions_dialog(thread)


def main():
    st.set_page_config(page_title="RecallForge", page_icon=":material/auto_awesome:", layout="wide")
    st.session_state.setdefault("active_thread_id", None)
    initialize_database()
    st.markdown(workspace_css(), unsafe_allow_html=True)
    render_sidebar()
    with st.sidebar:
        render_recent_threads()
    pending_delete_thread_id = st.session_state.pop("pending_delete_thread_id", None)
    if pending_delete_thread_id:
        delete_thread_dialog(pending_delete_thread_id)

    st.title("Research")
    st.markdown("<p class='rf-kicker'>RecallForge decides when to reuse stored evidence and when fresh web research is required.</p>", unsafe_allow_html=True)
    st.session_state.setdefault("search_fresh_control_version", 0)
    search_fresh = st.checkbox("Search fresh", key=f"search_fresh_{st.session_state.search_fresh_control_version}", help="Use fresh web research for this question only.")
    prompt = st.chat_input("Ask RecallForge anything...")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))

    for index, message in enumerate(st.session_state.messages):
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                if message["role"] == "assistant": render_response(message, index)
                else: st.markdown(message["content"])

    if not st.session_state.messages:
        st.subheader("Research with persistent memory")
        st.caption("RecallForge reuses prior evidence when it is sufficient and searches the web when information is fresh or missing.")

    if not prompt:
        return

    force_web = consume_search_fresh(st.session_state, search_fresh)
    if st.session_state.active_thread_id is None:
        st.session_state.active_thread_id = create_thread(prompt)
        st.session_state.thread_id = st.session_state.active_thread_id
    user_message = {"role": "user", "content": prompt}
    add_message(st.session_state.active_thread_id, user_message)
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        with st.spinner("Thinking...", show_time=True):
            try:
                query = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
                    current_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question=prompt
                )
                started_at = time.perf_counter()
                with citation_collection() as collector:
                    web_diagnostics, candidates = {}, []
                    memory_retrieval_seconds, synthesis_seconds = None, 0.0
                    if force_web:
                        decision = forced_web_decision()
                        evidence = asyncio.run(_fetch_sites(prompt, force_fresh=True))
                        web_diagnostics.update(parse_web_diagnostics(evidence))
                    else:
                        preliminary = choose_route(prompt)
                        decision = preliminary
                        active_documents = st.session_state.get("active_documents", [])
                        if not preliminary.freshness_sensitive:
                            retrieval_started = time.perf_counter()
                            if ENABLE_PDF_FEATURES and active_documents:
                                active_candidates = asyncio.run(retrieve_active_document_candidates(prompt, active_documents))
                                active_decision = choose_route(prompt, active_candidates)
                                logger.info("Active document retrieval: hashes=%s document_context=%s selected_scope=active candidates=%s route=%s", [item.get("document_hash") for item in active_documents], is_document_context_query(prompt, active_documents), [(getattr(document, "metadata", {}).get("filename"), getattr(document, "metadata", {}).get("page_number"), score) for document, score in active_candidates], active_decision.route)
                                if active_decision.route == "memory": candidates, decision = active_candidates, active_decision
                                else:
                                    candidates = asyncio.run(retrieve_memory_candidates(prompt, active_documents=[]))
                                    decision = choose_route(prompt, candidates)
                            else:
                                candidates = asyncio.run(retrieve_memory_candidates(prompt, active_documents=[]))
                                decision = choose_route(prompt, candidates)
                            memory_retrieval_seconds = time.perf_counter() - retrieval_started
                        if decision.route == "memory":
                            evidence = "EVIDENCE\n\n" + format_evidence(register_evidence([evidence_from_document(document, "memory") for document, _ in candidates]))
                        else:
                            evidence = asyncio.run(_fetch_sites(prompt, force_fresh=decision.freshness_sensitive))
                            web_diagnostics.update(parse_web_diagnostics(evidence))

                    async def synthesize(route: str, route_evidence: str) -> SynthesisResult:
                        nonlocal synthesis_seconds
                        synthesis_started = time.perf_counter()
                        try:
                            response = await synthesis_agent.ainvoke(
                                {"messages": [{"role": "user", "content": query + f"\n\nRoute: {route.upper()}\nReason: {decision.reason}\n\n{route_evidence}"}]},
                                config={"configurable": {"thread_id": st.session_state.thread_id}},
                            )
                        finally:
                            synthesis_seconds += time.perf_counter() - synthesis_started
                        return SynthesisResult.from_answer(response["messages"][-1].text)

                    async def acquire_fresh_web() -> str:
                        fresh_evidence = await _fetch_sites(prompt, force_fresh=True)
                        web_diagnostics.update(parse_web_diagnostics(fresh_evidence))
                        return fresh_evidence

                    outcome = asyncio.run(synthesize_with_single_fallback(
                        initial_route=decision.route, initial_reason=decision.reason, initial_evidence=evidence,
                        synthesize=synthesize, acquire_web=acquire_fresh_web, replace_request_evidence=collector.clear,
                    ))
                total_seconds = time.perf_counter() - started_at
                valid, invalid = validate_citations(outcome.answer, collector.source_map)
                answer = sanitize_answer_markdown(remove_invalid_citations(outcome.answer, invalid))
                warning = "" if not invalid else "\n\nCitation warning: model referenced unknown source ID " + ", ".join(sorted(invalid)) + "."
                debug = {"routing_mode": "FORCE_WEB" if force_web else "AUTO", "initial_route": outcome.diagnostics.initial_route.upper(), "final_route": outcome.diagnostics.final_route.upper(), "fallback_attempted": outcome.diagnostics.fallback_attempted, "total_seconds": total_seconds, "synthesis_seconds": synthesis_seconds, "source_count": len(valid), **web_diagnostics}
                if memory_retrieval_seconds is not None: debug["memory_retrieval_seconds"] = memory_retrieval_seconds
                assistant_message = complete_assistant_message(
                    answer + warning, outcome.diagnostics.final_route.upper(), outcome.reason, f"{round(total_seconds)}s",
                    render_authoritative_sources(collector.source_map, valid), debug,
                )
            except Exception as error:
                assistant_message = complete_assistant_message("I couldn't find enough relevant evidence to answer this confidently.")
                error_fallback(error)
        response_index = append_assistant_message(st.session_state.messages, assistant_message)
        add_message(st.session_state.active_thread_id, assistant_message)
        render_response(assistant_message, response_index)


if __name__ == "__main__":
    main()
