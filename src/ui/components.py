"""Workspace-only UI helpers; no routing or retrieval decisions live here."""
import json
import streamlit as st
from ui.debug_panel import build_debug_rows, debug_markdown

def copy_payload(answer: str) -> str:
    return answer.strip()

def build_copy_payload(message: dict) -> str:
    """Return only the assistant answer body, excluding UI/provenance metadata."""
    return copy_payload(str(message.get("content", "")))

def source_action_label(is_open: bool) -> str:
    return "Hide sources" if is_open else "Open sources"

def append_assistant_message(messages: list[dict], message: dict) -> int:
    """Append once and return the stable index used by response action keys."""
    messages.append(message)
    return len(messages) - 1

def clear_deleted_active_thread(state: dict, deleted_thread_id: str) -> bool:
    """Clear only the visible conversation when its UUID was deleted."""
    if state.get("active_thread_id") != deleted_thread_id:
        return False
    state["active_thread_id"] = None
    state["thread_id"] = None
    state["messages"] = []
    return True

def complete_assistant_message(content: str, route: str = "", reason: str = "", elapsed: str = "", sources: str = "", debug: dict | None = None) -> dict:
    """Build complete presentation state before append or canonical rendering."""
    message = {"role": "assistant", "content": content, "route": route, "reason": reason, "elapsed": elapsed, "sources": sources}
    if debug:
        message["debug"] = dict(debug)
    return message

def render_copy_control(answer: str, index: int) -> None:
    """Use a click-bound browser Clipboard API action with honest async feedback."""
    payload = json.dumps(copy_payload(answer), ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    button_id = f"rf-copy-{index}"
    st.html(f'''<button type="button" aria-label="Copy answer" id="{button_id}">Copy</button>
<script>
(() => {{
  const button = document.getElementById({json.dumps(button_id)});
  const payload = {payload};
  if (!button) return;
  button.addEventListener("click", async () => {{
    const original = "Copy";
    try {{
      if (!navigator.clipboard || !window.isSecureContext) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(payload);
      button.textContent = "Copied";
      setTimeout(() => {{ button.textContent = original; }}, 1500);
    }} catch (_error) {{
      button.textContent = "Copy failed";
      setTimeout(() => {{ button.textContent = original; }}, 1800);
    }}
  }});
}})();
</script>''', unsafe_allow_javascript=True)
    return None

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### RecallForge")
        st.caption("Persistent research memory")
        if st.button("New research", icon=":material/add:", width="stretch"):
            st.session_state.messages = []
            st.session_state.thread_id = None
            st.session_state.active_thread_id = None
            st.rerun()
        st.space("small")
        st.caption("RecallForge · Akshay Kumar")
    return None

def render_response(message: dict, index: int) -> None:
    route = message.get("route", "")
    if route:
        accent = "rf-memory" if route == "MEMORY" else "rf-web"
        st.markdown(f'<span class="rf-status {accent}">{route} · {message.get("elapsed", "")}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="rf-reason">{message.get("reason", "")}</div>', unsafe_allow_html=True)
    st.markdown(message["content"])
    sources = message.get("sources", "")
    open_key = f"sources_open_{index}"
    st.session_state.setdefault(open_key, False)
    left, right = st.columns((1, 5))
    with left:
        render_copy_control(message["content"], index)
    if sources and right.button(source_action_label(st.session_state[open_key]), key=f"sources_{index}", icon=":material/link:"):
        st.session_state[open_key] = not st.session_state[open_key]
        st.rerun()
    if sources and st.session_state[open_key]:
        with st.container(border=True):
            st.markdown(sources)
    rows = build_debug_rows(message)
    if rows:
        with st.expander("Debug details", icon=":material/monitoring:", type="compact"):
            st.markdown(debug_markdown(rows))
    return None
