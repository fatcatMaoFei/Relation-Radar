from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

API_BASE_ENV_VAR = "RELATION_RADAR_API_BASE"
DEFAULT_API_BASE = "http://127.0.0.1:8000"


def get_api_base() -> str:
    """
    Resolve the base URL for the backend Web API.

    The URL is taken from the RELATION_RADAR_API_BASE environment variable
    if present; otherwise it defaults to http://127.0.0.1:8000.
    """
    base = os.getenv(API_BASE_ENV_VAR, DEFAULT_API_BASE).strip()
    if base.endswith("/"):
        base = base[:-1]
    return base


def fetch_persons(tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch all persons from the Web API, optionally filtered by tag.
    """
    params: Dict[str, Any] = {}
    if tag:
        params["tag"] = tag.strip()

    resp = requests.get(f"{get_api_base()}/persons", params=params, timeout=5)
    resp.raise_for_status()
    return resp.json()


def fetch_events(
    person_id: int,
    start: Optional[date],
    end: Optional[date],
    tag: Optional[str],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Fetch events for a given person from the Web API.
    """
    params: Dict[str, Any] = {}

    if start is not None:
        params["start"] = datetime.combine(start, datetime.min.time()).isoformat()
    if end is not None:
        params["end"] = datetime.combine(end, datetime.max.time()).isoformat()
    if tag:
        params["tag"] = tag.strip()
    if limit is not None:
        params["limit"] = int(limit)

    resp = requests.get(
        f"{get_api_base()}/persons/{person_id}/events",
        params=params,
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()


def create_event(person_id: int, text: str) -> Dict[str, Any]:
    """
    Create a new text event for a person via the Web API.
    """
    payload = {"text": text}
    resp = requests.post(
        f"{get_api_base()}/persons/{person_id}/events",
        json=payload,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def ask_question_api(person_id: int, question: str, top_k: int) -> Dict[str, Any]:
    """
    Ask a question about a person via the Web API.
    """
    payload = {"question": question, "top_k": top_k}
    resp = requests.post(
        f"{get_api_base()}/persons/{person_id}/ask",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def render_sidebar(persons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Render the sidebar with person selection and basic info.
    """
    st.sidebar.header("Friends")

    if not persons:
        st.sidebar.info("No persons found. Use the CLI to add a friend first.")
        return None

    options = [
        f"{p.get('name', 'Unknown')} (#{p.get('id', '?')})"
        for p in persons
    ]

    selected_index = st.sidebar.selectbox(
        "Select a friend",
        options=list(range(len(persons))),
        format_func=lambda i: options[i],
    )

    person = persons[selected_index]

    tags = person.get("tags") or []
    if tags:
        st.sidebar.caption("Tags: " + ", ".join(tags))

    notes = person.get("notes")
    if notes:
        st.sidebar.write(notes)

    return person


def main() -> None:
    """
    Streamlit entrypoint for the "one person, one notebook" web UI.

    This UI is intentionally thin: it only talks to the FastAPI backend via
    HTTP and does not access the database or vector store directly. The same
    API can later be reused by iOS / Android clients.
    """
    st.set_page_config(page_title="Relation Radar", layout="wide")

    api_base = get_api_base()

    st.title("Relation Radar · Notebook")
    st.caption(f"Backend API: {api_base}")

    with st.sidebar:
        person_tag = st.text_input("Filter friends by person tag", value="")

        try:
            persons = fetch_persons(tag=person_tag or None)
        except requests.RequestException as exc:
            st.error(f"Could not reach backend API at {api_base}: {exc}")
            st.stop()

    selected_person = render_sidebar(persons)
    if not selected_person:
        return

    col_timeline, col_filters = st.columns([3, 1])

    with col_filters:
        st.subheader("Filters")

        use_date_range = st.checkbox("Filter by date range", value=False)
        start_date: Optional[date] = None
        end_date: Optional[date] = None
        if use_date_range:
            default_start = date.today() - timedelta(days=180)
            default_end = date.today()
            start_date, end_date = st.date_input(
                "Date range",
                value=(default_start, default_end),
            )

        event_tag = st.text_input("Event tag filter", value="")
        limit = st.number_input(
            "Max events",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
        )

        rag_top_k = st.slider(
            "RAG context size (top_k)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help=(
                "How many related pieces of history to use when asking questions. "
                "Stored in session for future Q&A views."
            ),
        )
        st.session_state["rag_top_k"] = rag_top_k

        st.markdown("---")
        st.caption(
            "Tip: this page only shows stored events. "
            "Question-answering will reuse the same person, filters and top_k."
        )

    with col_timeline:
        st.subheader(
            f"Notebook for {selected_person.get('name', 'Unknown')} "
            f"(ID {selected_person.get('id')})"
        )

        person_id = int(selected_person["id"])

        # --- New event input ---
        with st.expander("Add a new note for this friend", expanded=False):
            new_text = st.text_area(
                "New note (plain text)",
                value="",
                height=80,
                key="new_event_text",
            )
            if st.button("Save note", key="save_new_event"):
                if not new_text.strip():
                    st.warning("Text cannot be empty.")
                else:
                    try:
                        created = create_event(person_id=person_id, text=new_text)
                    except requests.RequestException as exc:
                        st.error(f"Failed to create event: {exc}")
                    else:
                        st.success(f"Saved event (id={created.get('id')}).")
                        try:
                            # Streamlit >= 1.30
                            st.rerun()
                        except AttributeError:
                            # Older versions used experimental_rerun
                            getattr(st, "experimental_rerun")()

        # --- Timeline ---
        try:
            events = fetch_events(
                person_id=person_id,
                start=start_date,
                end=end_date,
                tag=event_tag or None,
                limit=int(limit) if limit else None,
            )
        except requests.RequestException as exc:
            st.error(f"Failed to load events from API: {exc}")
            return

        if not events:
            st.info("No events found for this friend with current filters.")
        else:
            for event in events:
                occurred_at = event.get("occurred_at") or "Unknown time"
                tags = event.get("tags") or []
                header = f"**{occurred_at}**"
                if tags:
                    header += " · " + ", ".join(tags)
                st.markdown(header)

                summary = event.get("summary")
                if summary:
                    st.write(summary)

                raw_text = event.get("raw_text")
                if raw_text and raw_text != summary:
                    with st.expander("Raw note", expanded=False):
                        st.write(raw_text)

                emotion = event.get("emotion")
                preferences = event.get("preferences") or []
                taboos = event.get("taboos") or []

                meta_parts = []
                if emotion:
                    meta_parts.append(f"Emotion: {emotion}")
                if preferences:
                    meta_parts.append("Likes: " + "; ".join(preferences))
                if taboos:
                    meta_parts.append("Avoid: " + "; ".join(taboos))

                if meta_parts:
                    st.caption(" · ".join(meta_parts))

                st.markdown("---")

        # --- Q&A section ---
        st.subheader("Ask a question about this friend")
        question = st.text_area(
            "Your question",
            value="",
            height=80,
            key="qa_question",
        )
        if st.button("Ask", key="qa_ask_button"):
            if not question.strip():
                st.warning("Question cannot be empty.")
            else:
                try:
                    payload = ask_question_api(
                        person_id=person_id,
                        question=question,
                        top_k=int(st.session_state.get("rag_top_k", 10)),
                    )
                except requests.RequestException as exc:
                    st.error(f"Failed to ask question: {exc}")
                else:
                    st.session_state["qa_last_answer"] = payload

        last_answer = st.session_state.get("qa_last_answer")
        if last_answer:
            st.markdown("**Answer (for this friend):**")
            st.write(last_answer.get("answer", ""))
            st.caption(
                "Based on your recorded notes for this friend. "
                "Please double‑check before acting on suggestions.",
            )


if __name__ == "__main__":
    main()

