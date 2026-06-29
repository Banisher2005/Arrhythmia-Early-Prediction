"""Small reusable UI primitives for the Streamlit application."""

from __future__ import annotations

import html
from collections.abc import Iterable

import streamlit as st


def inject_clinical_styles() -> None:
    """Inject shared clinical dashboard styling."""

    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1220px;
                padding-top: 1.6rem;
                padding-bottom: 2.4rem;
            }
            h1, h2, h3 {
                letter-spacing: 0;
            }
            div[data-testid="stMetric"] {
                border: 1px solid rgba(100, 116, 139, 0.22);
                border-radius: 8px;
                padding: 1rem;
                background: rgba(255, 255, 255, 0.78);
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
            }
            .clinical-hero {
                border: 1px solid rgba(37, 99, 235, 0.22);
                border-radius: 8px;
                padding: 1.25rem;
                background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(255,255,255,0.68));
                margin-bottom: 1rem;
            }
            .clinical-card {
                border: 1px solid rgba(100, 116, 139, 0.22);
                border-radius: 8px;
                padding: 1rem;
                height: 100%;
                background: rgba(255, 255, 255, 0.78);
            }
            .clinical-card h3 {
                margin-top: 0;
                font-size: 1.05rem;
            }
            .clinical-muted {
                color: #64748b;
            }
            .clinical-eyebrow {
                color: #2563eb;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                border-radius: 999px;
                padding: 0.25rem 0.65rem;
                font-size: 0.78rem;
                font-weight: 700;
                margin-right: 0.4rem;
            }
            .status-ok {
                color: #166534;
                background: #dcfce7;
            }
            .status-warn {
                color: #92400e;
                background: #fef3c7;
            }
            .status-bad {
                color: #991b1b;
                background: #fee2e2;
            }
            @media (prefers-color-scheme: dark) {
                div[data-testid="stMetric"], .clinical-card {
                    background: rgba(15, 23, 42, 0.68);
                    border-color: rgba(148, 163, 184, 0.28);
                }
                .clinical-hero {
                    background: linear-gradient(135deg, rgba(37, 99, 235, 0.22), rgba(15,23,42,0.64));
                    border-color: rgba(96, 165, 250, 0.32);
                }
                .clinical-muted {
                    color: #94a3b8;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label: str, state: str = "ok") -> str:
    """Return a sanitized status badge HTML fragment."""

    css = {
        "ok": "status-ok",
        "warn": "status-warn",
        "bad": "status-bad",
    }.get(state, "status-warn")
    return f'<span class="status-badge {css}">{html.escape(label)}</span>'


def render_hero(eyebrow: str, title: str, body: str, badges: Iterable[str] = ()) -> None:
    """Render a compact clinical dashboard hero."""

    badge_html = " ".join(badges)
    st.markdown(
        f"""
        <div class="clinical-hero">
            <div class="clinical-eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p class="clinical-muted">{html.escape(body)}</p>
            <div>{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_navigation_cards(cards: list[tuple[str, str, str]]) -> None:
    """Render quick navigation cards."""

    columns = st.columns(3)
    for index, (title, body, target) in enumerate(cards):
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="clinical-card">
                    <h3>{html.escape(title)}</h3>
                    <p class="clinical-muted">{html.escape(body)}</p>
                    <p><strong>{html.escape(target)}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

