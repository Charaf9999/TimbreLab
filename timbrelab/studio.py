"""The studio's artwork, decorative motion and shared visual headings."""

from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTWORK = ROOT / "assets" / "studio-atmosphere.png"


@st.cache_data(show_spinner=False)
def artwork_data_uri(modified_ns):
    """Cache the local PNG as a data URI; its modification time invalidates the cache."""
    return "data:image/png;base64," + b64encode(ARTWORK.read_bytes()).decode("ascii")


def render_studio_header():
    """Draw the masthead and illustrated listening-room hero with ambient motion."""
    css = (ROOT / "assets" / "studio.css").read_text(encoding="utf-8")
    artwork = artwork_data_uri(ARTWORK.stat().st_mtime_ns)
    st.html(f'<style>:root {{ --studio-art: url("{artwork}"); }}\n{css}</style>')
    with st.container(key="studio-masthead"):
        st.html("""
            <div class="studio-brand">
                <span class="studio-mark" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>
                <div><span class="studio-wordmark">TimbreLab<span class="brand-dot">.</span></span>
                <span class="studio-descriptor">MUSIC FEATURE STUDIO</span></div>
            </div>
        """)
    st.html("""
        <div class="studio-atmosphere" aria-hidden="true">
            <div class="atmosphere-image"></div>
            <div class="ambient-halo halo-one"></div><div class="ambient-halo halo-two"></div>
        </div>
        <section class="studio-hero" aria-label="Welcome to TimbreLab">
            <div class="hero-art" aria-hidden="true"><div class="hero-image"></div></div>
            <div class="hero-content">
                <p class="studio-eyebrow"><span></span> MADE FOR CURIOUS EARS</p>
                <h1>Find the patterns<br>in your <em>sound.</em></h1>
                <p class="hero-description">A space to explore music, uncover its character,<br class="desktop-break">
                    and turn audio features into discoveries.</p>
                <div class="hero-tags"><span>Classify genres</span><span>Predict tempo</span><span>Discover patterns</span></div>
            </div>
            <div class="hero-caption" aria-hidden="true">
                <span class="sound-wave"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span>
                <span>A different way to listen.</span>
            </div>
        </section>
    """)


def section_heading(step, title, description=None):
    """Render an escaped, consistently styled heading for a workspace step."""
    detail = f'<p class="chapter-description">{escape(description)}</p>' if description else ""
    st.html(f'<div class="chapter-heading"><p class="chapter-kicker">{escape(step)}</p>'
            f'<h2>{escape(title)}</h2>{detail}</div>')


def show_footer():
    """Close the workspace with a quiet brand signature."""
    st.html('<footer class="studio-footer"><span>TimbreLab / MUSIC FEATURE STUDIO</span>'
            '<span>For the art of understanding sound.</span></footer>')
