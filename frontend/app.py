"""
Streamlit demo UI for the Agent Distillation Compiler.
Sends problems to the FastAPI backend and displays generated code,
routing decision, and latency. Includes a teacher-mode toggle.
"""

import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Agent Distillation Compiler", page_icon="🤖", layout="wide")

st.title("🤖 Agent Distillation Compiler")
st.caption("Hybrid student/teacher code generation with complexity routing")

col1, col2 = st.columns([2, 1])

with col1:
    problem = st.text_area(
        "Describe your coding problem:",
        placeholder="e.g. Write a function that finds the nth Fibonacci number.",
        height=150,
    )

with col2:
    st.markdown("### Settings")
    max_tokens = st.slider("Max tokens", min_value=64, max_value=1024, value=512, step=64)
    teacher_mode = st.toggle("Force teacher mode", value=False,
                              help="Always use the DPO-aligned teacher model, ignoring the router.")
    st.info("**Student:** Llama3.1-8B (fast)\n\n**Teacher:** DPO-aligned Qwen2.5-7B (higher quality)")

if st.button("Generate", type="primary", disabled=not problem.strip()):
    with st.spinner("Generating..."):
        try:
            payload = {"problem": problem, "max_new_tokens": max_tokens}
            # teacher-mode toggle overrides routing by appending a complexity hint
            if teacher_mode:
                payload["problem"] = problem + "\n\n[complex]"
            resp = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            st.success(f"Routed to: **{data['route'].upper()}** model | Latency: {data['latency_seconds']:.1f}s")
            st.code(data["code"], language="python")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Make sure the FastAPI server is running on port 8000.")
        except requests.exceptions.Timeout:
            st.error("Request timed out (>120s). The model may still be loading -- try again in a moment.")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("Agent Distillation Compiler -- Team: Yeshita, Faiza, Sakshi")