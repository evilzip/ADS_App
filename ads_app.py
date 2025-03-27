import streamlit as st


# --- PAGE SETUP ---
info_page = st.Page(
    page="views/info_page.py",
    title="Info Page",
    default=True
)

sandbox = st.Page(
    page="views/sandbox.py",
    title="Result here"
)

upload_page = st.Page(
    page="views/upload_page.py",
    title="Upload Data here"
)

# --- NAVIGATION SETUP ---

pg = st.navigation(pages=[info_page, upload_page, sandbox  ])

# # --- RUN NAVIGATION ---
pg.run()
