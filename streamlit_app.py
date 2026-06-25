import streamlit as st
import tabs.black_scholes_tab as bs_tab
import tabs.momentum_analysis_tab as ma_tab
import tabs.portfolio_optimization_tab as po_tab
import tabs.portfolio_statistics_tab as ps_tab

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Quantitative Analysis App',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Declare functions.

# Helper function to load CSS from a file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Call the function to apply the styles
local_css("style.css")

# -----------------------------------------------------------------------------
# Set the title that appears at the top of the page.
st.title("Quantitative Analysis Application")
st.write("`Created by:`")
linkedin_url = "https://www.linkedin.com/in/ansonrui/"
st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Anson (Hao Wen) Rui`</a>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Navigation

pg = st.navigation([
    st.Page(po_tab.render_tab, title="Portfolio Optimization", url_path="portfolio_optimization"),
    st.Page(ps_tab.render_tab, title="Portfolio Statistics", url_path="portfolio_statistics"),
    st.Page(bs_tab.render_tab, title="Black-Scholes Model", url_path="black_scholes"),
    st.Page(ma_tab.render_tab, title="Momentum Analysis", url_path="momentum_analysis"),
])
pg.run()
