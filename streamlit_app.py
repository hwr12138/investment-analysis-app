import streamlit as st
from app.views import black_scholes as bs_page
from app.views import dcf_valuation as dcf_page
from app.views import momentum_analysis as ma_page
from app.views import portfolio_optimization as po_page
from app.views import portfolio_statistics as ps_page

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
local_css("assets/style.css")

# -----------------------------------------------------------------------------
# Set the title that appears at the top of the page.
st.title("Quantitative Analysis Application")
st.write("`Created by:`")
linkedin_url = "https://www.linkedin.com/in/ansonrui/"
st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Anson (Hao Wen) Rui`</a>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Navigation

pg = st.navigation([
    st.Page(dcf_page.render_page, title="DCF Valuation", icon="💰", url_path="dcf_valuation"),
    st.Page(po_page.render_page, title="Portfolio Optimization", icon="📈", url_path="portfolio_optimization"),
    st.Page(ps_page.render_page, title="Portfolio Statistics", icon="📊", url_path="portfolio_statistics"),
    st.Page(bs_page.render_page, title="Black-Scholes Model", icon="📉", url_path="black_scholes"),
    st.Page(ma_page.render_page, title="Momentum Analysis", icon="🚀", url_path="momentum_analysis"),
])
pg.run()
