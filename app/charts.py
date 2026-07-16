"""Reusable Plotly chart helpers shared across pages."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_allocation(allocation_dict, title="Optimal Allocation"):
    """Creates a donut chart for portfolio weights, filtering out zero allocations."""
    # Filter out allocations less than 0.1% to keep the chart clean
    filtered = {k: v for k, v in allocation_dict.items() if v >= 0.001}
    if not filtered:
        return go.Figure()

    df = pd.DataFrame(list(filtered.items()), columns=['Asset', 'Weight'])
    fig = px.pie(df, values='Weight', names='Asset', title=title, hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template='plotly_dark', margin=dict(t=50, b=20, l=20, r=20))
    return fig


def plot_yearly_returns(yearly_returns_series, title="Yearly Returns"):
    """Creates a bar chart with green/red coloring for positive/negative years."""
    if yearly_returns_series.empty:
        return go.Figure()

    df = yearly_returns_series.reset_index()
    df.columns = ['Year', 'Return']
    df['Return %'] = df['Return'] * 100
    df['Color'] = df['Return %'].apply(lambda x: '#00cc96' if x >= 0 else '#ef553b')  # Plotly default green/red

    fig = go.Figure(data=[
        go.Bar(
            x=df['Year'].astype(str),
            y=df['Return %'],
            marker_color=df['Color'],
            text=df['Return %'].round(2).astype(str) + '%',
            textposition='auto'
        )
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Return (%)",
        template='plotly_dark',
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig
