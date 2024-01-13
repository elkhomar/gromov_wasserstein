import plotly.express as px
import plotly.graph_objects as go


def show_coupling(M):
    fig = px.imshow(M, labels={"x":"Target Samples", "y":"Source Samples"}, title="OT Matrix")
    fig.show()

def create_transport_plan_figure(source_samples, target_samples, G):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=source_samples[:, 0], y=source_samples[:, 1], mode='markers', name='Source Samples'))
    fig.add_trace(go.Scatter(x=target_samples[:, 0], y=target_samples[:, 1], mode='markers', name='Target Samples'))
    for i in range(source_samples.shape[0]):
        for j in range(target_samples.shape[0]):
            if G[i, j] > 1e-4:
                fig.add_trace(go.Scatter(x=[source_samples[i, 0], target_samples[j, 0]],
                                         y=[source_samples[i, 1], target_samples[j, 1]],
                                         mode='lines',
                                         line=dict(width=1, color=f'rgba(255, 0, 0, {G[i, j]})'),
                                         hoverinfo='none',
                                         showlegend=False))
    return fig