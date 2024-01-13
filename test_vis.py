import numpy as np
import ot
import plotly.graph_objects as go
import plotly.express as px
import os
import json

def optimal_transport_visualization(source_samples, target_samples, source_weights, target_weights, output_folder):
    """
    Computes the optimal transport between two 2D empirical distributions and generates Plotly visualizations.

    Parameters:
    source_samples (np.ndarray): 2D array of shape (n_samples, 2) representing the source distribution.
    target_samples (np.ndarray): 2D array of shape (n_samples, 2) representing the target distribution.
    source_weights (np.ndarray): 1D array of weights for the source distribution.
    target_weights (np.ndarray): 1D array of weights for the target distribution.
    output_folder (str): Path to the folder where the visualizations will be saved.

    Returns:
    str: Path to the folder containing the saved visualizations.
    """

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Compute distance matrix
    M = ot.dist(source_samples, target_samples)

    # Compute EMD (Exact Optimal Transport)
    G0 = ot.emd(source_weights, target_weights, M)

    # Compute Sinkhorn and Empirical Sinkhorn with error tracking
    lambd = 1e-1
    Gs, log_sinkhorn = ot.sinkhorn(source_weights, target_weights, M, lambd, log=True)
    Ges, log_emp_sinkhorn = ot.bregman.empirical_sinkhorn(source_samples, target_samples, lambd, log=True)

    # Plot and save the cost matrix
    fig = px.imshow(M, labels=dict(x="Target Samples", y="Source Samples"), title="Cost Matrix")
    save_plotly_plot(fig, "cost_matrix.html", output_folder)

    # Plot and save the OT matrices
    for G, name in zip([G0, Gs, Ges], ["G0", "Gs", "Ges"]):
        fig = px.imshow(G, labels=dict(x="Target Samples", y="Source Samples"), title=f"OT Matrix {name}")
        save_plotly_plot(fig, f"ot_matrix_{name}.html", output_folder)

    # Plot and save the transport plans with samples
    for G, name in zip([G0, Gs, Ges], ["G0", "Gs", "Ges"]):
        fig = create_transport_plan_figure(source_samples, target_samples, G)
        fig.update_layout(title=f"Transport Plan {name} with Samples")
        save_plotly_plot(fig, f"transport_plan_{name}.html", output_folder)

    # Plot and save the error curves
    for log, name in zip([log_sinkhorn, log_emp_sinkhorn], ["Sinkhorn", "Empirical Sinkhorn"]):
        fig = create_error_curve_figure(log['err'])
        fig.update_layout(title=f"Error Curve {name}")
        save_plotly_plot(fig, f"error_curve_{name}.html", output_folder)

    # Create and save the log file
    create_log_file({'source_mean': mu_s, 'source_cov': cov_s, 'target_mean': mu_t, 'target_cov': cov_t,
                     'lambd': lambd, 'iterations_sinkhorn': len(log_sinkhorn['err']),
                     'iterations_emp_sinkhorn': len(log_emp_sinkhorn['err']),
                     'dimension': source_samples.shape[1], 'num_samples': source_samples.shape[0]}, output_folder)

    return output_folder

# Helper Functions:

def save_plotly_plot(fig, filename, output_folder):
    plot_path = os.path.join(output_folder, filename)
    fig.write_html(plot_path)

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

def create_error_curve_figure(errors):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(errors))), y=errors, mode='lines+markers', name='Error'))
    fig.update_layout(xaxis_title="Iterations", yaxis_title="Error")
    return fig

def create_log_file(parameters, output_folder):
    with open(os.path.join(output_folder, 'experiment_log.json'), 'w') as file:
        json.dump(parameters, file, indent=4)

# Example use of the function (commented out as we do not have specific input data)
n = 50  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])

source_samples = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
target_samples = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

source_weights = np.ones(source_samples.shape[0]) / source_samples.shape[0]
target_weights = np.ones(target_samples.shape[0]) / target_samples.shape[0]
output_folder = "optimal_transport_visualizations"
optimal_transport_visualization(source_samples, target_samples, source_weights, target_weights, output_folder)

