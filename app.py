import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh, rice, norm

def simulate_propagation(channel_type, distance_m, freq_mhz, tx_power_dbm, noise_power_dbm, rician_k, num_samples):
    # 1. Path Loss Calculation (Free-space path loss formula)
    # FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
    c = 3e8
    f_hz = freq_mhz * 1e6
    fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(f_hz) + 20 * np.log10(4 * np.pi / c)
    rx_power_mean_dbm = tx_power_dbm - fspl_db
    mean_linear_power = 10 ** ((rx_power_mean_dbm - 30) / 10)  # in Watts
    mean_amplitude = np.sqrt(mean_linear_power)

    # 2. Channel Envelope Modeling
    samples = int(num_samples)
    if channel_type == "Free-Space":
        # Deterministic with minor AWGN perturbation
        sigma_awgn = mean_amplitude * 0.05
        envelope = mean_amplitude + np.random.normal(0, sigma_awgn, samples)
        envelope = np.clip(envelope, 0, None)
        analytical_label = "AWGN around Mean"

    elif channel_type == "Rayleigh (NLOS)":
        # Rayleigh scale parameter sigma: E[R^2] = 2*sigma^2 = mean_linear_power
        scale = np.sqrt(mean_linear_power / 2.0)
        envelope = np.random.rayleigh(scale=scale, size=samples)
        analytical_label = "Rayleigh Theoretical PDF"

    elif channel_type == "Rician (LOS)":
        # K-factor ratio between LOS power and scatter power
        k_linear = 10 ** (rician_k / 10.0)
        sigma = np.sqrt(mean_linear_power / (2 * (k_linear + 1)))
        nu = np.sqrt(k_linear * 2 * (sigma ** 2))
        # Rice distribution in scipy uses b = nu / sigma
        b = nu / sigma
        envelope = rice.rvs(b, scale=sigma, size=samples)
        analytical_label = f"Rician Theoretical PDF (K={rician_k} dB)"

    # 3. Statistical Analysis
    emp_mean = np.mean(envelope)
    emp_var = np.var(envelope)
    rx_signal_power_watts = np.mean(envelope ** 2)
    rx_signal_power_dbm = 10 * np.log10(rx_signal_power_watts) + 30
    snr_db = rx_signal_power_dbm - noise_power_dbm

    # 4. Generate Visualizations (PDF & CDF)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x_axis = np.linspace(0, max(envelope) * 1.15, 300)

    # Left: PDF Plot
    axes[0].hist(envelope, bins=50, density=True, alpha=0.6, color='steelblue', label='Empirical Histogram')
    if channel_type == "Rayleigh (NLOS)":
        axes[0].plot(x_axis, rayleigh.pdf(x_axis, scale=scale), 'r-', lw=2, label=analytical_label)
    elif channel_type == "Rician (LOS)":
        axes[0].plot(x_axis, rice.pdf(x_axis, b, scale=sigma), 'r-', lw=2, label=analytical_label)
    axes[0].set_title(f"Probability Density Function (PDF) - {channel_type}")
    axes[0].set_xlabel("Signal Amplitude (V)")
    axes[0].set_ylabel("Probability Density")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Right: CDF Plot
    sorted_data = np.sort(envelope)
    cdf_empirical = np.arange(1, samples + 1) / samples
    axes[1].plot(sorted_data, cdf_empirical, color='darkgreen', lw=2, label='Empirical CDF')
    if channel_type == "Rayleigh (NLOS)":
        axes[1].plot(x_axis, rayleigh.cdf(x_axis, scale=scale), 'r--', lw=2, label='Analytical CDF')
    elif channel_type == "Rician (LOS)":
        axes[1].plot(x_axis, rice.cdf(x_axis, b, scale=sigma), 'r--', lw=2, label='Analytical CDF')
    axes[1].set_title(f"Cumulative Distribution Function (CDF) - {channel_type}")
    axes[1].set_xlabel("Signal Amplitude (V)")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()

    # 5. Explanatory Channel Behavior Summary
    channel_effects = {
        "Free-Space": "• Direct Line-of-Sight (LOS) with no obstructions.\n• Signal degrades only due to spherical divergence over distance (FSPL).\n• No destructive multi-path interference; signal amplitude remains constant aside from background thermal noise.",
        "Rayleigh (NLOS)": "• Non-Line-of-Sight environment where direct path is fully blocked (urban canyons, dense indoor partitions).\n• Signals arrive via random reflections, scattering, and diffraction.\n• Causes severe deep fades where received amplitude drops close to zero with high probability.",
        "Rician (LOS)": f"• Composite environment containing one strong direct/dominant path (LOS) alongside diffuse scattered paths.\n• High K-factor ({rician_k} dB) reduces fading depth and converges toward Gaussian/AWGN behavior.\n• Low K-factor approaches Rayleigh fading conditions."
    }

    metrics_text = (
        f"**Path Loss (FSPL):** {fspl_db:.2f} dB\n\n"
        f"**Received Mean Amplitude:** {emp_mean:.6e} V\n\n"
        f"**Amplitude Variance:** {emp_var:.6e} V²\n\n"
        f"**Average Received Power:** {rx_signal_power_dbm:.2f} dBm\n\n"
        f"**Calculated SNR:** {snr_db:.2f} dB"
    )

    return fig, metrics_text, channel_effects[channel_type]

# Gradio Interface Setup
with gr.Blocks(title="Wireless Signal Propagation & Statistical Analysis") as demo:
    gr.Markdown("# Statistical Analysis of Wireless Signal Propagation")
    gr.Markdown("Analyze deterministic loss and fading distributions (Free-Space, Rayleigh, and Rician) across tunable RF parameters.")

    with gr.Row():
        with gr.Column(scale=1):
            channel_type = gr.Dropdown(
                choices=["Free-Space", "Rayleigh (NLOS)", "Rician (LOS)"],
                value="Rayleigh (NLOS)",
                label="Channel Propagation Model"
            )
            dist_slider = gr.Slider(minimum=10, maximum=5000, value=500, step=10, label="Distance (m)")
            freq_slider = gr.Slider(minimum=700, maximum=6000, value=2400, step=50, label="Carrier Frequency (MHz)")
            tx_pwr_slider = gr.Slider(minimum=-10, maximum=40, value=20, step=1, label="Transmitter Power (dBm)")
            noise_slider = gr.Slider(minimum=-120, maximum=-60, value=-95, step=1, label="Noise Floor (dBm)")
            rician_k_slider = gr.Slider(minimum=0, maximum=25, value=6, step=0.5, label="Rician K-factor (dB) [Rician Only]")
            samples_slider = gr.Slider(minimum=1000, maximum=50000, value=10000, step=1000, label="Sample Size")
            run_btn = gr.Button("Simulate & Analyze", variant="primary")

        with gr.Column(scale=2):
            output_plot = gr.Plot(label="Statistical Distributions (PDF & CDF)")
            with gr.Row():
                output_metrics = gr.Markdown(label="Calculated Metrics")
                output_effects = gr.Markdown(label="Channel Effects & Phenomenon")

    run_btn.click(
        fn=simulate_propagation,
        inputs=[channel_type, dist_slider, freq_slider, tx_pwr_slider, noise_slider, rician_k_slider, samples_slider],
        outputs=[output_plot, output_metrics, output_effects]
    )

# Launch in Colab with share=True to generate a temporary public link
import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
