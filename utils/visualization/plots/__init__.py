"""Visualization plotting functions for PFNCPS.

This module provides comprehensive visualization functions organized by:
- Core plots (C1-C10): Universal particle filter visualizations
- State-level plots (S1-S4): State-level PF specific
- Param-level plots (P1-P5): Parameter-level PF specific
- Dual plots (D1-D4): Dual PF specific
- SDE plots (E1-E6): SDE-based PF specific
- LTC plots (L1-L7): LTC architecture specific
- CfC plots (F1-F5): CfC architecture specific
- Wired plots (W1-W5): Wired/NCP architecture specific
"""

# Core plots (C1-C10)
from .core_plots import (
    plot_ess_timeline,
    plot_weight_distribution,
    plot_weight_entropy,
    plot_particle_trajectories,
    plot_particle_diversity,
    plot_resampling_events,
    plot_observation_likelihoods,
    plot_numerical_health,
    plot_weighted_output,
    animate_particles_2d,
)

# State-level plots (S1-S4)
from .state_plots import (
    plot_noise_injection_magnitude,
    plot_state_dependent_noise,
    plot_particle_pairwise_distances,
    plot_particle_cloud_evolution,
    animate_particle_cloud_3d,
)

# Param-level plots (P1-P5)
from .param_plots import (
    plot_param_posterior_marginals,
    plot_param_uncertainty_timeline,
    plot_param_correlation_matrix,
    plot_tracked_vs_base_params,
    plot_param_evolution_trajectory,
)

# Dual plots (D1-D4)
from .dual_plots import (
    plot_joint_state_param_scatter,
    plot_rao_blackwell_variance,
    plot_state_param_correlation,
    plot_marginal_posteriors,
    plot_joint_evolution,
)

# SDE plots (E1-E6)
from .sde_plots import (
    plot_diffusion_magnitude,
    plot_drift_diffusion_ratio,
    plot_unfold_convergence,
    plot_brownian_increments,
    plot_state_clamping_events,
    plot_euler_maruyama_stability,
    plot_sde_summary_dashboard,
)

# LTC plots (L1-L7)
from .ltc_plots import (
    plot_voltage_traces,
    plot_time_constants,
    plot_synapse_activations,
    plot_leak_vs_synaptic,
    plot_ode_unfold_dynamics,
    plot_reversal_potential_flow,
    plot_sparsity_mask_utilization,
    plot_ltc_summary_dashboard,
)

# CfC plots (F1-F5)
from .cfc_plots import (
    plot_interpolation_weights,
    plot_ff1_ff2_contributions,
    plot_time_constants_learned,
    plot_backbone_activations,
    plot_mode_comparison,
    plot_effective_time_constant_cfc,
    plot_cfc_summary_dashboard,
)

# Wired/NCP plots (W1-W5)
from .wired_plots import (
    plot_layer_activations,
    plot_ncp_connectivity_graph,
    plot_information_flow,
    plot_layer_wise_ess,
    plot_sensory_to_motor_path,
    plot_wired_summary_dashboard,
)

# Diagnostic plots
from .diagnostic_plots import create_dashboard

__all__ = [
    # Core plots (C1-C10)
    "plot_ess_timeline",
    "plot_weight_distribution",
    "plot_weight_entropy",
    "plot_particle_trajectories",
    "plot_particle_diversity",
    "plot_resampling_events",
    "plot_observation_likelihoods",
    "plot_numerical_health",
    "plot_weighted_output",
    "animate_particles_2d",
    # State-level plots (S1-S4)
    "plot_noise_injection_magnitude",
    "plot_state_dependent_noise",
    "plot_particle_pairwise_distances",
    "plot_particle_cloud_evolution",
    "animate_particle_cloud_3d",
    # Param-level plots (P1-P5)
    "plot_param_posterior_marginals",
    "plot_param_uncertainty_timeline",
    "plot_param_correlation_matrix",
    "plot_tracked_vs_base_params",
    "plot_param_evolution_trajectory",
    # Dual plots (D1-D4)
    "plot_joint_state_param_scatter",
    "plot_rao_blackwell_variance",
    "plot_state_param_correlation",
    "plot_marginal_posteriors",
    "plot_joint_evolution",
    # SDE plots (E1-E6)
    "plot_diffusion_magnitude",
    "plot_drift_diffusion_ratio",
    "plot_unfold_convergence",
    "plot_brownian_increments",
    "plot_state_clamping_events",
    "plot_euler_maruyama_stability",
    "plot_sde_summary_dashboard",
    # LTC plots (L1-L7)
    "plot_voltage_traces",
    "plot_time_constants",
    "plot_synapse_activations",
    "plot_leak_vs_synaptic",
    "plot_ode_unfold_dynamics",
    "plot_reversal_potential_flow",
    "plot_sparsity_mask_utilization",
    "plot_ltc_summary_dashboard",
    # CfC plots (F1-F5)
    "plot_interpolation_weights",
    "plot_ff1_ff2_contributions",
    "plot_time_constants_learned",
    "plot_backbone_activations",
    "plot_mode_comparison",
    "plot_effective_time_constant_cfc",
    "plot_cfc_summary_dashboard",
    # Wired/NCP plots (W1-W5)
    "plot_layer_activations",
    "plot_ncp_connectivity_graph",
    "plot_information_flow",
    "plot_layer_wise_ess",
    "plot_sensory_to_motor_path",
    "plot_wired_summary_dashboard",
    # Diagnostic
    "create_dashboard",
]
