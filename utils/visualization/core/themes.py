"""Visual theme configuration for PFNCPS visualizations.

Provides consistent styling across all plots with support for
different themes: default, paper (publication-ready), and dark.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Theme:
    """Visual theme configuration.

    Defines colors, fonts, and styling for all visualizations.

    Attributes:
        name: Theme identifier
        colors: Color palette for different elements
        font_family: Font family for labels and titles
        font_sizes: Font sizes for different text elements
        line_widths: Line widths for different plot elements
        alpha_values: Transparency values for different elements
        markers: Marker styles
        figure_defaults: Default figure settings
    """
    name: str

    # Color palette
    colors: Dict[str, str] = field(default_factory=dict)

    # Font settings
    font_family: str = "sans-serif"
    font_sizes: Dict[str, int] = field(default_factory=dict)

    # Line and marker settings
    line_widths: Dict[str, float] = field(default_factory=dict)
    alpha_values: Dict[str, float] = field(default_factory=dict)
    markers: Dict[str, str] = field(default_factory=dict)

    # Figure defaults
    figure_defaults: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Set default values if not provided."""
        # Default colors if not set
        if not self.colors:
            self.colors = self._default_colors()

        # Default font sizes if not set
        if not self.font_sizes:
            self.font_sizes = {
                "title": 14,
                "axis_label": 12,
                "tick_label": 10,
                "legend": 10,
                "annotation": 9,
            }

        # Default line widths if not set
        if not self.line_widths:
            self.line_widths = {
                "main": 2.0,
                "secondary": 1.5,
                "thin": 1.0,
                "thick": 2.5,
                "particle": 0.5,
                "mean": 2.0,
                "threshold": 1.5,
            }

        # Default alpha values if not set
        if not self.alpha_values:
            self.alpha_values = {
                "particle_line": 0.3,
                "confidence_band": 0.3,
                "background": 0.1,
                "highlight": 0.8,
                "grid": 0.3,
            }

        # Default markers if not set
        if not self.markers:
            self.markers = {
                "point": "o",
                "resampling": "v",
                "event": "x",
                "particle": ".",
            }

        # Default figure settings if not set
        if not self.figure_defaults:
            self.figure_defaults = {
                "figsize": (10, 6),
                "dpi": 100,
                "facecolor": "white",
            }

    def _default_colors(self) -> Dict[str, str]:
        """Return default color palette."""
        return {
            # Primary colors
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "tertiary": "#2ca02c",

            # Semantic colors
            "mean": "#1f77b4",
            "particles": "#7f7f7f",
            "weights": "#d62728",
            "ess": "#9467bd",
            "threshold": "#e377c2",
            "resampling": "#bcbd22",

            # Health indicators
            "healthy": "#2ca02c",
            "warning": "#ff7f0e",
            "error": "#d62728",

            # Gradients for heatmaps (start, end)
            "heatmap_low": "#440154",
            "heatmap_high": "#fde725",

            # Background/grid
            "background": "#ffffff",
            "grid": "#e0e0e0",
            "text": "#333333",

            # Particle colors (cycling)
            "particle_0": "#1f77b4",
            "particle_1": "#ff7f0e",
            "particle_2": "#2ca02c",
            "particle_3": "#d62728",
            "particle_4": "#9467bd",
            "particle_5": "#8c564b",
            "particle_6": "#e377c2",
            "particle_7": "#7f7f7f",
        }

    def get_particle_colors(self, n: int) -> List[str]:
        """Get list of colors for n particles.

        Args:
            n: Number of particles

        Returns:
            List of color strings
        """
        base_colors = [
            self.colors.get(f"particle_{i}", self.colors["particles"])
            for i in range(8)
        ]
        # Cycle through colors if needed
        return [base_colors[i % len(base_colors)] for i in range(n)]

    def apply_to_axes(self, ax) -> None:
        """Apply theme styling to matplotlib axes.

        Args:
            ax: Matplotlib axes object
        """
        # Set background color
        ax.set_facecolor(self.colors["background"])

        # Set grid
        ax.grid(True, alpha=self.alpha_values["grid"], color=self.colors["grid"])

        # Set tick parameters
        ax.tick_params(
            labelsize=self.font_sizes["tick_label"],
            colors=self.colors["text"],
        )

        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color(self.colors["grid"])

    def apply_to_figure(self, fig) -> None:
        """Apply theme styling to matplotlib figure.

        Args:
            fig: Matplotlib figure object
        """
        fig.patch.set_facecolor(self.figure_defaults["facecolor"])


# Pre-defined themes

DEFAULT_THEME = Theme(
    name="default",
    colors={
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "tertiary": "#2ca02c",
        "mean": "#1f77b4",
        "particles": "#7f7f7f",
        "weights": "#d62728",
        "ess": "#9467bd",
        "threshold": "#e377c2",
        "resampling": "#bcbd22",
        "healthy": "#2ca02c",
        "warning": "#ff7f0e",
        "error": "#d62728",
        "heatmap_low": "#440154",
        "heatmap_high": "#fde725",
        "background": "#ffffff",
        "grid": "#e0e0e0",
        "text": "#333333",
        "particle_0": "#1f77b4",
        "particle_1": "#ff7f0e",
        "particle_2": "#2ca02c",
        "particle_3": "#d62728",
        "particle_4": "#9467bd",
        "particle_5": "#8c564b",
        "particle_6": "#e377c2",
        "particle_7": "#7f7f7f",
    },
)

PAPER_THEME = Theme(
    name="paper",
    font_family="serif",
    colors={
        "primary": "#000000",
        "secondary": "#555555",
        "tertiary": "#888888",
        "mean": "#000000",
        "particles": "#888888",
        "weights": "#333333",
        "ess": "#000000",
        "threshold": "#666666",
        "resampling": "#444444",
        "healthy": "#000000",
        "warning": "#555555",
        "error": "#333333",
        "heatmap_low": "#ffffff",
        "heatmap_high": "#000000",
        "background": "#ffffff",
        "grid": "#cccccc",
        "text": "#000000",
        "particle_0": "#000000",
        "particle_1": "#333333",
        "particle_2": "#555555",
        "particle_3": "#777777",
        "particle_4": "#999999",
        "particle_5": "#aaaaaa",
        "particle_6": "#bbbbbb",
        "particle_7": "#cccccc",
    },
    font_sizes={
        "title": 12,
        "axis_label": 11,
        "tick_label": 10,
        "legend": 9,
        "annotation": 8,
    },
    line_widths={
        "main": 1.5,
        "secondary": 1.0,
        "thin": 0.75,
        "thick": 2.0,
        "particle": 0.3,
        "mean": 1.5,
        "threshold": 1.0,
    },
    alpha_values={
        "particle_line": 0.2,
        "confidence_band": 0.2,
        "background": 0.05,
        "highlight": 0.9,
        "grid": 0.2,
    },
    figure_defaults={
        "figsize": (6, 4),
        "dpi": 300,
        "facecolor": "white",
    },
)

DARK_THEME = Theme(
    name="dark",
    colors={
        "primary": "#58a6ff",
        "secondary": "#f78166",
        "tertiary": "#56d364",
        "mean": "#58a6ff",
        "particles": "#8b949e",
        "weights": "#f78166",
        "ess": "#a371f7",
        "threshold": "#db61a2",
        "resampling": "#e3b341",
        "healthy": "#56d364",
        "warning": "#e3b341",
        "error": "#f78166",
        "heatmap_low": "#0d1117",
        "heatmap_high": "#58a6ff",
        "background": "#0d1117",
        "grid": "#30363d",
        "text": "#c9d1d9",
        "particle_0": "#58a6ff",
        "particle_1": "#f78166",
        "particle_2": "#56d364",
        "particle_3": "#a371f7",
        "particle_4": "#db61a2",
        "particle_5": "#e3b341",
        "particle_6": "#79c0ff",
        "particle_7": "#8b949e",
    },
    figure_defaults={
        "figsize": (10, 6),
        "dpi": 100,
        "facecolor": "#0d1117",
    },
)

# Theme registry
AVAILABLE_THEMES = {
    "default": DEFAULT_THEME,
    "paper": PAPER_THEME,
    "dark": DARK_THEME,
}


def get_theme(name: str) -> Theme:
    """Get theme by name.

    Args:
        name: Theme name ("default", "paper", "dark")

    Returns:
        Theme configuration

    Raises:
        ValueError: If theme name is not recognized
    """
    if name not in AVAILABLE_THEMES:
        raise ValueError(
            f"Unknown theme '{name}'. Available themes: {list(AVAILABLE_THEMES.keys())}"
        )
    return AVAILABLE_THEMES[name]


def register_theme(name: str, theme: Theme) -> None:
    """Register a custom theme.

    Args:
        name: Theme name
        theme: Theme configuration
    """
    AVAILABLE_THEMES[name] = theme
