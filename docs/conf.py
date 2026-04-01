from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "EEG–fMRI Analysis Pipeline"
author = "JoshuaDuq"
release = "1.0.0"
copyright = "2026, JoshuaDuq"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

# ---------------------------------------------------------------------------
# Source / build
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "migration/**",
    "plans/**",
    "superpowers/**",
    "architecture/**",
    "index.md",
    "eeg/**",
    "fmri/**",
    "index_old.md",
]

# ---------------------------------------------------------------------------
# Autodoc / autosummary
# ---------------------------------------------------------------------------
autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
add_module_names = False
autodoc_mock_imports = [
    "pandas",
    "numpy",
    "scipy",
    "mne",
    "mne_bids",
    "mne_bids_pipeline",
    "mne_icalabel",
    "mne_connectivity",
    "nilearn",
    "nibabel",
    "sklearn",
    "torch",
    "joblib",
    "specparam",
    "antropy",
    "networkx",
    "statsmodels",
    "imblearn",
    "shap",
    "pyprep",
    "matplotlib",
    "seaborn",
    "pyarrow",
]

# ---------------------------------------------------------------------------
# Napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# MyST (Markdown support)
# ---------------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "mne": ("https://mne.tools/stable", None),
    "nilearn": ("https://nilearn.github.io/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output — furo theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "EEG–fMRI Pipeline"
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap",
    "custom.css",
]
pygments_style = "friendly"
pygments_dark_style = "monokai"
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
html_favicon = "_static/favicon.svg"

html_theme_options = {
    "dark_css_variables": {
        "color-background-primary": "#141720",
        "color-background-secondary": "#1b1f2e",
        "color-background-hover": "#222638",
        "color-background-border": "#252a3a",
        "color-foreground-primary": "#dce2ef",
        "color-foreground-secondary": "#8e97b0",
        "color-foreground-muted": "#525a72",
        "color-foreground-border": "#2c3148",
        "color-brand-primary": "#8ab4d4",
        "color-brand-content": "#6d9dbf",
        "color-highlight-on-target": "rgba(138, 180, 212, 0.07)",
        "color-admonition-background": "#1b1f2e",
        "font-stack": "'Inter', 'Segoe UI', system-ui, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', ui-monospace, monospace",
    },
    "light_css_variables": {
        "color-background-primary": "#f5f6fa",
        "color-background-secondary": "#eceef5",
        "color-brand-primary": "#2a6496",
        "color-brand-content": "#1f5280",
        "font-stack": "'Inter', 'Segoe UI', system-ui, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', ui-monospace, monospace",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/JoshuaDuq/eegfmri-pipeline",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/JoshuaDuq/eegfmri-pipeline",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8'
                "c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49"
                "-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01"
                "-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1"
                ".07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08"
                "-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53"
                "-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 "
                "3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 "
                "0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z\""
                "></path></svg>"
            ),
            "class": "",
        }
    ],
}
