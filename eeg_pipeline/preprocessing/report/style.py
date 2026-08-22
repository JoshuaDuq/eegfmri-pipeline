"""Colour, colormap, and scaling conventions shared by every report figure.

Every module that appends figures to the subject HTML report renders through these
constants so that one colour means one thing across the whole document.

Conventions
-----------
Categorical colours come from the Okabe-Ito palette, which stays distinguishable
under the common forms of colour vision deficiency.

Hue encodes a measured quantity: a processing stage, a physiological reference, a run.
Decisions the pipeline made — excluded, dropped, retained — are drawn as a neutral
light-to-dark ramp instead, so no reader has to work out whether an orange bar means
"before correction" or "this one was thrown away". A flag that has to sit on top of an
already-coloured series is drawn in :data:`MARK_COLOR`, because any hue chosen for it
would collide with whichever run already owns that hue.

Zero-centred power in decibels is drawn with a diverging, colour-vision-safe
colormap on a symmetric scale, so that the neutral colour always marks "no change
from baseline". Sequential rainbow colormaps such as ``jet`` and ``turbo`` are not
used: their non-monotonic lightness introduces visual boundaries that do not exist
in the data.

Embedding format follows the dominant content. Report figures are viewed in a browser
at a width the author does not control, so line-based figures are embedded as SVG and
keep legible text at any zoom. Figures dominated by a dense image layer stay raster,
because wrapping the same pixels in base64 inside an SVG only inflates the report. Any
dense layer inside a vector figure must still be drawn with ``rasterized=True`` so that
a vector frame carries a raster interior.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from eeg_pipeline.infra.matplotlib import setup_matplotlib
from eeg_pipeline.preprocessing.report.phases import phases_as_json

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

#: Colour for a signal, source, or condition shown on its own.
PRIMARY_COLOR = OKABE_ITO["blue"]
#: Colour for the state before a correction step.
#:
#: Orange rather than vermillion so that "before correction" and "flagged" are never the
#: same ink. They previously shared vermillion, which meant a single figure could use one
#: colour for a pre-ICA trace in one panel and for a flagged run in the next.
BEFORE_COLOR = OKABE_ITO["orange"]
#: Colour for the state after a correction step.
AFTER_COLOR = OKABE_ITO["blue"]
#: Colour for a physiological reference trace (ECG, EOG) shown as context.
REFERENCE_COLOR = OKABE_ITO["reddish_purple"]
#: Colour for a detector flag or threshold crossing drawn as a fill or a line.
#:
#: Use only where no per-run colours share the axis; :data:`MARK_COLOR` covers the rest.
FLAG_COLOR = OKABE_ITO["vermillion"]
#: Colour for a flag mark drawn on top of an already-coloured series.
#:
#: Rings and crosses that annotate run-coloured points cannot carry a hue of their own:
#: any hue collides with whichever run happens to wear it. Black reads as annotation
#: rather than as another series.
MARK_COLOR = OKABE_ITO["black"]
#: Neutral colour for guides, medians, and zero lines.
GUIDE_COLOR = "0.35"

#: Fill for an item the pipeline excluded or dropped.
#:
#: Status is drawn as a light-to-dark neutral ramp rather than a hue, so that the hues in
#: a figure always mean "measured quantity" and never "decision". This also keeps the
#: status strips from out-inking the measurements they annotate.
EXCLUDED_COLOR = "0.25"
#: Fill for an item the pipeline retained, paired with :data:`EXCLUDED_COLOR`.
RETAINED_COLOR = "0.85"

#: Background wash marking a whole panel as excluded, rather than a mark inside it.
#:
#: Distinct from :data:`EXCLUDED_COLOR` because this ink sits *behind* a topography and
#: has to stay light enough for the map on top to keep its own contrast. It is the same
#: neutral ramp, taken from the light end, and it is paired with an
#: :data:`EXCLUDED_COLOR` edge so the panel still reads as marked and not merely tinted.
EXCLUDED_PANEL_FILL = "0.85"

#: Per-run colours, cycled. Shared so that a run keeps one colour across every panel.
RUN_COLORS = (
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
)

#: Diverging colormap for baseline-relative power and other zero-centred decibels.
DIVERGING_POWER_COLORMAP = "RdBu_r"

#: Format for figures built from lines, scatters, and bars.
#:
#: Vector text stays legible at any zoom, and the path data for these figures is small.
REPORT_IMAGE_FORMAT = "svg"

#: Format for raster figures: those dominated by a dense image layer, and every list.
#:
#: A rasterized mesh inside an SVG is the same pixel data carried as base64, so it costs
#: more bytes while only sharpening the axis text. Measured on a three-row component
#: dossier at MNE's embedding resolution: 399 KB as WebP, 500 KB as PNG, 714 KB as SVG.
#: Across the ~310 dossiers in a five-band review that is 121 MB against 216 MB, so
#: these figures stay raster and use the smallest raster MNE supports.
REPORT_RASTER_IMAGE_FORMAT = "webp"


def report_image_format(*, has_dense_image: bool = False, is_figure_list: bool = False) -> str:
    """Return a safe embedding format for one ``Report.add_figure`` call.

    ``is_figure_list`` must be true whenever a list of figures is passed, because MNE
    renders a list through its slider template. That template has no SVG branch and
    always emits ``data:image/{format};base64``, which for SVG produces the invalid MIME
    type ``image/svg`` instead of ``image/svg+xml`` — browsers silently refuse to draw
    it, so every figure in the slider disappears. Only the single-figure template
    special-cases SVG and inlines the markup. Figure lists therefore stay raster
    regardless of their content.
    """
    if is_figure_list or has_dense_image:
        return REPORT_RASTER_IMAGE_FORMAT
    return REPORT_IMAGE_FORMAT


def run_entity(recording_id: object) -> str | None:
    """The run entity of a recording id, or ``None`` where it carries none.

    "Does this recording have a run entity" is a question with a real answer, and callers
    that need it were inferring it from :func:`run_label` returning its input unchanged.
    That made a labelling decision load-bearing: as soon as the label improved, a cohort
    grid read the new label as a run token and headed a column ``run-task-rest_acq-a``.

    Both spellings are read. A full recording id carries ``_run-2``; cohort tables carry a
    bare ``run-2``, and missing that spelling once collapsed every run of a participant
    into a single column.
    """
    text = str(recording_id)
    _, separator, run = text.partition("_run-")
    if separator:
        return run.split("_")[0]
    for part in text.split("_"):
        if part.startswith("run-"):
            return part[len("run-") :]
    return None


def run_label(recording_id: object, *, bare: bool = False) -> str:
    """Return the run-identifying tail of a BIDS recording id.

    A figure that repeats ``sub-0015_task-thermalactive_run-1`` in six panel titles spends
    most of its title bar restating the subject and task named in the report heading, so
    panels are labelled ``run-1``. A table column already headed "run" wants the bare
    ``1``, which is what ``bare`` selects.

    Shared because three modules previously parsed this string three ways and two of them
    disagreed about whether the ``run-`` prefix was part of the label.

    A recording with no ``run-`` entity is an ordinary acquisition, not a broken one: a
    baseline or single-session recording is complete without one, and BIDS only requires
    the entity to tell several apart. Returning the whole id then put
    ``sub-0001_task-baseline`` in every row of every per-run table — the subject and task
    the report is already titled with, identical in all of them and identifying nothing.
    What is left after the subject is what actually distinguishes such a recording, so
    that is the label.

    A value that is already a label is returned unchanged. Cohort tables carry a bare
    ``run-2`` rather than a full path, and rewriting those collapsed six runs onto one
    value, which read as a participant who had recorded a single run.
    """
    text = str(recording_id)
    run = run_entity(text)
    if run is not None:
        return run if bare else f"run-{run}"
    if not text.startswith("sub-"):
        return text
    # A full BIDS recording id carrying no run entity. Drop the subject, which the report
    # is titled with, and the derivative suffix, which names the processing stage rather
    # than the recording.
    _, _, remainder = text.partition("_")
    for suffix in ("_proc-", "_desc-", "_eeg", "_raw"):
        remainder = remainder.partition(suffix)[0]
    return remainder or "recording"


def separated_labels(
    values: Iterable[tuple[float, str]],
    *,
    minimum_gap: float,
) -> list[tuple[float, str]]:
    """Nudge overlapping labels apart along one axis, keeping their order.

    Three panels label individual participants at the right edge of a trace or at the end
    of a paired line, and all three need the same thing: the label identifies a line while
    the marker carries the value, so moving a label a little to keep it readable costs
    nothing a reader relies on, whereas two identifiers printed over each other cost the
    panel its point.

    Order is preserved rather than merely the positions adjusted, because a label that
    overtook its neighbour would name the wrong line -- which is worse than an unreadable
    one, since it is legibly wrong.

    Shared because this was written twice, verbatim, and a third panel that needed it drew
    every label at one coordinate instead.
    """
    placed: list[tuple[float, str]] = []
    for value, label in sorted(values):
        if placed and value - placed[-1][0] < minimum_gap:
            value = placed[-1][0] + minimum_gap
        placed.append((value, label))
    return placed


def draw_figure_footnote(figure, text: str) -> None:
    """Write a caveat under a whole figure, in space the layout engine has reserved.

    ``figure.text`` places an artist in figure coordinates, which ``constrained_layout``
    never consults. Two panels wrote their footnote at y=0.005 and the layout engine then
    put something else there: an outside legend, and the ICA variance
    panel's own axis label. Both printed through the note and neither could be read.

    ``supxlabel`` is the same statement made where the layout engine can see it. It is
    laid out below every subplot and below their axis labels, so the note keeps its
    meaning -- a remark about the figure rather than about one panel -- and the engine
    grows the figure to fit it instead of stacking it on whatever was already there.

    Shared because this was written twice, in the two modules that got it wrong.
    """
    figure.supxlabel(text, fontsize=7, color=GUIDE_COLOR)


#: Marker that makes :func:`apply_report_css` idempotent across repeated opens.
_REPORT_CSS_SENTINEL = "/* eeg-pipeline report css */"

#: Style rules the subject report needs and MNE's template does not provide.
#:
#: Matplotlib writes an absolute ``width`` in points onto every SVG it exports, and MNE
#: inlines that markup verbatim. Raster figures escape the problem because MNE gives them
#: Bootstrap's ``img-fluid``; inline SVGs get no such class and no ``max-width``, so a
#: figure wider than the column overflows it rather than scaling down to fit.
#:
#: The selector stays scoped to ``figure`` on purpose. A bare ``svg`` rule would also
#: match the accordion chevrons, whose size comes from a rule of MNE's own.
#:
#: The table rules style ``.report-table`` alone, which is the class every table this
#: pipeline builds carries. Bootstrap's ``table-striped`` is deliberately not used:
#: zebra striping is a light-to-dark neutral ramp, and this document reserves that ramp
#: for pipeline decisions (:data:`EXCLUDED_COLOR`, :data:`RETAINED_COLOR`). Striping
#: every row would spend the document's decision ink on rows that decide nothing.
#:
#: ``tabular-nums`` matters more than it looks. Arial's default figures are
#: proportional, so a column of run durations does not line up its decimal points and a
#: reader cannot compare magnitudes by scanning down it.
REPORT_CSS = f"""{_REPORT_CSS_SENTINEL}
figure svg {{ max-width: 100%; height: auto; }}
table.report-table {{ border-collapse: collapse; margin: 0.5rem 0 0.75rem; }}
table.report-table th,
table.report-table td {{ padding: 0.25rem 0.75rem 0.25rem 0; border-bottom: 1px solid #e4e4e4; }}
table.report-table thead th {{ border-bottom: 1px solid #b8b8b8; font-weight: 600; }}
table.report-table tbody tr:last-child th,
table.report-table tbody tr:last-child td {{ border-bottom: none; }}
table.report-table th[scope='row'] {{ font-weight: 400; text-align: left; }}
table.report-table tr.key th,
table.report-table tr.key td {{ font-weight: 600; }}
table.report-table tr.sub th[scope='row'] {{ padding-left: 1.5rem; color: #555; }}
table.report-table .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.report-phase-heading {{
  margin: 0.9rem 0 0.15rem;
  padding: 0 0.25rem;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: #6b6b6b;
}}
.report-phase-heading:first-child {{ margin-top: 0; }}
"""


def _without_block(include: str, *, tag: str, sentinel: str) -> str:
    """``include`` with any ``tag`` block carrying ``sentinel`` removed."""
    pattern = re.compile(
        rf"\n?<{tag}[^>]*>\n?(?:(?!</{tag}>).)*?{re.escape(sentinel)}.*?</{tag}>",
        re.DOTALL,
    )
    return pattern.sub("", include)


def apply_report_css(report) -> None:
    """Replace this pipeline's stylesheet in ``report`` with the current one.

    ``Report.add_custom_css`` appends to ``Report.include`` with no deduplication, and a
    subject report is opened and saved once per review stage, so a six-stage run would
    embed the same stylesheet six times.

    Replacing rather than skipping, because skipping on the sentinel also froze the
    styling of every report already on disk: a rule added later never reached one, and
    the phase headings shipped unstyled that way.
    """
    include = getattr(report, "include", "")
    # A real report always carries a string here. Anything else means there is nothing to
    # have found the sentinel in, so treat it as "not yet applied" rather than failing.
    if isinstance(include, str) and _REPORT_CSS_SENTINEL in include:
        report.include = _without_block(include, tag="style", sentinel=_REPORT_CSS_SENTINEL)
    report.add_custom_css(REPORT_CSS)


_REPORT_JS_SENTINEL = "/* eeg-pipeline report js */"

#: Behaviour MNE's template does not provide, applied to the rendered document.
#:
#: Written as script rather than as a build step because MNE renders the contents list
#: from its own template at save time, and the grouping has to describe the document that
#: was actually produced -- which sections a report contains depends on configuration and
#: on what was recorded.
REPORT_JS = f"""{_REPORT_JS_SENTINEL}
(function () {{
  var phases = {phases_as_json()};

  function groupContents() {{
    var nav = document.getElementById('toc-navbar');
    if (!nav || nav.dataset.phasesApplied) return;
    var links = Array.prototype.slice.call(nav.querySelectorAll('a.nav-link'));
    if (!links.length) return;

    phases.forEach(function (phase) {{
      var first = null;
      links.forEach(function (link) {{
        if (first) return;
        var title = link.textContent.trim();
        var belongs = phase.sections.some(function (section) {{
          return title.indexOf(section) === 0;
        }});
        if (belongs) first = link;
      }});
      // A phase with no section in this document contributes no heading.
      if (!first) return;
      var heading = document.createElement('div');
      heading.className = 'report-phase-heading';
      heading.textContent = phase.title;
      first.parentNode.insertBefore(heading, first);
    }});

    nav.dataset.phasesApplied = '1';
  }}

  function collapseAllButFirst() {{
    var container = document.getElementById('container');
    if (!container || container.dataset.collapseApplied) return;
    var items = Array.prototype.slice.call(
      container.querySelectorAll('.accordion-item')
    ).filter(function (item) {{
      // Top-level sections only. A nested item keeps its state, so expanding a section
      // shows its contents rather than a second row of closed accordions.
      return !item.parentNode.closest('.accordion-item');
    }});

    items.forEach(function (item, index) {{
      if (index === 0) return;
      Array.prototype.forEach.call(
        item.querySelectorAll(':scope > .accordion-collapse'),
        function (panel) {{ panel.classList.remove('show'); }}
      );
      Array.prototype.forEach.call(
        item.querySelectorAll(':scope > .accordion-header .accordion-button'),
        function (button) {{
          button.classList.add('collapsed');
          button.setAttribute('aria-expanded', 'false');
        }}
      );
    }});

    container.dataset.collapseApplied = '1';
  }}

  function apply() {{ groupContents(); collapseAllButFirst(); }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', apply);
  }} else {{
    apply();
  }}
}})();
"""


def apply_report_js(report) -> None:
    """Add :data:`REPORT_JS` to ``report`` unless it is already there.

    Replaced like :func:`apply_report_css` and for the same reasons: a subject report is
    opened and saved once per review stage, and a report already carrying an older script
    has to receive the current one.
    """
    include = getattr(report, "include", "")
    if isinstance(include, str) and _REPORT_JS_SENTINEL in include:
        report.include = _without_block(include, tag="script", sentinel=_REPORT_JS_SENTINEL)
    report.add_custom_js(REPORT_JS)


#: Largest number of component ticks drawn before the labels are thinned.
_MAX_COMPONENT_TICKS = 32


def draw_component_status_strip(
    axis,
    *,
    excluded,
    component_count: int,
    legend: bool = True,
) -> None:
    """Draw the excluded/retained decision for every component as a strip under an axis.

    Shared because two panels ask a reader to relate a per-component measurement to the
    decision taken about that component, and only one of them used to make the decision
    legible. The correlation panel drew this strip; the variance panel encoded the same
    fact as the marker's own lightness, which at the marker size that fits 62 components
    is a distinction between two pale greys.

    A strip rather than shading behind the measurement: full-height shading covers a
    third of the axis and out-inks the quantity the panel is about.

    The neutral ramp is deliberate and follows this module's convention — "excluded" is
    a decision, not a measurement, so it never takes a hue. It is also not
    :data:`FLAG_COLOR`: "this component was excluded" and "this component crossed a
    detector's threshold" are different statements, and most exclusions in a typical
    decomposition were never flagged by the detector whose panel they sit under.
    """
    from matplotlib.patches import Patch

    components = np.arange(component_count)
    excluded_set = {int(component) for component in excluded}
    axis.bar(
        components,
        1.0,
        width=1.0,
        color=[EXCLUDED_COLOR if c in excluded_set else RETAINED_COLOR for c in components],
    )
    # Component index is categorical, so the ticks are the indices themselves rather than
    # whatever round numbers a continuous locator picks: "2.5" names no component and does
    # not line up with the cells of this strip.
    step = max(1, int(np.ceil(component_count / _MAX_COMPONENT_TICKS)))
    axis.set(
        xlim=(-0.7, component_count - 0.3),
        ylim=(0, 1),
        yticks=[],
        xticks=components[::step],
        xlabel="ICA component",
    )
    axis.tick_params(axis="x", labelsize=7)
    axis.set_ylabel(
        f"excluded\n({len(excluded_set)}/{component_count})",
        fontsize=6,
        rotation=0,
        ha="right",
        va="center",
    )
    if legend:
        # Two greys need a key: the count alone never said which shade carried it.
        axis.legend(
            handles=[
                Patch(facecolor=EXCLUDED_COLOR, label="Excluded"),
                Patch(facecolor=RETAINED_COLOR, label="Retained"),
            ],
            frameon=False,
            fontsize=6.5,
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.55),
            handlelength=1.2,
            handleheight=0.9,
        )
    axis.spines[["top", "right", "left"]].set_visible(False)


#: Percentile of the absolute values that defines a robust symmetric colour limit.
COLOR_LIMIT_PERCENTILE = 98.0


def apply_report_style() -> None:
    """Configure the non-interactive backend and shared render defaults."""
    setup_matplotlib()


def robust_symmetric_limit(
    *values: np.ndarray,
    percentile: float = COLOR_LIMIT_PERCENTILE,
) -> float:
    """Return a symmetric colour limit that a few extreme samples cannot dominate.

    A shared colour scale lets a reviewer compare components against each other, but
    taking the scale from the single largest absolute value lets one extreme
    component flatten every other component to the neutral colour. The limit is
    therefore taken from a high percentile of the pooled absolute values, and the
    caller is expected to state the resulting limit on the figure so that clipped
    samples are declared rather than hidden.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"Colour limit percentile must lie in (0, 100], got {percentile!r}.")
    pooled = np.concatenate([np.abs(np.asarray(value, dtype=float)).ravel() for value in values])
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        raise ValueError("Colour limits require at least one finite value.")
    limit = float(np.percentile(finite, percentile))
    if limit <= 0.0:
        raise ValueError("Colour limits require a non-zero spread of values.")
    return limit


def power_colorbar_label(limit: float, *, quantity: str = "Baseline-relative power") -> str:
    """Return a colourbar label that declares the symmetric clipping limit."""
    return f"{quantity} (dB, clipped at ±{limit:.1f})"


__all__ = [
    "AFTER_COLOR",
    "BEFORE_COLOR",
    "COLOR_LIMIT_PERCENTILE",
    "DIVERGING_POWER_COLORMAP",
    "EXCLUDED_COLOR",
    "EXCLUDED_PANEL_FILL",
    "FLAG_COLOR",
    "GUIDE_COLOR",
    "MARK_COLOR",
    "OKABE_ITO",
    "RETAINED_COLOR",
    "PRIMARY_COLOR",
    "REFERENCE_COLOR",
    "RUN_COLORS",
    "REPORT_CSS",
    "REPORT_IMAGE_FORMAT",
    "REPORT_RASTER_IMAGE_FORMAT",
    "apply_report_css",
    "apply_report_style",
    "draw_component_status_strip",
    "draw_figure_footnote",
    "report_image_format",
    "power_colorbar_label",
    "robust_symmetric_limit",
    "run_entity",
    "run_label",
    "separated_labels",
]
