"""Build every table and figure in the report from the saved result files (no model, no GPU).

    python paper/build_assets.py        writes paper/tables/*.tex and paper/figs/*.pdf

Numbers are read from the markdown and JSON files the analysis scripts wrote under runs/, never typed in here,
so the report cannot drift from the results. Rerun after rerunning an analysis.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS, TABLES, FIGS = ROOT / "runs", Path(__file__).parent / "tables", Path(__file__).parent / "figs"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # fixed order; checked for colour-blind separation
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
CELL = re.compile(r"(-?[\d.]+)%? \[(-?[\d.]+)%?, (-?[\d.]+)%?\]")


def md_table(path, heading):
    """Rows (lists of cell strings) of the first table after the line starting with `heading`."""
    lines = (RUNS / path).read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(heading))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("|---"))
    rows = []
    for line in lines[j + 1:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def ci(cell):
    """'0.537 [0.483, 0.592] (n=..)' -> (0.537, 0.483, 0.592)"""
    return tuple(map(float, CELL.search(cell).groups()))


def pct(cell, digits=1):
    v, lo, hi = ci(cell)
    return f"{100 * v:.{digits}f} {{\\scriptsize[{100 * lo:.{digits}f}, {100 * hi:.{digits}f}]}}"


def num(cell, digits=3):
    v, lo, hi = ci(cell)
    return f"{v:.{digits}f} {{\\scriptsize[{lo:.{digits}f}, {hi:.{digits}f}]}}"


def n_of(cell):
    m = re.search(r"n=(\d+)", cell)
    return f"{int(m.group(1)):,}".replace(",", "{,}") if m else ""


def wrap(h, width=14):
    """Break a long column header over several lines so the table fits the page."""
    lines, cur = [], ""
    for w in h.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return h if len(lines) == 1 else "\\shortstack[l]{" + "\\\\".join(lines) + "}"


def write(name, header, rows, align):
    header = [wrap(h) for h in header]
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    (TABLES / f"{name}.tex").write_text(
        f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


def style(ax, ylabel):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ chapter 4: fading
def fading():
    fade = md_table("fade2/section.md", "### Steps 2 and 3")
    write("fade", ["Tokens after entity", "Concept-vector readout", "Trained probe"],
          [[r[0], pct(r[1]), pct(r[2])] for r in fade], "cll")

    A = {r[0]: r for r in md_table("fade2/extra-clean.md", "### A.")}
    B = {r[0]: r for r in md_table("fade2/extra-clean.md", "### B.")}
    C = {r[0]: r for r in md_table("fade2/extra-clean.md", "### C.")}
    F = {r[0]: r for r in md_table("fade2/extra-clean.md", "### F.")}
    ds = ["0", "1", "4", "16"]
    write("names", ["Tokens after entity", "Probe reads it", "Names the true entity", "At least one wrong name",
                    "Only wrong names", "Wrong names per explanation"],
          [[d, pct(C[d][2]), pct(A[d][1]), pct(B[d][1]), pct(A[d][2]), B[d][4]] for d in ds], "cllllc")
    write("fair", ["Tokens after entity", "Names any same-type entity", "Verbalizer precision", "Probe, equally selective"],
          [[d, C[d][1].replace("%", "\\%"), pct(C[d][4]), pct(C[d][3])] for d in ds], "ccll")
    write("position", ["Tokens after entity", "True name, context", "True name, continuation", "Wrong name, context",
                       "Wrong name, continuation"],
          [[d, pct(F[d][1]), pct(F[d][2]), pct(F[d][3]), pct(F[d][4])] for d in ds], "cllll")

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = range(len(ds))
    series = [("Probe reads the entity", [ci(C[d][2]) for d in ds], BLUE, "o", "-"),
              ("Explanation names it", [ci(A[d][1]) for d in ds], ORANGE, "s", "-"),
              ("Explanation has a wrong name", [ci(B[d][1]) for d in ds], AQUA, "^", "--")]
    for label, vals, color, marker, ls in series:
        v = [100 * a for a, _, _ in vals]
        err = [[100 * (a - lo) for a, lo, _ in vals], [100 * (hi - a) for a, _, hi in vals]]
        ax.errorbar(x, v, yerr=err, color=color, marker=marker, markersize=6, linewidth=2, linestyle=ls, capsize=2.5,
                    elinewidth=1, label=label)
        ax.annotate(f"{v[-1]:.0f}%", (x[-1], v[-1]), xytext=(9, -3), textcoords="offset points", fontsize=9, color=INK)
    ax.axhline(10, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate("probe chance", (0.02, 10), xytext=(0, 3), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xticks(list(x), ds)
    ax.set_xlabel("Tokens after the entity", color=MUTED, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.2, 3.45)
    style(ax, "Share (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
    save(fig, "fade")


# ------------------------------------------------------------------ chapter 5: incentives
def incentives():
    T = md_table("fade2/task1.md", "| tokens after entity | n |")
    rows = [[r[0].replace("0 and 1 pooled", "0 and 1"), f"{int(r[1]):,}".replace(",", "{,}"), num(r[2]), num(r[3]), num(r[4]),
             pct(r[5]), pct(r[6])] for r in T]
    write("edit", ["Tokens after entity", "$n$", "Original", "Wrong names removed", "True name removed",
                   "Original beats ``wrong removed''", "Original beats ``true removed''"], rows, "crlllll")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    w = 0.34
    for k, (label, col, color) in enumerate((("Wrong names removed", 7, BLUE), ("True name removed", 8, ORANGE))):
        vals = [ci(r[col]) for r in T[:2]]
        xs = [i + (k - 0.5) * (w + 0.03) for i in range(2)]
        ax.bar(xs, [v for v, _, _ in vals], width=w, color=color, label=label,
               yerr=[[v - lo for v, lo, _ in vals], [hi - v for v, _, hi in vals]], error_kw=dict(elinewidth=1, capsize=2.5, ecolor=INK))
        for xx, (v, _, hi) in zip(xs, vals):
            ax.annotate(f"{v:.3f}", (xx, hi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1], ["At the entity's token", "One token later"])
    ax.set_ylim(0, 0.25)
    style(ax, "Cosine lost by the edit")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    save(fig, "edit")


# ------------------------------------------------------------------ chapter 6: agreement
def agreement():
    D = {r[0]: r for r in md_table("fade2/extra-clean.md", "### D.")}
    order = ["0", "1", "4", "16", "1, 4 and 16 pooled"]
    cell = lambda c: f"{pct(c)} ({n_of(c)})"
    write("consensus", ["Tokens after entity", "1 of 4", "2 of 4", "3 of 4", "4 of 4"],
          [[d.replace(" pooled", ""), *[cell(D[d][k]) for k in (1, 2, 3, 4)]] for d in order], "lllll")

    E = md_table("fade2/extra-clean.md", "### E.")
    pooled = [r for r in E if r[0].startswith("1, 4 and 16")]
    def frow(r):
        first = r[1] == "no filter"
        return [("None" if first else "All 4" if r[1] == "4" else f"At least {r[1]} of 4"), ("0" if first else pct(r[2])), ("100" if first else pct(r[3])),
                pct(r[4]), f"{int(r[5]):,}".replace(",", "{,}")]
    write("filter", ["Keep a name if given by", "Wrong names removed", "True names kept", "Precision", "Names left"],
          [frow(r) for r in pooled], "lllll")
    write("filter-full", ["Tokens after entity", "Keep a name if given by", "Wrong names removed", "True names kept", "Precision",
                          "Names left"], [[r[0].replace(" pooled", "")] + frow(r) for r in E], "llllll")

    V = md_table("fade2/section.md", "### Step 4: can a verifier")
    auc = {(r[0], r[1]): r[6] for r in V}
    write("verifiers", ["Tokens after entity", "Concept vectors", "Probe directions", "Agreement count"],
          [[d, num(auc[(d, "readout A")], 2), num(auc[(d, "trained probe")], 2), num(D[d][5], 2)] for d in ["0", "1", "4", "16"]], "clll")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    vals = [ci(D["1, 4 and 16 pooled"][k]) for k in (1, 2, 3, 4)]
    v = [100 * a for a, _, _ in vals]
    ax.bar(range(4), v, width=0.55, color=BLUE,
           yerr=[[100 * (a - lo) for a, lo, _ in vals], [100 * (hi - a) for a, _, hi in vals]],
           error_kw=dict(elinewidth=1, capsize=2.5, ecolor=INK))
    for i, (a, _, hi) in enumerate(vals):
        ax.annotate(f"{100 * a:.0f}%", (i, 100 * hi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(4), ["1 of 4", "2 of 4", "3 of 4", "4 of 4"])
    ax.set_xlabel("Samples that give the name", color=MUTED, fontsize=9)
    ax.set_ylim(0, 80)
    style(ax, "Names that are true (%)")
    save(fig, "consensus")


if __name__ == "__main__":
    TABLES.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    for build in (fading, incentives, agreement):
        build()
    print("tables:", sorted(p.name for p in TABLES.glob("*.tex")))
    print("figures:", sorted(p.name for p in FIGS.glob("*.pdf")))
