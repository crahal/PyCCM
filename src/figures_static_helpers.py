import os
import re
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
import glob
import unicodedata
import geopandas as gpd
import mapclassify as mc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyBboxPatch
from shapely.geometry import Polygon, MultiPolygon


def plot_choropleth(root_shp_dir, fig_path, proj, scenario, base_year_req,
                    target_year_req, death_choices_trip):

    def deaccent(s):
        if s is None: return None
        return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

    def norm_name(s: str) -> str:
        if s is None: return None
        t = deaccent(s).upper()
        t = (t.replace("&"," Y ").replace("/", " ").replace(".", " ").replace(",", " ")
               .replace("’","'").replace("`","'"))
        t = re.sub(r"\s+", " ", t).strip()
        repl = {
            "C MARCA":"CUNDINAMARCA","C/MARCA":"CUNDINAMARCA","CUNDINAM ARCA":"CUNDINAMARCA",
            "N SANTANDER":"N DE SANTANDER","NORTE SANTANDER":"N DE SANTANDER","NORTE DE SANTANDER":"N DE SANTANDER",
            "BOGOTA DC":"BOGOTA D C","BOGOTA D.C":"BOGOTA D C","SANTAFE DE BOGOTA DC":"BOGOTA D C",
            "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA":"SAN ANDRES",
            "SAN ANDRES Y PROVIDENCIA":"SAN ANDRES",
            "VALLE DEL CAUCA":"VALLE",
        }
        return repl.get(t, t)

    def choose_name_col(gdf):
        pref = ["NMG","DPTO_NOMBRE","NOMBRE_DEP","NOMBRE_DPT","NOMBRE","NAME_1","NOM_DEP","DEPTO","DEPARTAMEN"]
        for c in pref:
            if c in gdf.columns and gdf[c].dtype == object and gdf[c].nunique(dropna=True) >= 20:
                return c
        obj_cols = [c for c in gdf.columns if gdf[c].dtype == object]
        if not obj_cols:
            raise KeyError("No string-like name column in this layer.")
        return max(obj_cols, key=lambda c: gdf[c].nunique(dropna=True))

    def find_department_layer(root_dir):
        shps = glob.glob(os.path.join(root_dir, "**", "*.shp"), recursive=True)
        if not shps:
            raise FileNotFoundError(f"No .shp found under {root_dir}")
        best = None; best_score = -1
        for path in shps:
            try:
                g = gpd.read_file(path)
                if not (20 <= len(g) <= 5000):
                    continue
                c = choose_name_col(g)
                n = g[c].nunique(dropna=True)
                score = -abs(n - 33) if 20 <= n <= 60 else -(abs(n - 33) + 100)
                if score > best_score:
                    best = (path, c)
                    best_score = score
            except Exception:
                continue
        if best is None:
            raise RuntimeError("Could not auto-detect a departmental layer (wanted ~33 units).")
        path, col = best
        g = gpd.read_file(path)[[col, "geometry"]].rename(columns={col:"_raw_name"})
        print(f"[shp] Using: {path}\n      name column: {col}  (features: {len(g)})")
        g["_name_key"] = g["_raw_name"].map(norm_name)
        g = g.dissolve(by="_name_key", as_index=False)
        return g

    def safe_pretty_breaks(values, k=5):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return [-10, 0, 10, 20, 30][:k]
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if abs(vmax - vmin) < 1e-9:
            c = vmin
            return [c-10, c-5, c, c+5, c+10][:k]
        try:
            br = mc.PrettyBreaks(vals, k=k).bins
            b = [int(np.floor(br[0]))] + [int(round(x)) for x in br[1:-1]] + [int(np.ceil(br[-1]))]
            b = np.unique(b).tolist()
            if len(b) >= k: return b[:k]
        except Exception:
            pass
        qs = np.percentile(vals, np.linspace(0, 100, k+1)[1:])
        b = np.unique([int(round(x)) for x in qs]).tolist()
        while len(b) < k: b.append(b[-1] + 5)
        return b[:k]

    def compute_pct_change(proj_all, death_choice, base_year, target_year):
        d = proj_all[(proj_all["death_choice"] == death_choice)].copy()
        tot = d.groupby(["__key","year"], as_index=False)["population"].sum()
        piv = tot.pivot(index="__key", columns="year", values="population").fillna(0)
        need_cols = [base_year, target_year]
        if any(c not in piv.columns for c in need_cols):
            missing = [c for c in need_cols if c not in piv.columns]
            raise ValueError(f"Missing years for {death_choice}: {missing}")
        df = (
            piv[[base_year, target_year]]
            .rename(columns={base_year:"P_base", target_year:"P_target"})
            .reset_index()
        )
        df["pct_change"] = 100.0 * (df["P_target"] - df["P_base"]) / df["P_base"]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    # ---------- inset helpers ----------
    def to_metric_crs(gdf):
        """Reproject to a metric CRS (estimated UTM if available; fallback to EPSG:3857)."""
        try:
            utm = gdf.estimate_utm_crs()
            if utm is not None:
                return gdf.to_crs(utm)
        except Exception:
            pass
        return gdf.to_crs(3857)

    def largest_component(geom):
        """Return the polygon of maximal area from a (Multi)Polygon or GeometryCollection."""
        if geom is None or geom.is_empty:
            return None
        if isinstance(geom, Polygon):
            return geom
        if isinstance(geom, MultiPolygon):
            return max(geom.geoms, key=lambda p: p.area)
        polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
        return max(polys, key=lambda p: p.area) if polys else None

    def find_non_mainland_keys(gdf, name_col="_name_key", area_share_thresh=0.20):
        """
        Identify departments whose overlap with the largest connected component ("mainland")
        is less than area_share_thresh. Works without hard-coding names.
        """
        gm = to_metric_crs(gdf[[name_col, "geometry"]].copy())
        whole = gm.unary_union
        main = largest_component(whole)
        if main is None or main.is_empty:
            # conservative fallback: only target well-known island dept if present
            return set([k for k in gdf[name_col].unique() if "SAN ANDRES" in k])
        off = []
        for _, r in gm.iterrows():
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            a = geom.area
            share = (geom.intersection(main).area / a) if a > 0 else 1.0
            if share < area_share_thresh:
                off.append(r[name_col])
        return set(off)

    def nice_inset_ax(ax, width_frac=0.36, height_frac=0.36, loc="upper right", borderpad=0.6):
        """Create a clean, rounded inset axes in the specified corner."""
        iax = inset_axes(
            ax,
            width=f"{int(width_frac*100)}%",
            height=f"{int(height_frac*100)}%",
            loc=loc,
            borderpad=borderpad,
        )
        iax.set_axis_off()
        # Subtle rounded white backdrop
        x0, x1 = iax.get_xlim()
        y0, y1 = iax.get_ylim()
        pad = 0.02 * max(x1 - x0, y1 - y0)
        panel = FancyBboxPatch(
            (x0 - pad, y0 - pad),
            (x1 - x0) + 2*pad,
            (y1 - y0) + 2*pad,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            fc="white", ec="#bbbbbb", lw=0.8, alpha=0.95, zorder=-1,
            transform=iax.transData,
        )
        iax.add_patch(panel)
        return iax

    def set_axes_to_bounds(ax, bounds, pad_frac=0.08):
        """Set x/y limits to bounds with a fractional padding for aesthetics."""
        xmin, ymin, xmax, ymax = bounds
        dx, dy = xmax - xmin, ymax - ymin
        ax.set_xlim(xmin - pad_frac*dx, xmax + pad_frac*dx)
        ax.set_ylim(ymin - pad_frac*dy, ymax + pad_frac*dy)

    # -------------------- load shapefile --------------------
    gdf = find_department_layer(root_shp_dir)
    need = ["DPTO_NOMBRE","year","Sex","EDAD","population","scenario","death_choice"]
    miss = [c for c in need if c not in proj.columns]
    if miss:
        raise KeyError(f"Missing columns in projections: {miss}")

    proj = proj[(proj["Sex"]=="T") & (proj["scenario"]==scenario)].copy()
    proj = proj[proj["DPTO_NOMBRE"].str.upper() != "TOTAL_NACIONAL"]
    proj["__key"] = proj["DPTO_NOMBRE"].map(norm_name)

    # years (ensure common base/target across all choices)
    years = np.sort(proj["year"].unique())
    base_year   = base_year_req   if base_year_req   in years else years.min()
    target_year = target_year_req if target_year_req in years else years.max()
    print(f"[years] Using base={base_year}, target={target_year} (available {years[0]}..{years[-1]})")

    # -------------------- merge data for each death_choice --------------------
    merged_maps = {}
    all_vals = []

    for dc in death_choices_trip:
        df_chg = compute_pct_change(proj, dc, base_year, target_year)
        g = gdf.merge(df_chg[["__key","pct_change"]], left_on="_name_key", right_on="__key", how="left")
        merged_maps[dc] = g
        all_vals.append(g["pct_change"].to_numpy())

    # common breaks across all three panels
    bins = safe_pretty_breaks(np.concatenate(all_vals), k=5)

    # --- detect non-mainland departments once (now that gdf exists) ---
    non_mainland_keys = find_non_mainland_keys(gdf, name_col="_name_key")

    # -------------------- plot 1×3 --------------------
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))

    titles = {
        "censo_2018": "censo_2018",
        "EEVV":       "EEVV",
        "midpoint":   "midpoint",
    }

    for i, dc in enumerate(death_choices_trip):
        ax = axes[i]
        g = merged_maps[dc]

        # Separate mainland vs. inset geometries
        is_inset = g["_name_key"].isin(non_mainland_keys)
        g_main   = g.loc[~is_inset].copy()
        g_inset  = g.loc[is_inset].copy()

        # Main panel
        g_main.plot(
            ax=ax,
            column="pct_change",
            scheme="UserDefined",
            classification_kwds={"bins": bins},
            cmap="Spectral_r",
            edgecolor="k",
            linewidth=0.5,
            legend=(i == 2),                    # legend only on the right-most panel
            legend_kwds={"title": "Population Change (%)", "fontsize": 12, "loc": "lower left", "frameon": True, "edgecolor": "k",
                         "bbox_to_anchor": (1.8, 0.8), "loc":'upper right'})
        g_main.boundary.plot(ax=ax, color="#333333", linewidth=0.5, alpha=0.85)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(titles[dc], fontsize=12)

        # Inset(s): plot non-mainland departments in a neat top-right inset
        if not g_inset.empty:
            iax = nice_inset_ax(ax, width_frac=0.36, height_frac=0.36, loc="upper right", borderpad=0.6)
            g_inset.plot(
                ax=iax,
                column="pct_change",
                scheme="UserDefined",
                classification_kwds={"bins": bins},
                cmap="Spectral_r",
                edgecolor="k",
                linewidth=0.5,
                legend=False
            )
            g_inset.boundary.plot(ax=iax, color="#333333", linewidth=0.5, alpha=0.9)
            iax.set_aspect("equal")
            iax.set_axis_off()
            set_axes_to_bounds(iax, g_inset.total_bounds, pad_frac=0.10)
    plt.savefig(fig_path, bbox_inches="tight")



def plot_tfr_converge(asfr, scenario, death_choice, fig_path):
    #    @TODO this is generalised badly
    df = asfr.query(f"scenario == 'high' and death_choice == 'censo_2018'").copy()
    if df.empty:
        raise ValueError(f"No ASFR rows after filtering: death_choice={death_choice}, scenario={scenario}.")

    lo_w = df["EDAD"].map(parse_edad_width)
    df["age_lo"] = lo_w.map(lambda t: t[0]).astype(int)
    df["width"]  = lo_w.map(lambda t: t[1]).astype(int)

    # keep childbearing ages (15–49 inclusive); if a 80+ bin exists it is automatically excluded
    df = df[(df["age_lo"] >= 15) & (df["age_lo"] <= 49)].copy()

    # ---------------- TFR = sum(asfr * width) ----------------
    tfr = (
        df.assign(contrib = df["asfr"].astype(float) * df["width"].astype(float))
          .groupby(["DPTO_NOMBRE","year"], as_index=False)["contrib"]
          .sum()
          .rename(columns={"contrib": "TFR"})
          .sort_values(["DPTO_NOMBRE","year"])
    )

    # target TFR (from file if present; otherwise set explicitly)
    if "default_tfr_target" in asfr.columns and asfr["default_tfr_target"].notna().any():
        target_tfr = float(asfr.loc[asfr["default_tfr_target"].notna(), "default_tfr_target"].iloc[0])
    else:
        target_tfr = 1.45

    # ---------------- plotting with unique colors ----------------
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    # style national separately if present
    has_nat = "total_nacional" in tfr["DPTO_NOMBRE"].unique()
    nat = tfr[tfr["DPTO_NOMBRE"] == "total_nacional"] if has_nat else pd.DataFrame(columns=tfr.columns)
    rest = tfr[tfr["DPTO_NOMBRE"] != "total_nacional"] if has_nat else tfr

    dptos = sorted(rest["DPTO_NOMBRE"].unique())
    cmap  = cm.get_cmap("nipy_spectral", len(dptos))  # large, discrete palette

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for i, d in enumerate(dptos):
        g = rest[rest["DPTO_NOMBRE"] == d]
        ax.plot(g["year"], g["TFR"], lw=1.3, alpha=0.9, color=cmap(i), label=d)

    if has_nat and not nat.empty:
        ax.plot(nat["year"], nat["TFR"], lw=2.2, color="black", alpha=0.85, label="total_nacional")

    ax.axhline(target_tfr, ls="--", lw=1.6, color="black", alpha=0.8,
               label=f"Target TFR = {target_tfr:.2f}")

    ax.set_xlabel("Year")
    ax.set_ylabel("Total Fertility Rate (TFR)")
    ax.set_title(f"TFR convergence by DPTO — death_choice={death_choice}, scenario={scenario}")
    ax.grid(alpha=0.3, ls=":")

    # legend outside
    ax.set_title('a.', loc='left', fontsize=19, y=1.025, x=-0.05, fontweight='bold')
    ax.legend(fontsize=7, ncol=1, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')


def parse_edad_width(label: str) -> tuple[int, int]:
    """
    Return (lower_age, width_in_years) from EDAD label.
    Accepts 'x', 'x-y', 'x+'; inclusive upper bounds for 'x-y'.
    """
    s = str(label).strip()
    if s.endswith("+"):
        a = int(re.sub(r"\D", "", s))
        return a, 1  # width=1 suffices; we later restrict to <=49 so tail is dropped
    if "-" in s:
        lo, hi = s.split("-", 1)
        lo, hi = int(lo), int(hi)
        return lo, hi - lo + 1
    return int(s), 1


def plot_tfr(asfr, colors, year, fig_path):
    df = asfr.query(f"scenario == 'high' and death_choice == 'EEVV' and year == {year} and Sex == 'F'").copy()

    # ---------- helpers: parse EDAD -> (lower_age, width) ----------
    def parse_edad(s: str):
        s = str(s).strip()
        if s.endswith("+"):
            a = int(re.sub(r"\D", "", s))
            return a, math.inf
        if "-" in s:
            lo, hi = s.split("-", 1)
            lo, hi = int(lo), int(hi)
            return lo, hi - lo + 1
        return int(s), 1

    # parse and restrict to childbearing ages (lower bound in [15, 49])
    lo_w = df["EDAD"].apply(parse_edad)
    df["lower_age"] = lo_w.apply(lambda t: t[0])
    df["width"] = lo_w.apply(lambda t: (49 - t[0] + 1) if (math.isinf(t[1]) and t[0] <= 49) else t[1])
    df = df[(df["lower_age"] >= 15) & (df["lower_age"] <= 49)].copy()

    # ---------- compute TFR by DPTO ----------
    # TFR = sum_a ASFR_a * n_a over childbearing ages
    df["contrib"] = df["asfr"].astype(float) * df["width"].astype(float)
    tfr_by_dpto = (
        df.groupby("DPTO_NOMBRE", as_index=False)["contrib"]
        .sum()
        .rename(columns={"contrib": "TFR"})
        .sort_values("TFR", ascending=False)
    )

    # ---------- plot (single subfigure) ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(tfr_by_dpto["DPTO_NOMBRE"], tfr_by_dpto["TFR"], edgecolor="k")
    ax.set_title("TFR (year=2018) — death_choice=EEVV, scenario=high")
    ax.set_ylabel("Total Fertility Rate")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90)
    ax.set_title('a.', loc='left', fontsize=19, y=1.025, x=-0.05, fontweight='bold')
    fig.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')



def plot_pyramid(conteos, colors, fig_path, all_projections, proj_scenario='high',
                 proj_death_choice='censo_2018',
                 proj_year = 2070, conteos_source="censo_2018",
                 AGE_ORIENTATION = "youngest_bottom"):
    male_color = colors[0]
    fem_color  = colors[2]

    val_cols = [c for c in ["VALOR_corrected", "VALOR", "valor_corrected", "valor"] if c in conteos.columns]
    if not val_cols: raise KeyError(
        "No value column among ['VALOR_corrected','VALOR','valor_corrected','valor'] in conteos.rds.")
    VALCOL = val_cols[0]

    # -------------------- build pyramids --------------------
    pyr_2018 = agg_conteos_pyramid(conteos, VALCOL, year=2018, source=conteos_source)
    pyr_2070 = agg_projection_pyramid(all_projections, year=proj_year,
                                      scenario=proj_scenario,
                                      death_choice=proj_death_choice)

    # Harmonize bins
    bins_all = sorted(set(pyr_2018.index) | set(pyr_2070.index), key=bin_order_key)
    pyr_2018 = pyr_2018.reindex(bins_all, fill_value=0.0)
    pyr_2070 = pyr_2070.reindex(bins_all, fill_value=0.0)

    # -------------------- common x-scale --------------------
    max_abs = float(max(pyr_2018[["M","F"]].to_numpy().max(), pyr_2070[["M","F"]].to_numpy().max()))
    xlim = max_abs * 1.05 if max_abs > 0 else 1.0

    # -------------------- plot --------------------
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

    # 2018 pyramid
    ax = axes[0]
    ax.barh(pyr_2018.index, -pyr_2018["M"].values, color=male_color, edgecolor="k", height=0.85, label="Male")
    ax.barh(pyr_2018.index,  pyr_2018["F"].values, color=fem_color,  edgecolor="k", height=0.85, label="Female")
    ax.set_xlim(-xlim, xlim); ax.set_title("2018 (conteos, censo_2018)"); ax.set_xlabel("Population")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_abs)); ax.grid(axis="x", alpha=0.25, linestyle=":"); ax.legend()

    # 2070 pyramid (censo_2018 deaths)
    ax = axes[1]
    ax.barh(pyr_2070.index, -pyr_2070["M"].values, color=male_color, edgecolor="k", height=0.85, label="Male")
    ax.barh(pyr_2070.index,  pyr_2070["F"].values, color=fem_color,  edgecolor="k", height=0.85, label="Female")
    ax.set_xlim(-xlim, xlim); ax.set_title(f"{proj_year} (projection: scenario={proj_scenario}, death=censo_2018)")
    ax.set_xlabel("Population"); ax.xaxis.set_major_formatter(FuncFormatter(fmt_abs))
    ax.grid(axis="x", alpha=0.25, linestyle=":"); ax.legend(edgecolor='k')

    # shared y label & vertical orientation
    axes[0].set_ylabel("Age (5-year bins)")
    for ax in axes:
        if AGE_ORIENTATION == "youngest_top": ax.invert_yaxis()
        else:
            if ax.yaxis_inverted(): ax.invert_yaxis()

    axes[0].legend(frameon=True, loc="upper left", edgecolor='k')
    axes[1].legend(frameon=True, loc="upper left", edgecolor='k')
    fig.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')


def parse_age_lo(label):
    """Return lower bound of EDAD as int, or np.nan if unparseable."""
    if pd.isna(label): return np.nan
    s = str(label).strip()
    if s == "": return np.nan
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", s)
    if m: return int(m.group(1))
    m = re.fullmatch(r"\s*(\d+)\s*\+\s*", s)
    if m: return int(m.group(1))
    m = re.fullmatch(r"\s*(\d+)\s*", s)
    if m: return int(m.group(1))
    try: return int(float(s))
    except Exception: return np.nan

def to_5y_bin_label(lo: int) -> str:
    if lo >= 90: return "90+"
    k = (lo // 5) * 5
    return f"{k}-{k+4}"

def bin_order_key(lbl: str) -> int:
    s = str(lbl).strip()
    if s.endswith("+"): return 90
    return int(s.split("-")[0])

def sex_to_MF(x):
    try:
        xi = int(float(x))
        return "M" if xi == 1 else ("F" if xi == 2 else None)
    except Exception:
        s = str(x).strip().upper()
        if s in {"M","MALE"}: return "M"
        if s in {"F","FEMALE"}: return "F"
        return None

def agg_conteos_pyramid(conteos: pd.DataFrame, value_col: str, year: int, source: str) -> pd.DataFrame:
    df = conteos.copy()
    if "VARIABLE" in df.columns: df = df[df["VARIABLE"] == "poblacion_total"]
    if "ANO" in df.columns: df = df[df["ANO"] == year]
    if "FUENTE" in df.columns and source is not None: df = df[df["FUENTE"] == source]
    if "DPTO_NOMBRE" in df.columns and "total_nacional" in df["DPTO_NOMBRE"].astype(str).unique():
        df = df[df["DPTO_NOMBRE"] == "total_nacional"].copy()
    df["Sex"] = df["SEXO"].map(sex_to_MF)
    df = df[df["Sex"].isin(["M","F"])]
    df["age_lo"] = df["EDAD"].map(parse_age_lo)
    df = df.dropna(subset=["age_lo"]).copy()
    df["age_lo"] = df["age_lo"].astype(int)
    df["bin"] = df["age_lo"].map(to_5y_bin_label)
    grp = df.groupby(["bin","Sex"], as_index=False)[value_col].sum()
    out = grp.pivot(index="bin", columns="Sex", values=value_col).fillna(0.0)
    for c in ["M","F"]:
        if c not in out.columns: out[c] = 0.0
    out = out.reindex(sorted(out.index, key=bin_order_key))
    return out

def agg_projection_pyramid(proj: pd.DataFrame, year: int, scenario: str, death_choice: str) -> pd.DataFrame:
    df = proj[
        (proj["year"] == year) &
        (proj["scenario"] == scenario) &
        (proj["death_choice"] == death_choice) &
        (proj["Sex"].isin(["M","F"]))
    ].copy()
    if df.empty:
        raise ValueError(f"No projection rows for year={year}, scenario={scenario}, death_choice={death_choice}.")
    if "DPTO_NOMBRE" in df.columns and "total_nacional" in df["DPTO_NOMBRE"].astype(str).unique():
        df = df[df["DPTO_NOMBRE"] == "total_nacional"].copy()
    df["age_lo"] = df["EDAD"].map(parse_age_lo)
    df = df.dropna(subset=["age_lo"]).copy()
    df["age_lo"] = df["age_lo"].astype(int)
    df["bin"] = df["age_lo"].map(to_5y_bin_label)
    grp = df.groupby(["bin","Sex"], as_index=False)["population"].sum()
    out = grp.pivot(index="bin", columns="Sex", values="population").fillna(0.0)
    for c in ["M","F"]:
        if c not in out.columns: out[c] = 0.0
    out = out.reindex(sorted(out.index, key=bin_order_key))
    return out

def fmt_abs(x, pos):
    x_abs = abs(x)
    if x_abs >= 1e6: return f"{x_abs/1e6:.1f}M"
    if x_abs >= 1e3: return f"{x_abs/1e3:.0f}k"
    return f"{x_abs:.0f}"



def slice_e0(df, sex, death_choice, year):
    """Return dataframe of e0 at birth (mean ex at EDAD ∈ {0,0-1,0-4}) by scenario & DPTO."""
    d = df[(df['Sex'] == sex) &
           (df['death_choice'] == death_choice) &
           (df['year'] == year)]
    d = d[d['EDAD'].isin(['0','0-1','0-4'])]
    return d.groupby(['DPTO_NOMBRE','scenario'])['ex'].mean().unstack()


def plot_e0_birth(all_lt, colors, fig_path, death_choice, year_focus, target_year):
    # --- figure ---
    sex_labels = {"T": "Total", "M": "Male", "F": "Female"}
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, sharey=True)
    for i, (Sex, ax) in enumerate(zip(["T","M","F"], axes)):
        d = slice_e0(all_lt, Sex, death_choice, year_focus)
        if d is None or d.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        d = d[['low','mid','high']].dropna()
        if d.empty:
            ax.text(0.5, 0.5, "No common DPTOs", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        lo = d[['low','high']].min(axis=1)
        hi = d[['low','high']].max(axis=1)
        mid = d['mid']
        span = (hi - lo).abs()
        keep = span > 0
        if mid.empty:
            ax.text(0.5, 0.5, "low == mid == high (no error bars)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        order = mid.sort_values(ascending=False).index
        x = np.arange(len(order))
        y_mid = mid.loc[order].to_numpy()
        yerr = np.vstack([y_mid - lo.loc[order].to_numpy(),
                          hi.loc[order].to_numpy() - y_mid])
        ax.errorbar(
            x, y_mid,
            yerr=yerr,
            fmt='o', ms=10,
            mfc=colors[i], mec='black', mew=0.6,
            ecolor='black', elinewidth=1.2, capsize=3,
            alpha=0.9
        )
        ax.set_title(f"e0 at birth — {sex_labels[Sex]} "
                     f"({death_choice}, year={year_focus}; mid ± [low, high])")
        ax.set_ylabel("e0")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=90)
        ax.text(-0.05, 1.02, f"{chr(97+i)}.", transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='bottom', ha='left')
    fig.tight_layout()
    sns.despine()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def year_totaller(df, year='year'):
    return df.groupby([year, "DPTO_NOMBRE"])["population"]. \
        sum(). \
        reset_index()


def get_age_projections(all_proj, SEXO, death_choice, DPTO,
                        default_tfr_target,
                        improvement_total,
                        ma_window, target_year):
    all_proj = all_proj[all_proj['Sex'] == SEXO]
    all_proj = all_proj[all_proj['death_choice'] == death_choice]
    all_proj = all_proj[all_proj['DPTO_NOMBRE'] == DPTO]

    all_proj = all_proj[all_proj['default_tfr_target'] == default_tfr_target]
    all_proj = all_proj[all_proj['improvement_total'] == improvement_total]
    all_proj = all_proj[all_proj['ma_window'] == ma_window]
    all_proj = all_proj[all_proj['year'] <= target_year]

    df_mid = all_proj[all_proj['scenario'] == 'mid']
    df_high = all_proj[all_proj['scenario'] == 'high']
    df_low = all_proj[all_proj['scenario'] == 'low']

    df_mid = year_totaller(df_mid)
    df_low = year_totaller(df_low)
    df_high = year_totaller(df_high)
    return df_low, df_mid, df_high


def plot_death_choices(conteos, label, fig_path):
    value_cols = [c for c in ["VALOR",
                              "VALOR_corrected",
                              "valor",
                              "valor_corrected"] if c in conteos.columns]
    if not value_cols:
        raise KeyError("No value column found ['VALOR','VALOR_corrected','valor','valor_corrected'].")
    VALCOL = value_cols[0]

    # ---------- filter: deaths in 2018 for each source ----------
    required = ["VARIABLE", "FUENTE", "ANO", "DPTO_NOMBRE", VALCOL]
    missing = [c for c in required if c not in conteos.columns]
    if missing:
        raise KeyError(f"Missing required columns in conteos: {missing}")

    base = conteos.query("VARIABLE == 'defunciones' and ANO == 2018").copy()

    eevv = (
        base[base["FUENTE"] == "EEVV"]
        .groupby("DPTO_NOMBRE", as_index=False)[VALCOL].sum()
        .rename(columns={VALCOL: "deaths_EEVV_2018"})
    )

    censo = (
        base[base["FUENTE"] == "censo_2018"]
        .groupby("DPTO_NOMBRE", as_index=False)[VALCOL].sum()
        .rename(columns={VALCOL: "deaths_censo_2018"})
    )

    df = pd.merge(censo, eevv, on="DPTO_NOMBRE", how="inner")
    if df.empty:
        raise ValueError("No common DPTOs had both EEVV and censo_2018 death counts for 2018.")

    # ---------- plotting ----------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    x = df["deaths_censo_2018"].astype(float).to_numpy()
    y = df["deaths_EEVV_2018"].astype(float).to_numpy()

    ax.scatter(x, y, s=28, facecolor="#4575b4", edgecolor="k", alpha=0.9, zorder=2)

    # 45-degree reference line (y = x)
    lims_min = float(min(x.min(), y.min()))
    lims_max = float(max(x.max(), y.max()))
    pad = 0.06 * (lims_max - lims_min)
    lo, hi = lims_min - pad, lims_max + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black", alpha=0.7, label="y = x", zorder=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Deaths — censo_2018 (2018)")
    ax.set_ylabel("Deaths — EEVV (2018)")
#    ax.set_title("Deaths per DPTO in 2018: EEVV vs censo_2018")

    # ---------- label ALL points ----------
    # Use a small offset and a white stroke for readability.
    for _, r in df.iterrows():
        ax.annotate(
            r["DPTO_NOMBRE"],
            (float(r["deaths_censo_2018"]), float(r["deaths_EEVV_2018"])),
            textcoords="offset points", xytext=(4, 3), ha="left", va="bottom",
            fontsize=8, color="black",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)],
            zorder=3
        )

    ax.grid(True, alpha=0.25, linestyle=":", zorder=0)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(label, loc='left', fontsize=19, y=1.025, x=-0.05, fontweight='bold')
    sns.despine()
    fig.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')



def plot_proj(all_projections,
              Sex,
              death_choice,
              DPTO_NOMBRE,
              default_tfr_target,
              improvement_total,
              ma_window,
              target_year,
              title_label,
              fig_path,
              colors,
              DANE=True):
    df_low, df_mid, df_high = get_age_projections(all_projections,
                                                  Sex,
                                                  death_choice,
                                                  DPTO_NOMBRE,
                                                  default_tfr_target,
                                                  improvement_total,
                                                  ma_window,
                                                  target_year)

    if DANE:
        df = pd.read_csv('../data/Projections_DANE_NAL_2018_2070.csv')
        df_clean = (df.sort_values('POPULATION')
                    .drop_duplicates(subset=['ANO', 'SEXO', 'EDAD'], keep='last'))
        if Sex == 'T':
            totals_from_ages = (df_clean.query("SEXO == 'Total' and EDAD.notna()")
                                .groupby('ANO')['POPULATION'].sum())
        elif Sex == 'M':
            totals_from_ages = (df_clean.query("SEXO == 'Hombres' and EDAD.notna()")
                                .groupby('ANO')['POPULATION'].sum())
        elif Sex == 'F':
            totals_from_ages = (df_clean.query("SEXO == 'Mujeres' and EDAD.notna()")
                                .groupby('ANO')['POPULATION'].sum())
    # ---------------------------------
    # Plot
    # ---------------------------------
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6.5))

    ax1.plot(
        df_low.loc[:, "year"],
        df_low.loc[:, "population"],
        marker="x",
        color=colors[0],
        label="Low",
        linestyle='--'
    )
    ax1.plot(
        df_mid.loc[:, "year"],
        df_mid.loc[:, "population"],
        marker="*",
        color=colors[1],
        label="Mid",
        linestyle='--'
    )
    ax1.plot(
        df_high.loc[:, "year"],
        df_high.loc[:, "population"],
        marker=".",
        color=colors[2],
        label="High",
        linestyle='--'
    )
    if DANE:
        ax1.plot(totals_from_ages.index, totals_from_ages, label='DANE', linestyle='--', color='k')

    # Uncertainty band from E low/high (aligned by year)
    low = (df_low.loc[:, ["year", "population"]]
           .set_index("year").sort_index())
    high = (df_high.loc[:, ["year", "population"]]
            .set_index("year").sort_index())
    band = low.join(high, how="inner", lsuffix="_low", rsuffix="_high")
    y1 = band[["population_low", "population_high"]].min(axis=1)
    y2 = band[["population_low", "population_high"]].max(axis=1)
    ax1.fill_between(band.index, y1, y2, color=(255 / 255, 223 / 255, 0 / 255, 10 / 255))

    # Axes cosmetics
    for ax in [ax1]:
        ax.set_ylabel("Population", fontsize=13)
        ax.set_xlabel('Year', fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend(loc='upper right', frameon=True,
                  fontsize=10, framealpha=1, facecolor='w',
                  edgecolor=(0, 0, 0, 1),
                  ncol=2,
                  title='Omissions')

    ax1.set_title(title_label, loc='left', fontsize=19, y=1.025, x=-0.05, fontweight='bold')

    lines = [
        f"Projections: 2018-{target_year}  | SEXO: {Sex} | Region: {DPTO_NOMBRE}",
        f"Death Choice: {death_choice} | TFR target: {default_tfr_target}",
        f"Mortality improvement: {improvement_total}, MA window = {ma_window})"
    ]

    # Render centered annotation inside the figure
    fig.text(
        0.5, 0.15, "\n".join(lines),
        ha='center', va='bottom', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, linewidth=0.5)
    )

    sns.despine()
    plt.savefig(fig_path,
                bbox_inches='tight')