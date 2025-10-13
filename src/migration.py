# src/migration.py


import numpy as np
import pandas as pd

EDAD_ORDER = [
    '0-4','5-9','10-14','15-19','20-24','25-29',
    '30-34','35-39','40-44','45-49','50-54','55-59',
    '60-64','65-69','70-74','75-79','80+'
]

def create_migration_frame(conteos: pd.DataFrame, year: int | None = 2018) -> pd.DataFrame:
    """
    Build a national (ANO, EDAD, SEXO) panel with:
      inmigracion_F, emigracion_F, net_migration, poblacion_total, net_mig_rate.

    Inputs consumed from `conteos`:
      - VARIABLE in {"poblacion_total", "flujo_inmigracion", "flujo_emigracion"}
      - ANO, EDAD, SEXO, VALOR

    If `year` is not None, the returned panel is filtered to that ANO; otherwise all years are returned.
    """

    IN = "flujo_inmigracion"
    OUT = "flujo_emigracion"

    df = conteos.copy()

    # Coerce numerics & normalize labels
    df["VALOR"] = pd.to_numeric(df["VALOR"], errors="coerce").fillna(0.0)
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype("Int64")
    df["SEXO"] = pd.to_numeric(df["SEXO"], errors="coerce").astype("Int64")
    df["EDAD"] = df["EDAD"].astype(str).str.strip()

    # Population by (ANO, EDAD, SEXO) -- national totals (sum over DPTO)
    pop_nat = (
        df.loc[df["VARIABLE"] == "poblacion_total", ["ANO", "EDAD", "SEXO", "VALOR"]]
          .groupby(["ANO","EDAD","SEXO"], as_index=False)["VALOR"].sum()
          .rename(columns={"VALOR": "poblacion_total"})
    )

    # Migration movements (entries/exits), national totals
    mig = df.loc[df["VARIABLE"].isin([IN, OUT]), ["ANO","EDAD","SEXO","VARIABLE","VALOR"]]

    # Pivot to wide with zeros for missing categories
    mig_nat = (
        pd.pivot_table(
            mig,
            index=["ANO","EDAD","SEXO"],
            columns="VARIABLE",
            values="VALOR",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .rename(columns={IN: "inmigracion_F", OUT: "emigracion_F"})
    )

    # Ensure both migration columns exist (in case one is entirely absent)
    for col in ["inmigracion_F", "emigracion_F"]:
        if col not in mig_nat.columns:
            mig_nat[col] = 0.0

    # Merge population and movements; keep population grid
    nat_age_sex = pop_nat.merge(mig_nat, on=["ANO","EDAD","SEXO"], how="left")
    nat_age_sex[["inmigracion_F","emigracion_F"]] = nat_age_sex[["inmigracion_F","emigracion_F"]].fillna(0.0)

    # Derived quantities
    nat_age_sex["net_migration"] = nat_age_sex["inmigracion_F"] - nat_age_sex["emigracion_F"]
    denom = pd.to_numeric(nat_age_sex["poblacion_total"], errors="coerce")
    nat_age_sex["net_mig_rate"] = np.where(denom > 0, nat_age_sex["net_migration"] / denom, np.nan)

    # Ordered ages, numeric SEXO
    nat_age_sex["EDAD"] = pd.Categorical(nat_age_sex["EDAD"], categories=EDAD_ORDER, ordered=True)
    nat_age_sex["SEXO"] = pd.to_numeric(nat_age_sex["SEXO"], errors="coerce")

    out_cols = [
        "ANO","EDAD","SEXO","inmigracion_F","emigracion_F",
        "net_migration","poblacion_total","net_mig_rate"
    ]
    out_cols = [c for c in out_cols if c in nat_age_sex.columns]

    panel = (
        nat_age_sex[out_cols]
        .sort_values(["ANO","EDAD","SEXO"], kind="mergesort")
        .reset_index(drop=True)
    )

    if year is not None:
        panel = panel[panel["ANO"] == year].reset_index(drop=True)

    return panel
