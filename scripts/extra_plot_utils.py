
from typing import TYPE_CHECKING, Any, Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib.font_manager import FontProperties
from typing import Optional
from pcntoolkit.dataio.norm_data import NormData

if TYPE_CHECKING:
    from pcntoolkit.normative_model import NormativeModel
import os
import copy

def plot_centiles_log(
    model: "NormativeModel",
    scatter_data: Optional[NormData] = None,
    centiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
    covariate: str | None = None,
    scatter_kwargs: dict = {},
    save_dir: str | None = None,
    log_transform: bool = False,
):
    """
    Plot the centiles of the model.

    Parameters
    ----------
    model: NormativeModel
        The model to plot the centiles for.
    scatter_data: NormData
        The data to scatter on top of the centiles.
    centiles: List[float], optional
        The centiles to plot.
    covariate: str, optional
        The covariate to plot on the x-axis.
    scatter_kwargs: dict, optional
        Keyword arguments for the scatter plot.
        May include:
        - color: The color of the scatter points. Hex code or matplotlib color name.
        - alpha: The transparency of the scatter points. Between 0 and 1.
        - s: The size of the scatter points.
        - marker: The marker of the scatter points. Uses matplotlib marker syntax: https://matplotlib.org/stable/api/markers_api.html
        - edgecolor: The edge color of the scatter points. Hex code or matplotlib color name.
        - linewidth: The width of the edge of the scatter points. 0 for no edge.
    """
    default_scatter_kwargs = {
        "color": "#f7932f",
        "alpha": min(1, 20/np.sqrt(len(scatter_data.X))),
        "s": 30,
        "marker": "o",
        "edgecolor": "black",
        "linewidth": 0,
    }
    complete_scatter_kwargs = default_scatter_kwargs | scatter_kwargs


    if covariate is None:
        covariate = model.covariates[0]
        assert isinstance(covariate, str)
    else:
        assert covariate in model.covariates, f"{covariate} is not a valid covariate for the model"
    cov_min = model.covariate_ranges[covariate]["min"]
    cov_max = model.covariate_ranges[covariate]["max"]
    covariate_range = (cov_min, cov_max)

    batch_effects = {k: max(v.items(), key=lambda x: x[1])[0] for k, v in model.batch_effect_counts.items()}

    # Create some synthetic data with a single batch effect
    # The plotted covariate is just a linspace
    centile_covariates = np.linspace(covariate_range[0], covariate_range[1], 150)
    centile_df = pd.DataFrame({covariate: centile_covariates})

    # TODO: use the mean here
    # Any other covariates are taken to be the midpoint between the observed min and max
    for cov in model.covariates:
        if cov != covariate:
            minc = model.covariate_ranges[cov]["min"]
            maxc = model.covariate_ranges[cov]["max"]
            centile_df[cov] = (minc + maxc) / 2

    # Batch effects are the first ones in the highlighted batch effects
    for be, v in batch_effects.items():
        centile_df[be] = v
    # Response vars are all 0, we don't need them
    for rv in model.response_vars:
        centile_df[rv] = 0

    centile_data = NormData.from_dataframe(
        "centile",
        dataframe=centile_df,
        covariates=model.covariates,
        response_vars=model.response_vars,
        batch_effects=list(batch_effects.keys()),
    )  # type:ignore

    if not hasattr(centile_data, "centiles"):
        model.compute_centiles(centile_data, centiles=centiles, recompute=False)

    if not model.has_batch_effect:
        batch_effects = {}

    if scatter_data:
        model.harmonize(scatter_data, reference_batch_effect=batch_effects)

    for response_var in model.response_vars:
        _plot_centiles_log(centile_data=centile_data, response_var=response_var, covariate=covariate, scatter_data=scatter_data, scatter_kwargs=complete_scatter_kwargs, save_dir=save_dir, log_transform=log_transform)


def _plot_centiles_log(
    centile_data: NormData,
    response_var: str,
    covariate: str = None,  # type: ignore
    scatter_data: Optional[NormData] = None,
    scatter_kwargs: dict = {},
    save_dir: str | None = None,
    log_transform: bool = False,
) -> None:
    sns.set_style("whitegrid")
    plt.figure()

    filter_dict = {
        "covariates": covariate,
        "response_vars": response_var,
    }

    filtered = centile_data.sel(filter_dict)

    for centile in centile_data.coords["centile"][::-1]:
        d_mean = abs(centile - 0.5)
        if d_mean == 0:
            thickness = 2
        else:
            thickness = 1
        if d_mean <= 0.25:
            style = "-"
        elif d_mean <= 0.475:
            style = "--"
        else:
            style = ":"

        yvec =filtered.centiles.sel(centile=centile).to_numpy()
        if log_transform:
            yvec = np.expm1(yvec)
        sns.lineplot(
            x=filtered.X,
            y=yvec,
            color="black",
            linestyle=style,
            linewidth=thickness,
            zorder=2,
            legend="brief",
        )

        font = FontProperties()
        font.set_weight("bold")
        yvec = filtered.centiles.sel(centile=centile)[0].to_numpy()
        if log_transform:
            yvec = np.expm1(yvec)
        plt.text(
            s=centile.item(),
            x=filtered.X[0] - 1,
            y=yvec,
            color="black",
            horizontalalignment="right",
            verticalalignment="center",
            fontproperties=font,
        )
        yvec = filtered.centiles.sel(centile=centile)[-1].to_numpy()
        if log_transform:
            yvec = np.expm1(yvec)
        plt.text(
            s=centile.item(),
            x=filtered.X[-1] + 1,
            y=yvec,
            color="black",
            horizontalalignment="left",
            verticalalignment="center",
            fontproperties=font,
        )

    minx, maxx = plt.xlim()
    plt.xlim(minx - 0.1 * (maxx - minx), maxx + 0.1 * (maxx - minx))

    if scatter_data:
        scatter_filter = scatter_data.sel(filter_dict)
        df = scatter_filter.to_dataframe()
        data_name = "Y_harmonized"
        columns = [("X", covariate), (data_name, response_var)]
        columns.extend([("batch_effects", be.item()) for be in scatter_data.batch_effect_dims])
        df = df[columns]
        df.columns = [c[1] for c in df.columns]
        df[response_var]
        # sns.scatterplot(
        #     data=df,
        #     x=covariate,
        #     y=response_var,
        #     **scatter_kwargs
        # )
        xvec = df[covariate].to_numpy()
        yvec = df[response_var].to_numpy()
        if log_transform:
            yvec = np.expm1(yvec)
        sns.scatterplot(
            x=xvec,
            y=yvec,
            **scatter_kwargs
        )

        if scatter_data:
            plotname = f"centiles_{response_var}_{scatter_data.name}_harmonized"
            title = f"Centiles of {response_var}\n With harmonized {scatter_data.name} data"
        else:
            plotname = f"centiles_{response_var}"
            title = f"Centiles of {response_var}"

    plt.title(title)
    plt.xlabel(covariate)
    plt.ylabel(response_var)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{plotname}.png"), dpi=300)
    else:
        plt.show(block=False)
    plt.tight_layout()
    plt.close()