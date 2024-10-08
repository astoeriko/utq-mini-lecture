import numpy as np
import xarray as xr
import hvplot.xarray
import panel as pn
import holoviews as hv


def make_concentration_dataarray(c, x, time):
    "Create a labeled DataArray with coordinates for the concentration"
    return xr.DataArray(c).assign_coords(x=x, time=time).rename("c")


def analytical_solution(func, x, t, c_boundary, v, D):
    """Compute the analytical solution with `func` and return labeled array

    Parameters
    ----------
    func: function
      It should accept 8 arguments, namely a spatial coordinate `x`, a time coordinate
      `t`, a boundary concentration `c_boundary`, a velocity `v`, dispersion
      coefficient `D`.
    """
    c = func(x, t, c_boundary, v, D)
    return make_concentration_dataarray(c, x, t)


def make_fixed_coordinates():
    "Create Arrays for the space and time dimension"
    x = xr.DataArray(
        np.linspace(0, 200, num=11), dims="x", attrs=dict(units="m"), name="x"
    )
    t = xr.DataArray(
        np.linspace(0, 20, num=201), dims="time", attrs=dict(units="years"), name="t"
    )
    return x, t


def make_coordinate_widgets(t):
    "Make sliders to vary the observation time and observation location"
    i_observation = pn.widgets.Player(
        name="step", start=0, end=len(t) - 1, align="center", interval=40
    )
    t_observation = t.interactive().isel(time=i_observation)
    x_observation = pn.widgets.FloatSlider(
        name="x", start=0.1, end=200, step=20, value=5, align="center"
    )
    return x_observation, t_observation


def make_parameter_widgets():
    "Create sliders to vary the parameter values"
    c_boundary = 1
    D = pn.widgets.FloatSlider(name="D", start=1, end=10, step=2, value=3)
    v = pn.widgets.FloatSlider(name="v", start=5, end=30, step=2, value=10)
    return c_boundary, D, v


def make_interactive_concentration(func, x, t, c_boundary, v, D):
    "Create an interactive array containing the analytical solution"
    return hvplot.bind(
        analytical_solution,
        func,
        x,
        t,
        c_boundary,
        v,
        D,
    ).interactive()


def make_plots(profile, btc, c_boundary):
    "Create plots of the profile and break-through curve"
    hvplot.output(backend="bokeh")
    plot_profile = profile.hvplot(ylim=(0, c_boundary)).opts(
        tools=["hover", "box_zoom", "reset", "save"], default_tools=[]
    )
    plot_btc = btc.hvplot(ylim=(0, c_boundary)).opts(
        tools=["hover", "box_zoom", "reset", "save"], default_tools=[]
    )
    return plot_profile, plot_btc


def arrange_plots_and_widgets(
    plot_profile, plot_btc, D, v, t_observation, x_observation
):
    "Arrange all plots and widgets"
    return pn.Column(
        # pn.Row(
        # pn.Column("# Profile", plot_profile.panel(), t_observation.widgets()),
        # pn.Column("# Profile", plot_profile.panel()),
        # pn.layout.Divider(),
        # pn.Column("# Breakthrough Curve", plot_btc.panel(), x_observation),
        "# Parameters",
        pn.WidgetBox(
            pn.Row(
                D,
                v,
            ),
            align="center",
        ),
        pn.layout.Spacer(height=30),
        pn.Column("# Breakthrough Curve", plot_btc.panel()),
    )


def plot_solution_interactive(func):
    "Create an interactive visualization of the analytical solution"
    c_boundary, D, v = make_parameter_widgets()
    x, t = make_fixed_coordinates()
    # x_observation, t_observation = make_coordinate_widgets(t)
    x_observation = 100
    t_observation = 10

    profile = make_interactive_concentration(
        func,
        x,
        t_observation,
        c_boundary,
        v,
        D,
    )

    btc = make_interactive_concentration(
        func,
        x_observation,
        t,
        c_boundary,
        v,
        D,
    )
    plot_profile, plot_btc = make_plots(profile, btc, c_boundary)
    return arrange_plots_and_widgets(
        plot_profile, plot_btc, D, v, t_observation, x_observation
    )
