from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas
import plotly.express as px

from trame.app import get_server
from trame.decorators import TrameApp, change
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import html, markdown, matplotlib, plotly, trame, vega, vuetify3

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

ABOUT_HELP = """
## About
 - Data: [U.S. Census Bureau](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html).
 - <span style="color:orange">**Components**</span>: Decade population change and the natural and migration additions and subtractions.
 - <span style="color:orange">**Years**</span>: Individual years in the decade that spans 2010 to 2019.
 - <span style="color:orange">**Gains/Losses**</span>: states with high and low annual population growth for selected year.
 - <span style="color:orange">**States Growth**</span>: percentage of states with above 50K and below -50K annual population growth.
"""

COMPONENTS = [
    "Change",
    "Natural",
    "Births",
    "Deaths",
    "Migration",
    "International",
    "Domestic",
]

YEARS = [
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
]

THEMES = [
    "blues",
    "cividis",
    "greens",
    "inferno",
    "magma",
    "plasma",
    "reds",
    "rainbow",
    "turbo",
    "viridis",
]

TABLE_HEADER = [
    {"title": "State", "key": "state", "sortable": False},
    {"title": "Population", "key": "population", "sortable": False},
]

COMPONENTS_AND_2010 = [*COMPONENTS, "2010"]

DATA_FILE = str(Path(__file__).with_name("us_population.csv").resolve())

# -----------------------------------------------------------------
# Helper methods using various libraries
# -----------------------------------------------------------------


# Convert population to text
def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f"{num // 1000000} M"
        return f"{round(num / 1000000, 1)} M"
    return f"{num // 1000} K"


# Calculation year-over-year population migrations
def calculate_population_difference(input_df, input_year):
    if input_year in COMPONENTS_AND_2010:
        return None
    else:
        selected_df = input_df[input_df["year"] == input_year].reset_index()
        previous_df = input_df[
            input_df["year"] == str(int(input_year) - 1)
        ].reset_index()
        selected_df["difference"] = selected_df.population.sub(
            previous_df.population, fill_value=0
        )
        return pandas.concat(
            [
                selected_df.states,
                selected_df.id,
                selected_df.population,
                selected_df.difference,
            ],
            axis=1,
        ).sort_values(by="difference", ascending=False)


# Gains - Markdown
def make_gains_loss(input_df, input_year, gain=True):
    index = 0 if gain else -1
    if input_year in COMPONENTS_AND_2010:
        name, population, delta, color, symbol = "N/A", "0 M", "0 K", "black", "&harr;"
    else:
        name = input_df.states.iloc[index]
        population = format_number(input_df.population.iloc[index])
        delta = format_number(input_df.difference.iloc[index])
        if delta[0] != "-":
            color, symbol = "green", "&uarr;"
        else:
            color, symbol = "red", "&darr;"

    return f"{name}  \n<span style='font-size:2.0em;'>{population}</span>  \n<span style='color:{color}'>{symbol}{delta}</span>"


# Donut chart - Altair/vega
def make_donut(input_value, input_text, option):
    chart_color = (
        ["#27AE60", "#12783D"] if option == "above" else ["#E74C3C", "#781F16"]
    )
    source = pandas.DataFrame(
        {"Topic": ["", input_text], "% value": [100 - input_value, input_value]}
    )
    source_bg = pandas.DataFrame({"Topic": ["", input_text], "% value": [100, 0]})

    plot = (
        alt.Chart(source)
        .mark_arc(innerRadius=45, cornerRadius=25)
        .encode(
            theta="% value",
            color=alt.Color(
                "Topic:N",
                scale=alt.Scale(domain=[input_text, ""], range=chart_color),
                legend=None,
            ),
        )
        .properties(width=130, height=130)
    )
    text = plot.mark_text(
        align="center",
        color=chart_color[0],
        font="Lato",
        fontSize=32,
        fontWeight=700,
        fontStyle="italic",
    ).encode(text=alt.value(f"{input_value} %"))
    plot_bg = (
        alt.Chart(source_bg)
        .mark_arc(innerRadius=45, cornerRadius=20)
        .encode(
            theta="% value",
            color=alt.Color(
                "Topic:N",
                scale=alt.Scale(domain=[input_text, ""], range=chart_color),
                legend=None,
            ),
        )
        .properties(width=130, height=130)
    )
    return plot_bg + plot + text


# Choropleth map - Plotly
def make_choropleth(input_df, input_id, input_column, input_color_theme, height):
    choropleth = px.choropleth(
        input_df,
        locations=input_id,
        color=input_column,
        locationmode="USA-states",
        color_continuous_scale=input_color_theme,
        range_color=(min(input_df.population), max(input_df.population)),
        scope="usa",
        labels={"population": ""},
    )
    choropleth.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height)
    return choropleth


# Heatmap - Altair/Vega
def make_heatmap(
    input_df, input_y, input_x, input_color, input_color_theme, width, height
):
    heatmap = (
        alt.Chart(input_df)
        .mark_rect()
        .encode(
            y=alt.Y(
                f"{input_y}:O",
                axis=alt.Axis(
                    title="",
                    titleFontSize=18,
                    titlePadding=15,
                    titleFontWeight=900,
                    labelAngle=0,
                ),
            ),
            x=alt.X(
                f"{input_x}:O",
                axis=alt.Axis(
                    title="", titleFontSize=18, titlePadding=15, titleFontWeight=900
                ),
            ),
            color=alt.Color(
                f"max({input_color}):Q",
                legend=None,
                scale=alt.Scale(scheme=input_color_theme),
            ),
            stroke=alt.value("black"),
            strokeWidth=alt.value(0.25),
        )
        .properties(width=int(width - 60), height=int(height - 130))
        .configure_axis(labelFontSize=12, titleFontSize=12)
    )
    return heatmap


# Line - matplotlib
def make_line(input_years, input_population, width, height, dpi):
    w = (width - 10) / dpi
    h = (height - 10) / dpi
    plt.figure(figsize=(w, h), layout="compressed")
    plt.plot(np.asarray(input_years, int), np.asarray(input_population, int))
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xticks([])
    return fig


# Top states - tables
def make_top(input_df, count=5):
    np_input = input_df.to_numpy()
    top_states = []
    population_max = float(max(input_df["population"]))
    population_min = float(min(input_df["population"]))
    if abs(population_min) > population_max:
        population_max = abs(population_min)
    for i in range(0, count):
        percent = round(100 * float(np_input[i][5]) / population_max)
        top_states.append({"state": np_input[i][1], "population": percent, "rank": i})
    return top_states


# Bottom states - tables
def make_bottom(input_df, count=5):
    df_reverse_rows = input_df.iloc[::-1]
    np_input = df_reverse_rows.to_numpy()
    bottom_states = []
    population_max = float(max(input_df["population"]))
    population_min = float(min(input_df["population"]))
    if abs(population_min) > population_max:
        population_max = abs(population_min)
    for i in range(0, count):
        percent = round(100 * float(np_input[i][5]) / population_max)
        bottom_states.append(
            {"state": np_input[i][1], "population": percent, "rank": i}
        )
    return bottom_states


# -----------------------------------------------------------------
# Trame application defining interaction and user interface
# -----------------------------------------------------------------


@TrameApp()
class UsPopulation:
    def __init__(self, file_to_load, server=None):
        self.server = get_server(server, client_type="vue3")
        self.last_computed_component = None

        # dataset
        self.df_raw = pandas.read_csv(file_to_load)
        self.df_years = self.df_raw[self.df_raw.year.str.contains("20")]
        self.population = [
            int(self.df_raw[self.df_raw.year == year]["population"].sum())
            for year in YEARS
        ]

        # matplotlib plot
        self.line = None

        # Web UI
        self.ui = self._build_ui()

    @property
    def state(self):
        return self.server.state

    @property
    def ctrl(self):
        return self.server.controller

    def _calculate_dataframes(self, active_selection):
        if self.last_computed_component == active_selection:
            return

        self.last_computed_component = active_selection
        self.df_selected_cmp_or_yr = self.df_raw[
            self.df_raw.year == active_selection
        ]
        self.df_selected_sorted = self.df_selected_cmp_or_yr.sort_values(
            by="population", ascending=False
        )
        self.df_difference_sorted = calculate_population_difference(
            self.df_raw, active_selection
        )

    @change("active_selection", "nb_states")
    def update_top_bottom(self, active_selection, nb_states, **kwargs):
        self._calculate_dataframes(active_selection)
        self.state.top_states = make_top(self.df_selected_sorted, nb_states)
        self.state.bottom_states = make_bottom(self.df_selected_sorted, nb_states)

    @change("active_selection")
    def update_gains_losses(self, active_selection, **kwargs):
        self._calculate_dataframes(active_selection)
        self.ctrl.gains_update(
            make_gains_loss(
                self.df_difference_sorted, active_selection, gain=True
            )
        )
        self.ctrl.losses_update(
            make_gains_loss(
                self.df_difference_sorted, active_selection, gain=False
            )
        )

    @change("active_selection")
    def update_donuts(self, active_selection, **kwargs):
        self._calculate_dataframes(active_selection)
        if active_selection in COMPONENTS_AND_2010:
            states_above = 0
            states_below = 0
            donut_above = make_donut(states_above, "Above", "above")
            donut_below = make_donut(states_below, "Below", "below")
        else:
            df_greater_50000 = self.df_difference_sorted[
                self.df_difference_sorted.difference > 50000
            ]
            df_less_50000 = self.df_difference_sorted[
                self.df_difference_sorted.difference < -50000
            ]
            states_above = round(
                (len(df_greater_50000) / self.df_difference_sorted.states.nunique())
                * 100
            )
            states_below = round(
                (len(df_less_50000) / self.df_difference_sorted.states.nunique()) * 100
            )
            donut_above = make_donut(states_above, "Above", "above")
            donut_below = make_donut(states_below, "Below", "below")
        self.ctrl.above_view_update(donut_above)
        self.ctrl.below_view_update(donut_below)

    @change("active_selection", "color_theme", "map_size")
    def update_choropleth(
        self, active_selection, color_theme, map_size, **kwargs
    ):
        self._calculate_dataframes(active_selection)
        if map_size is None:
            return
        height = map_size.get("size").get("height")

        choropleth = make_choropleth(
            self.df_selected_cmp_or_yr,
            "states_code",
            "population",
            color_theme,
            height,
        )
        self.ctrl.choropleth_view_update(choropleth)
        self.state.figure_ready = True

    @change("active_selection", "color_theme", "heatmap_size")
    def update_heatmap(
        self, active_selection, color_theme, heatmap_size, **kwargs
    ):
        self._calculate_dataframes(active_selection)
        if heatmap_size is None:
            return
        size = heatmap_size.get("size")
        heatmap = make_heatmap(
            self.df_years,
            "year",
            "states",
            "population",
            color_theme,
            size.get("width"),
            size.get("height"),
        )
        self.ctrl.heatmap_view_update(heatmap)

    @change("line_size")
    def update_line_size(self, line_size, **kwargs):
        if self.line != None:
            plt.close(self.line)
        if line_size is None:
            self.line = make_line(YEARS, self.population, 300, 300, 192)
            self.ctrl.line_view_update(self.line)
            return
        size = line_size.get("size")
        width = size.get("width")
        height = size.get("height")
        dpi = line_size.get("dpi")
        self.line = make_line(YEARS, self.population, width, height, dpi)
        self.ctrl.line_view_update(self.line)

    def _build_ui(self):
        self.state.trame__title = "US Population"

        with SinglePageLayout(self.server) as layout:
            # Toolbar
            with layout.toolbar.clear() as toolbar:
                toolbar.density = "compact"
                vuetify3.VToolbarTitle("&#x1f1fa;&#x1f1f8; Population")
                vuetify3.VSpacer()
                with vuetify3.VRow(align="center", classes="mx-4"):
                    with vuetify3.VCol():
                        vuetify3.VSelect(
                            v_model=("active_selection", "2011"),
                            items=("component_year", [*COMPONENTS, *YEARS]),
                            label="Select data component or year",
                            density="compact",
                            hide_details=True,
                            outlined=True,
                        )
                    with vuetify3.VCol():
                        vuetify3.VSelect(
                            v_model=("color_theme", "blues"),
                            items=("theme_options", THEMES),
                            label="Select color theme",
                            density="compact",
                            hide_details=True,
                            outlined=True,
                        )
                    with vuetify3.VBtn(
                        icon=True,
                        density="compact",
                        hide_details=True,
                        click="show_about = !show_about",
                    ):
                        vuetify3.VIcon("mdi-help")

            # Main content
            with layout.content:
                with vuetify3.VContainer(fluid=True):
                    # About
                    with vuetify3.VDialog(v_model=("show_about", False)):
                        with vuetify3.VCard(
                            classes="pa-4 mx-auto", style="width: 80vw;"
                        ):
                            markdown.Markdown(ABOUT_HELP, classes="ma-2")

                    # Flow of data
                    with vuetify3.VRow(align="stretch"):
                        with vuetify3.VCol(cols=12, md=4):
                            with vuetify3.VCard(
                                classes="fill-height",
                                style="min-height: max(200px, 40vh)",
                            ):
                                vuetify3.VCardTitle("Population over time")
                                with html.Div(
                                    style="height: calc(100% - 4rem); margin-top: -1rem;"
                                ):
                                    with trame.SizeObserver("line_size"):
                                        self.ctrl.line_view_update = (
                                            matplotlib.Figure().update
                                        )

                        with vuetify3.VCol(cols=12, md=5):
                            with vuetify3.VCard(
                                classes="fill-height",
                                style="min-height: max(200px, 40vh)",
                            ):
                                vuetify3.VCardTitle(
                                    "Population {{ active_selection }}"
                                )
                                with html.Div(style="height: calc(100% - 4rem);"):
                                    with trame.SizeObserver("map_size"):
                                        self.ctrl.choropleth_view_update = (
                                            plotly.Figure(
                                                display_mode_bar=("false",),
                                                v_show=("figure_ready", False),
                                            ).update
                                        )

                        with vuetify3.VCol(cols=12, md=3):
                            with vuetify3.VCard(classes="fill-height"):
                                with vuetify3.VCardTitle():
                                    with vuetify3.VRow(
                                        classes="pa-2",
                                        align="center",
                                        justify="space-between",
                                    ):
                                        html.Div("Top States")
                                        vuetify3.VTextField(
                                            v_model_number=("nb_states", 5),
                                            type="number",
                                            step=1,
                                            max=10,
                                            min=5,
                                            density="compact",
                                            hide_details=True,
                                            variant="solo",
                                            flat=True,
                                            __properties=["min", "max", "step"],
                                            style="max-width: 5rem;",
                                        )
                                with vuetify3.VCardText():
                                    with vuetify3.VDataTable(
                                        headers=("table_headers", TABLE_HEADER),
                                        items=("top_states", []),
                                        density="compact",
                                    ):
                                        with vuetify3.Template(
                                            raw_attrs=[
                                                'v-slot:item.population="{ value }"'
                                            ]
                                        ):
                                            with vuetify3.VProgressLinear(
                                                model_value=("value",),
                                                color=("value>0 ? 'green':'red'",),
                                                height="25",
                                            ):
                                                with vuetify3.Template(
                                                    raw_attrs=[
                                                        'v-slot:default="{ value }"'
                                                    ]
                                                ):
                                                    html.Div(
                                                        "<strong>{{Math.round(value)}}%</strong>"
                                                    )
                                        vuetify3.Template(raw_attrs=["v-slot:bottom"])

                        with vuetify3.VCol(cols=6, md=2):
                            with vuetify3.VCard(classes="fill-height"):
                                vuetify3.VCardTitle("Gains/Losses")
                                with html.Div(
                                    style="height: calc(100% - 3rem);",
                                    classes="d-flex flex-column",
                                ):
                                    with vuetify3.VRow(
                                        justify="center", align="center"
                                    ):
                                        self.ctrl.gains_update = markdown.Markdown(
                                            classes="pa-4"
                                        ).update
                                    with vuetify3.VRow(
                                        justify="center", align="center"
                                    ):
                                        self.ctrl.losses_update = markdown.Markdown(
                                            classes="pa-4"
                                        ).update

                        with vuetify3.VCol(cols=6, md=2):
                            with vuetify3.VCard(classes="fill-height"):
                                vuetify3.VCardTitle("States Growth")
                                with vuetify3.VCol(align_self="stretch"):
                                    vuetify3.VCardSubtitle("Above")
                                    with html.Div(classes="text-center"):
                                        self.ctrl.above_view_update = (
                                            vega.Figure().update
                                        )
                                    vuetify3.VCardSubtitle("Below")
                                    with html.Div(classes="text-center"):
                                        self.ctrl.below_view_update = (
                                            vega.Figure().update
                                        )

                        with vuetify3.VCol(cols=12, md=5):
                            with vuetify3.VCard(
                                classes="fill-height",
                                style="min-height: max(250px, 40vh)",
                            ):
                                vuetify3.VCardTitle("Heatmap")
                                with html.Div(style="height: calc(100% - 3rem);"):
                                    with trame.SizeObserver("heatmap_size"):
                                        self.ctrl.heatmap_view_update = vega.Figure(
                                            style="width: 100%;"
                                        ).update

                        with vuetify3.VCol(cols=12, md=3):
                            with vuetify3.VCard(classes="fill-height"):
                                vuetify3.VCardTitle("Bottom {{ nb_states }} States")
                                with vuetify3.VCardText():
                                    with vuetify3.VDataTable(
                                        headers=("table_headers", TABLE_HEADER),
                                        items=("bottom_states", []),
                                        density="compact",
                                    ):
                                        with vuetify3.Template(
                                            raw_attrs=[
                                                'v-slot:item.population="{ value }"'
                                            ]
                                        ):
                                            with vuetify3.VProgressLinear(
                                                model_value=("Math.abs(value)",),
                                                color=("value>0 ? 'green':'red'",),
                                                reverse=("value>0 ? false : true",),
                                                height="25",
                                            ):
                                                with vuetify3.Template(
                                                    raw_attrs=[
                                                        'v-slot:default="{ value }"'
                                                    ]
                                                ):
                                                    html.Div(
                                                        "<strong>{{Math.round(value)}}%</strong>"
                                                    )
                                        vuetify3.Template(raw_attrs=["v-slot:bottom"])

            return layout


def main():
    app = UsPopulation(DATA_FILE)
    app.server.start()


if __name__ == "__main__":
    main()
