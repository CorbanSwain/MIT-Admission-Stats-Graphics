#!/bin/python3
# __init__.py
# Corban Swain, 2019

import c_swain_python_utils as csutils
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines


source_dir = csutils.get_filepath(__file__)
project_dir, _ = os.path.split(source_dir)
log_dir, figure_dir, data_dir = (os.path.join(project_dir, subdir)
                                 for subdir in ['logs',
                                                'figures',
                                                'data'])
logpath = os.path.join(log_dir, 'default_log.log')
csutils.touchdir(log_dir)
L = csutils.get_logger('default', filepath=logpath)


pretty_str = {
    'phd': 'PhD Students',
    'all': 'All Students',
    'non-phd': 'Non-PhD Students',
    'asian': 'Asian American',
    'black': 'Black or\nAfrican-American',
    'white': 'White',
    'native_american': 'Native American or\nAlaskan Native',
    'hispanic': 'Hispanic\nor Latinx',
    'urm': 'URM',
    'non-urm': 'non-URM',
    'pacific_islander': 'Native Hawaiian or\nother Pacific Islander',
    'overall': 'Aggregate',
    'international': 'International',
    'num_apply': 'Applied',
    'num_admit': 'Admitted',
    'num_enroll': 'Enrolled',
    'unknown': 'Unknown',
}


def import_spreadsheet(sheet_name: str = None):
    sheet_name = sheet_name or 'admission_data.csv'
    sheet_path = os.path.join(data_dir, sheet_name)
    df = pd.read_csv(sheet_path)  # FIXME - assuming csv
    return df


def wrangle_data(data_frame: pd.DataFrame = None):
    data_frame = import_spreadsheet() if data_frame is None else data_frame

    # FIXME - `5` should not be hard-coded
    year_begin_index = 5
    ybi = year_begin_index

    years_selection = np.where(data_frame['field'] == 'year')[0].item()
    years = data_frame.values[years_selection][ybi:].astype(int)
    new_data_frame = data_frame.drop(years_selection)
    column_rename_dict = dict(zip(data_frame.columns[ybi:], years))
    new_data_frame = new_data_frame.rename(columns=column_rename_dict)

    drop_columns_indexes = [0, 2, 3, 4]
    drop_columns = new_data_frame.columns[drop_columns_indexes]
    multi_indexes_frame: pd.DataFrame = new_data_frame[drop_columns]
    nan_filter = pd.isna(multi_indexes_frame['category_3'])
    replacement_values = (multi_indexes_frame
                          .loc[nan_filter, 'category_2']
                          .values)
    multi_indexes_frame.loc[nan_filter, 'category_3'] = replacement_values
    multi_indexes = pd.MultiIndex.from_frame(multi_indexes_frame)
    new_data_frame = new_data_frame.drop(columns=drop_columns)

    index_rename_dict = dict(zip(np.arange(new_data_frame.shape[0]) + 1,
                                 multi_indexes))
    new_data_frame = new_data_frame.rename(index=index_rename_dict)
    new_data_frame = new_data_frame.reindex(index=multi_indexes)

    pivot = new_data_frame.pivot(columns='field')
    pivot.columns.set_names(['Year', 'field'], inplace=True)
    pivot = pivot.reorder_levels([1, 0], 'columns')
    pivot.sort_index(axis='columns',
                     level=0,
                     sort_remaining=True,
                     inplace=True)

    pivot.index.set_names(level=range(1, 4),
                          names=['population', 'class', 'subclass'],
                          inplace=True)
    pivot = pivot.reorder_levels([2, 3, 1, 0])
    pivot.sort_index(axis='index',
                     level=0,
                     sort_remaining=True,
                     inplace=True)

    return pivot


def regularize_table(data_frame: pd.DataFrame = None):
    data_frame = wrangle_data() if data_frame is None else data_frame
    grouped = data_frame.groupby(level=list(range(3)))

    def dropna_agg_fun(x: pd.DataFrame):
        output = x.dropna().values
        if output.size == 0:
            return np.NaN
        elif output.size == 1:
            return output[0]
        else:
            if not np.all(output[0] == output[1:]):
                L.warn(f'Non-NaN values are not equal, '
                       f'defaulting to first value.\n\t{output}')
            return output[0]

    new_data_frame = grouped.agg(dropna_agg_fun)

    for select in ['urm', 'non-urm']:
        select_df = new_data_frame.loc[select]
        indexes = np.unique(list(zip(*select_df.index.values))[0])
        other_indexes = indexes[np.logical_not(indexes == select)].tolist()

        select_slice = select_df.loc[[select]]
        other_slice = select_df.loc[other_indexes]

        agg_other = other_slice.groupby('population').sum()
        uncat_values = select_slice.values - agg_other.values
        uncat_columns = agg_other.columns
        uncat_index = pd.MultiIndex.from_tuples(
            [(select, 'uncategorized', 'all'),
             (select, 'uncategorized', 'phd')],
            names=new_data_frame.index.names)
        uncat_df = pd.DataFrame(uncat_values,
                                index=uncat_index,
                                columns=uncat_columns)
        uncat_df.fillna(0, inplace=True)
        uncat_df.where(uncat_df >= 0, 0, inplace=True)
        new_data_frame.drop(index=[(select, select)], inplace=True)
        new_data_frame = new_data_frame.append(uncat_df)

    new_data_frame.sort_index(axis='index',
                              level=0,
                              sort_remaining=True,
                              inplace=True)

    return new_data_frame


def extract_population(data_frame: pd.DataFrame, pop_name):
    _POP_LABEL = 'population'

    if pop_name == 'non-phd':
        return (extract_population(data_frame, 'all')
                - extract_population(data_frame, 'phd'))
    else:
        population_select = (data_frame.index
                             .get_level_values(_POP_LABEL)
                             .get_loc(pop_name))
        filtered_df = data_frame.loc[population_select]
        filtered_df.index = filtered_df.index.droplevel(_POP_LABEL)
        return filtered_df


def extract_overall(data_frame: pd.DataFrame, func=np.sum):
    return pd.DataFrame(data_frame.apply(func).values.reshape((1, -1)),
                        columns=data_frame.columns,
                        index=pd.Index(['overall', ]))


def extract_classes(data_frame: pd.DataFrame, func=np.sum):
    return data_frame.groupby('class').apply(func)


def extract_subclasses(data_frame: pd.DataFrame, func=np.sum):
    subclass_mask = np.logical_not(csutils.logical_or(
        *[label == data_frame.index.get_level_values('subclass')
          for label in ['uncategorized', 'unknown', 'international']]))
    return (data_frame
            .loc[subclass_mask]
            .groupby('subclass')
            .apply(func))


def summarize_process_data(pop_name,
                           process_func,
                           data_frame: pd.DataFrame = None):
    df = regularize_table() if data_frame is None else data_frame
    df = extract_population(df, pop_name)
    return pd.concat([process_func(sub_df) for sub_df in
                      [extract_func(df) for extract_func in
                       [extract_overall,
                        extract_classes,
                        extract_subclasses]]])


def plot_accept_rate(data_frame: pd.DataFrame = None, population='phd',
                     **kwargs):

    def accept_rate(x: pd.Series):
        return x['num_admit'] / x['num_apply']

    summary_df = summarize_process_data(pop_name=population,
                                        process_func=accept_rate,
                                        data_frame=data_frame)

    plot_summary_table(data=summary_df * 100,
                       ylabel='Acceptance Rate, %',
                       legend_title=population,
                       **kwargs)

    accept_rate_delta = ((summary_df - summary_df.loc['overall', :].values)
                         * 100)
    plot_summary_table(data=accept_rate_delta,
                       ylabel=u'Acceptance Rate Difference, \u0394%',
                       legend_title=population,
                       **kwargs)


def plot_yield(data_frame: pd.DataFrame = None, population='phd',
               **kwargs):

    def _yield(x: pd.Series):
        return x['num_enroll'] / x['num_admit']

    summary_df = summarize_process_data(pop_name=population,
                                        process_func=_yield,
                                        data_frame=data_frame)
    summary_df = summary_df * 100

    plot_summary_table(data=summary_df,
                       ylabel='Yield, %',
                       legend_title=population,
                       **kwargs)

    accept_rate_delta = summary_df - summary_df.loc['overall', :].values
    plot_summary_table(data=accept_rate_delta,
                       ylabel=u'Yield Difference, \u0394%',
                       legend_title=population,
                       **kwargs)


def plot_year_to_year_pct_change(data_frame: pd.DataFrame = None,
                                 population='phd',
                                 **kwargs):

    def year_delta(x: pd.DataFrame):
        new_x_values = []
        for i in x.index:
            x_slice = x.loc[i, :]
            nan_filt = x_slice.notna().values
            deltas = np.diff(x_slice.index[nan_filt],
                             prepend=np.NaN)
            new_slice = x_slice.values
            new_slice[nan_filt] = deltas
            new_x_values.append(deltas)
        return pd.DataFrame(data=new_x_values,
                            index=x.index,
                            columns=x.columns)

    def norm_pct_growth(x, value):
        extracted_values = x.loc[:, value]
        pct_change = extracted_values.pct_change(axis='columns') * 100
        _year_delta = year_delta(extracted_values)
        return pct_change / _year_delta

    for value_name in ['num_apply', 'num_admit', 'num_enroll']:
        summary_df = summarize_process_data(
            pop_name=population,
            process_func=lambda x: norm_pct_growth(x, value_name),
            data_frame=data_frame)
        plot_summary_table(data=summary_df,
                           ylabel=(u'Annual Change in \u2116 '
                                   f'{pretty_str[value_name]}, %'),
                           legend_title=population,
                           **kwargs)


def plot_pct_change(data_frame: pd.DataFrame = None,
                    population='phd',
                    **kwargs):

    def pct_change(x, value):
        extracted_values = x.loc[:, value]
        return ((extracted_values
                / extracted_values.loc[:, [1999, ]].values) * 100) - 100

    for value_name in ['num_apply', 'num_admit', 'num_enroll']:
        summary_df = summarize_process_data(
            pop_name=population,
            process_func=lambda x: pct_change(x, value_name),
            data_frame=data_frame)
        plot_summary_table(data=summary_df,
                           ylabel=(u'Change in \u2116 '
                                   f'{pretty_str[value_name]}, %'),
                           legend_title=population,
                           **kwargs)


def plot_raw_numbers(data_frame: pd.DataFrame = None,
                     population='phd',
                     **kwargs):
    def raw(x, value):
        return x.loc[:, value]

    for value_name in ['num_apply', 'num_admit', 'num_enroll']:
        summary_df = summarize_process_data(
            pop_name=population,
            process_func=lambda x: raw(x, value_name),
            data_frame=data_frame)
        plot_summary_table(data=summary_df,
                           ylabel=(u'\u2116 '
                                   f'{pretty_str[value_name]}'),
                           legend_title=population,
                           **kwargs)


def plot_breakdown(data_frame: pd.DataFrame = None,
                   population='phd',
                   **kwargs):
    def raw(x, value):
        return x.loc[:, value]

    for value_name in ['num_apply', 'num_admit', 'num_enroll']:
        summary_df = summarize_process_data(
            pop_name=population,
            process_func=lambda x: raw(x, value_name),
            data_frame=data_frame)
        summary_df = (100
                      * (summary_df / summary_df.loc[['overall', ], :].values))
        plot_summary_table(data=summary_df,
                           ylabel=('Relative % '
                                   f'{pretty_str[value_name]}'),
                           legend_title=population,
                           **kwargs)


def plot_summary_table(data: pd.DataFrame,
                       ylabel: str,
                       axes: plt.Axes = None,
                       plot_subclasses=True,
                       plot_overall=True,
                       plot_classes=True,
                       plot_unknown=True,
                       legend_title: str = None):

    import matplotlib.ticker

    if axes is None:
        _new_figure = plt.figure(figsize=(7, 3.5))
        axes = _new_figure.add_axes([0.13, 0.1, 0.63, 0.8])

    color_dict = {
        'overall':          'k',
        'urm':              '#B35F59',
        'non-urm':          '#33597F',
        'international':    '#B7BF73',
        'unknown':          'grey',
        'asian':            '#2F4A2F',
        'white':            '#578A57',
        'black':            '#7D7268',
        'hispanic':         '#7D5F42',
        'native_american':  '#FDE7D2',
        'pacific_islander': '#C9996B'
    }

    color_dict_keys = list(color_dict.keys())
    overall_only = color_dict_keys[0:1]
    classes_only = color_dict_keys[1:4]
    unknown_only = color_dict_keys[4:5]
    subclasses_only = color_dict_keys[5:]

    categories_to_plot = ([]
                          + (overall_only if plot_overall else [])
                          + (classes_only if plot_classes else [])
                          + (subclasses_only if plot_subclasses else [])
                          + (unknown_only if plot_unknown else []))

    for cat in categories_to_plot:
        series: pd.Series = data.loc[cat, :]
        series.dropna(inplace=True)
        plot = axes.plot(series.index, series.values,
                         label=pretty_str[cat])
        plot: mpl.lines.Line2D = plot[0]
        plot.set_color(color_dict[cat])
        plot.set_marker('o')
        plot.set_linewidth(2.5)
        plot.set_markersize(7)

    csutils.despine(axes, **{k: True for k in ['top', 'right']})
    axes.legend(
        title=(None if legend_title is None
               else (pretty_str[legend_title] + ':')),
        title_fontsize=9.5,
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        fontsize=9,
        frameon=False,
        ncol=1)
    axes.set_xticks(np.arange(1999, 2020, 10))  # FIXME - hardcoded
    axes.xaxis.set_minor_locator(mpl.ticker.IndexLocator(1, 0))
    axes.xaxis.set_major_formatter(
        mpl.ticker.FormatStrFormatter('%04d'))
    axes.set_xlim(1998, 2020)  # FIXME - hardcoded
    axes.spines['bottom'].set_bounds(1999, 2019)  # FIXME - hardcoded
    axes.grid(True, which='major', axis='both')
    axes.set_ylabel(ylabel)


def set_mpl_defaults():
    font_spec = {'family': 'sans-serif',
                 'sans-serif': 'Helvetica',
                 'size': 13}
    mpl.rc('font', **font_spec)

    grid_spec = {'color': 'k',
                 'alpha': 0.1}
    mpl.rc('grid', **grid_spec)


def _main():
    df = regularize_table()

    set_mpl_defaults()

    plot_kwargs = dict(
        data_frame=df,
        plot_subclasses=False,
        plot_unknown=False,
        population='all')
    plot_accept_rate(**plot_kwargs)
    plot_yield(**plot_kwargs)
    plot_pct_change(**plot_kwargs)
    plot_raw_numbers(**plot_kwargs,
                     plot_overall=False)
    plot_breakdown(**plot_kwargs,
                   plot_overall=False)

    csutils.save_figures(directory=figure_dir,
                         filename=plot_kwargs['population'] + '_students')
    plt.show()


if __name__ == "__main__":
    _main()