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


def plot_norm_accept_rate(data_frame: pd.DataFrame = None, population='phd',
                          **kwargs):
    data_frame = regularize_table() if data_frame is None else data_frame

    population_select = (data_frame.index
                         .get_level_values('population')
                         .get_loc(population))
    filt_df = data_frame.loc[population_select]
    filt_df.index = filt_df.index.droplevel('population')

    totals = filt_df.sum()
    totals_df = pd.DataFrame(totals.values.reshape((1, -1)),
                             columns=totals.index,
                             index=pd.Index(['overall']))

    def accept_rate(x: pd.Series):
        return x['num_admit'] / x['num_apply']

    total_accept_rate = accept_rate(totals_df)
    class_df = filt_df.groupby('class').sum()
    class_accept_rate = accept_rate(class_df)
    subclass_mask = np.logical_not(csutils.logical_or(
        *[label == filt_df.index.get_level_values('subclass')
          for label in ['uncategorized', 'unknown', 'international']]
    ))
    subclass_df = filt_df.loc[subclass_mask].groupby('subclass').sum()
    subclass_accept_rate = accept_rate(subclass_df)

    summary_df = pd.concat([total_accept_rate,
                            class_accept_rate,
                            subclass_accept_rate])

    normalized_accept_rate = (summary_df - total_accept_rate.values) * 100

    plot_summary_table(data=summary_df * 100,
                       ylabel='Acceptance Rate (%)',
                       **kwargs)
    plot_summary_table(data=normalized_accept_rate,
                       ylabel='Normalized Acceptance Rate (\Delta%)',
                       **kwargs)


def plot_summary_table(data: pd.DataFrame,
                       ylabel: str,
                       axes: plt.Axes = None,
                       plot_subclasses=True,
                       plot_overall=True,
                       plot_classes=True,
                       plot_unknown=True):

    from matplotlib.ticker import FormatStrFormatter

    if axes is None:
        _new_figure = plt.figure()
        axes = _new_figure.add_subplot(1, 1, 1)

    color_dict = {
        'overall':          'k',
        'international':    '#386cb0',
        'non-urm':          np.array((127, 201, 127)) / 255,
        'urm':              np.array((253, 192, 134)) / 255,
        'unknown':          np.array((190, 174, 212)) / 255,
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

    categories_to_plot = ((classes_only if plot_classes else [])
                          + (subclasses_only if plot_subclasses else [])
                          + (overall_only if plot_overall else [])
                          + (unknown_only if plot_unknown else []))

    for cat in categories_to_plot:
        series: pd.Series = data.loc[cat, :]
        series.dropna(inplace=True)
        plot = axes.plot(series.index, series.values)
        plot: mpl.lines.Line2D = plot[0]
        plot.set_color(color_dict[cat])
        plot.set_marker('o')
        plot.set_linewidth(1.5)
        plot.set_markersize(5)

    csutils.despine(axes, **{k: True for k in ['top', 'right']})
    axes.set_xticks(np.arange(1999, 2020, 10))
    axes.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%04d'))
    axes.set_ylabel(ylabel)
    axes.set_xlabel(data.columns.name)


def set_mpl_defaults():
    font_name = 'Helvetica'
    mpl.rcParams['font.sans-serif'] = font_name


def _main():
    df = regularize_table()
    set_mpl_defaults()
    plot_norm_accept_rate(data_frame=df,
                          plot_subclasses=True,
                          plot_unknown=False)
    plt.show()


if __name__ == "__main__":
    _main()