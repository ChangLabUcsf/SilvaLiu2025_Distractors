# Imports
import os
import logging
import operator

import numpy as np
import seaborn as sns
from scipy.stats import sem
from sklearn.metrics import confusion_matrix

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.image as mpimg


def default_plot_settings(return_colormap=False, colormap='default',
                          linecolor='404040', fontsize=14, font='Helvetica',
                          linewidth=2, ticklength=6):
    """
    Set default plot settings, like color and fontsize for axes. Quiet the
    matplotlib logger warning.
    """

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    mpl.rcParams.update({'font.size': fontsize})
    mpl.rcParams['font.sans-serif'] = font
    mpl.rcParams['font.family'] = 'sans-serif'

    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

    mpl.rcParams['xtick.major.size'] = ticklength
    mpl.rcParams['xtick.major.width'] = linewidth
    mpl.rcParams['ytick.major.size'] = ticklength
    mpl.rcParams['ytick.major.width'] = linewidth

    mpl.rcParams['xtick.minor.size'] = ticklength // 2
    mpl.rcParams['xtick.minor.width'] = linewidth
    mpl.rcParams['ytick.minor.size'] = ticklength // 2
    mpl.rcParams['ytick.minor.width'] = linewidth

    mpl.rcParams['lines.linewidth'] = linewidth
    mpl.rcParams['axes.linewidth'] = linewidth

    mpl.rcParams['text.color'] = linecolor
    mpl.rcParams['axes.labelcolor'] = linecolor
    mpl.rcParams['axes.edgecolor'] = linecolor
    mpl.rcParams['xtick.color'] = linecolor
    mpl.rcParams['ytick.color'] = linecolor

    if return_colormap:
        if colormap == 'default':
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif colormap == 'custom_set2':
            # Custom version of seaborn's Set2 palette
            pal = sns.color_palette('Set2')
            pal = pal.as_hex()
            colors = list(reversed(pal[:3])) + pal[3:]
        else:
            print(f'The colormap {colormap} is not implemented.')
            colors = None

        return colors
    
def plot_erps(erps,elec,t_ar,ax,colors=sns.color_palette('Set2'),do_axes=False,legends=None):
    for i,erp in enumerate(erps):
        signal = np.mean(erp,axis=0)[:,elec]
        sem_ = sem(erp,axis=0)[:,elec]
        if legends is not None:
            ax.plot(t_ar,signal,color=colors[i],label=legends[i])
        else:
            ax.plot(t_ar,signal,color=colors[i],label='_hidden_')

        ax.fill_between(t_ar,signal + sem_,signal - sem_,color=colors[i],alpha=0.2,label='_hidden_')
    ax.spines[['top','right']].set_visible(False)
    if(do_axes):
        ax.set(xlabel='Time (s)',ylabel='HGA (Z)',ylim=[-0.5,1.5])
    else:
        ax.set(xticks=[],yticks=[])
    return(ax)
    
def plot_scatter(ar1,ar2,ax,alpha=0.8):
    ax.scatter(ar1,ar2,clip_on=False,alpha=alpha)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    sns.despine(ax=ax,offset=5)
    return(ax)


def plot_vals_on_brain(weights,color_map,fig,ax,data_dir,subject='bravo3',size_max=50.,size_min=1,region_highlight='precentral_gyrus',region_alpha=0.15,region_color='#DE3C37',add_mprcg_mask=True):
    cbar_params = {
        'plot_colorbar'        : False,
    }

    elec_size_color_params = {
        'color_spec'   : color_map,
        'color_params' : {'min': 0,  'max': 1,   'relative': True},
        'size_params'  : {'min': size_min, 'max': size_max, 'relative': True, 'scale': size_max-size_min},
    }

    
    all_plot_params = {
        'elec_loc_file_path': f'{data_dir}/{subject}_lateral_elecmat_2D.npy',
        'all_image_params': [
            {
                'file_name': f'{data_dir}/{subject}_lateral_brain_2D.png',
                'invert_y': True,
                'alpha': 0.35
            },
            dict(file_name=f'{data_dir}/{subject}_{region_highlight}_mask.npy', invert_y=True, alpha=region_alpha, only_apply_alpha_to_nonzero=True,
                 mask_pixel_values=('#00000000', region_color)),
        ],
        'y_scale': -1.0,
        'add_height': True,
        'elec_plot_params': {'linewidths': 0.0, 'zorder': 100000.0},
    }
    
    
    all_plot_params['elec_size_color_params'] = elec_size_color_params
    all_plot_params['elec_weights'] = weights

    all_plot_params.update(cbar_params)
    plot_images_and_elecs(**all_plot_params,ax=ax,show_fig=False)
    mprcg_roi = plt.imread(os.path.join(data_dir, f'{subject}_mprcg_roi.png'))
    
    if(add_mprcg_mask):
        ax.imshow(np.flipud(mprcg_roi), alpha=0.75)

    
def adjust_order(arr):
    new_neural_data = np.concatenate([arr[:, :, :107], arr[:, :, 108:112], arr[:, :, 113:117], arr[:, :, 118:]], axis=-1)
    neural_data = np.zeros_like(arr)
    neural_data[:, :, :250] = new_neural_data
    print('made new neural data')
    return(neural_data)

def adjust_elec_inds(arr):
    arr = arr[(arr != 107)]
    arr = arr[(arr != 112)]
    arr = arr[(arr != 117)]   
    arr[(arr>=108) & (arr<=111)] -= 1
    arr[(arr>=113) & (arr<=116)] -= 2
    arr[(arr>=118)] -= 3
    return(arr)


def get_cm(d):
    preds = []
    for p in d['preds']:
        for pp in p:
            preds.extend(pp)

    labs = []
    for p in d['labs']:
        labs.extend(p)
        
    C = confusion_matrix(labs, preds)
    C = C/np.sum(C,axis=1)
    return(C)


class SemiFrozenDict(dict):
    """
    A dictionary sub-class that can be used to create frozen dictionaries.

    An instance of this class cannot have items added to it, and existing keys
    cannot be re-assigned to new values. However, if a value in an instance of
    this class is mutable, it is possible for that value to be modified.

    Note
    ----
    Creating a copy of an instance of this class using the `copy` method will
    return a `dict` object (not another instance of this class).
    """

    def __setitem__(self, key, value):
        """
        Raises a `TypeError` if item assignment is attempted.

        Raises
        ------
        TypeError
            Disables item assignment for instances of this class by raising
            this exception if item assignment is attempted.
        """

        raise TypeError("'{}' object does not support item assignment".format(
            self.__class__.__name__
        ))

def empty_dict():
    """
    Returns an empty `SemiFrozenDict`.

    This function is useful for creating an empty dictionary default argument
    value for other functions.

    Returns
    -------
    SemiFrozenDict
        An empty dictionary instance.
    """

    return SemiFrozenDict(())

class Bunch(dict):
    """
    A dictionary sub-class that replicates keys as attributes.

    In addition to normal dictionary functionality, this class also allows data
    to be read from and written to instances as attributes. Any attempts to
    read/write attributes are handled internally as dictionary operations
    (except for valid dictionary attribute reads).

    For example, given a Bunch object named b with a key named "abc" within
    the dictionary, the following expression would evaluate to True:
    b.abc == b['abc'] == getattr(b, 'abc')
    """

    def __getattr__(self, name):
        try:
            return dict.__getitem__(self, name)
        except KeyError:
            try:
                return getattr(self, name)
            except AttributeError:
                print(
                    "Attribute or key '{}' not found.".format(name))

    def __setattr__(self, key, value):
        if hasattr(self, key):
            print(
                "Attribute '{}' is read-only.".format(key))
        else:
            return dict.__setitem__(self, key, value)

    def __delattr__(self, name):
        try:
            return dict.__delitem__(self, name)
        except KeyError:
            print(
                "Cannot attribute or key '{}'.".format(name))

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, dict.__repr__(self))


# Freesurfer anatomical RGB colors
# fs_colors = {
#     'caudalmiddlefrontal': (100, 25, 0), 
#     'middletemporal': (160, 100, 50), 
#     'parsopercularis': (220, 180, 140),
#     'postcentral': (220, 20, 20), 
#     'precentral': (60, 20, 220), 
#     'superiortemporal': (140, 220, 220), 
#     'supramarginal': (80, 160, 20),
#     'superiorfrontal': (20, 220, 160)
# }
fs_colors = {
    'caudalmiddlefrontal' : (0.39215686274509803, 0.09803921568627451, 0.0),
    'middletemporal' : (0.6274509803921569, 0.39215686274509803, 0.19607843137254902),
    'parsopercularis' : (0.8627450980392157, 0.7058823529411765, 0.5490196078431373),
    'postcentral' : (0.8627450980392157, 0.0784313725490196, 0.0784313725490196),
    'precentral' : (0.23529411764705882, 0.0784313725490196, 0.8627450980392157),
    'superiortemporal' : (0.5490196078431373, 0.8627450980392157, 0.8627450980392157),
    'supramarginal' : (0.3137254901960784, 0.6274509803921569, 0.0784313725490196),
    'superiorfrontal': (0.07843137, 0.8627451 , 0.62745098)
}


fs_colors = {
    'caudalmiddlefrontal' : (0.39215686274509803, 0.09803921568627451, 0.0),
    'middletemporal' : (0.6274509803921569, 0.39215686274509803, 0.19607843137254902),
    'parsopercularis' : (0.8627450980392157, 0.7058823529411765, 0.5490196078431373),
    'postcentral' : (0.8627450980392157, 0.0784313725490196, 0.0784313725490196),
    'precentral' : (0.23529411764705882, 0.0784313725490196, 0.8627450980392157),
    'superiortemporal' : (0.5490196078431373, 0.8627450980392157, 0.8627450980392157),
    'supramarginal' : (0.3137254901960784, 0.6274509803921569, 0.0784313725490196),
    'superiorfrontal': (0.07843137, 0.8627451 , 0.62745098)
}


def setup_figure(all_panel_params, row_specs, col_specs,
                 panel_label_params=None, **kwargs):
    """
    Creates and sets up a figure with multiple panels (subplots).

    The axes will be created using the `createMultiPanelFigure` function.

    Parameters
    ----------
    all_panel_params : dict
        A dictionary in which each key is a string representing the name of
        the current panel (subplot) and each value is a dictionary specifying
        setup parameters for that panel. The length of this iterable should
        be equal to the number of plots (panels) to generate. Each dictionary
        can contain the following items:
        - "row_and_col_spec" : list or tuple
            A 2-element list or tuple that will be used as the value for the
            appropriate item in the dictionary that will be passed to the
            `createMultiPanelFigure` function. This item is required.
        - "panel_label" : str or None
            A string label for the current panel. If this is `None`, no label
            will be used for that panel.
        - "panel_label_params" : dict or None
            A value to use for the current panel instead of the
            `panel_label_params`  argument value (see the description for that
            parameter).
    row_specs : dict
        The row specifications to pass to the `createMultiPanelFigure`
        function. See the documentation for that function for more details.
    col_specs : dict
        The column specifications to pass to the `createMultiPanelFigure`
        function. See the documentation for that function for more details.
    panel_label_params : dict or None
        A dictionary containing the keyword arguments to pass to the
        `annotate` method of each panel `AxesSubplot` object. This value can be
        overridden using the dictionaries within the plot parameters. If this
        is `None` (for any panel), this annotation will not be added.
    **kwargs
        Additional keyword arguments to accept and pass to the `create_figure`
        function, which will then be passed to the `pyplot.figure` function
        during creation of the new `Figure` instance.

    Returns
    -------
    Figure
        The `Figure` instance.
    Bunch
        A `Bunch` instance (which is an instance of `dict` that inherits most
        of its functionality) in which each key is a string specifying the name
        of the current panel and each value is the associated `AxesSubplot`
        instance. The set of keys in this returned value should equal the
        set of keys in the `all_panel_params` dictionary.
    """

    # Creates the figure and axes
    fig, axs = create_figure(
        panel_specs={cur_name: cur_params['row_and_col_spec']
                     for cur_name, cur_params in all_panel_params.items()},
        row_specs=row_specs, col_specs=col_specs, **kwargs
    )

    # Iterates through each set of panel parameters and adds any desired
    # panel labels
    for cur_panel_name, cur_panel_params in all_panel_params.items():

        cur_panel_label = cur_panel_params.get('panel_label')
        cur_panel_label_params = cur_panel_params.get(
            'panel_label_params', panel_label_params
        )

        if cur_panel_label is not None and cur_panel_label_params is not None:
            cur_label_kwargs = dict(s=cur_panel_label, annotation_clip=False)
            cur_label_kwargs.update(cur_panel_label_params)
            axs[cur_panel_name].annotate(**cur_label_kwargs)

    # Returns the new figure and axes `Bunch`
    return (fig, axs)

def create_figure(panel_specs, row_specs, col_specs, **kwargs):
    """
    Creates a figure with multiple panels (subplots).

    This function creates multiple panels using a `matplotlib` `GridSpec`
    object.

    Parameters
    ----------
    panel_specs : dict
        A dictionary in which each key is a string representing the name of
        the current panel (subplot) and each value is a 2-element tuple. Each
        of these sub-tuples should contain 2 string elements. These two
        elements should be the row and column names, respectively, for the
        current panel within the `row_specs` and `col_specs` arguments. The
        length of this argument should be equal to the desired number of
        panels.

        For example, if an element of this dictionary is `(aaa, bbb)`, then the
        `GridSpec` range assigned to the current panel (subplot) will range
        from row `aaa_start` to row `aaa_stop` and from column `bbb_start` to
        column `bbb_stop`, where all four of these row/column names should be
        elements in the respective `row_specs` and `col_specs` dictionaries.
    row_specs : dict
        A dictionary in which each key is a row specification (which should end
        in either `_start` or `_stop`), and each value is the integer
        `GridSpec` location. This dictionary should also contain a `total` key
        associated with an integer value that specifies the total number of
        rows to allot to the `GridSpec`.
    col_specs : dict
        A dictionary in which each key is a column specification (which should
        end in either `_start` or `_stop`), and each value is the integer
        `GridSpec` location. This dictionary should also contain a `total` key
        associated with an integer value that specifies the total number of
        columns to allot to the `GridSpec`.
    **kwargs
        Additional keyword arguments to pass to the `pyplot.figure` function
        during creation of the new `Figure` instance.

    Returns
    -------
    Figure
        The `Figure` instance.
    Bunch
        A `Bunch` instance (which is an instance of `dict` that inherits most
        of its functionality) in which each key is a string specifying the name
        of the current panel and each value is the associated `AxesSubplot`
        instance. The set of keys in this returned value should equal the
        set of keys in the `all_panel_params` dictionary.
    """

    # Sets up the figure, the axes `Bunch`, and the `GridSpec` instance
    fig = plt.figure(**kwargs)
    axs = Bunch()
    gs = matplotlib.gridspec.GridSpec(row_specs['total'], col_specs['total'])

    # Adds the panels to the figure and to the axes `Bunch`
    for cur_panel_name, (cur_row, cur_col) in panel_specs.items():
        axs[cur_panel_name] = fig.add_subplot(
            gs[row_specs[cur_row + '_start'] : row_specs[cur_row + '_stop'],
               col_specs[cur_col + '_start'] : col_specs[cur_col + '_stop']]
        )

    # Returns the figure and the axes `Bunch`
    return (fig, axs)

def plot_images_and_elecs(
        elec_loc_file_path, all_image_params, elec_size_color_params,
        elec_weights=None, plot_colorbar=False, x_shift=0., y_shift=0.,
        x_scale=1., y_scale=1., add_width=False, add_height=False,
        elec_plot_params=empty_dict(),
        figure_params=empty_dict(), apply_formatting=True,
        colorbar_title='', colorbar_orientation='horizontal',
        fig=None, ax=None, ax_cbar=None, show_fig=True,
        save_fig_kwargs=empty_dict(), compute_color_size=True, cur_sizes=None, cur_colors=None
):
    """
    Plots one or more images with overlaid electrode information.

    Parameters
    ----------
    elec_loc_file_path : str
        The file path at which the electrode coordinate locations are stored.
        This file should be loadable using `numpy.load`, and should contain a
        float array of shape `(num_electrodes, 2)` containing the X-Y
        coordinates for each electrode. These coordinates should correspond to
        the pixels contained in the images.
    all_image_params : list or tuple or dict
        A list or tuple in which each element is a dictionary containing
        keyword arguments to pass to the `load_asset_image` function to load
        the desired image data. If this is a dictionary instance instead, it is
        treated as if it were a 1-element list containing the provided
        dictionary.
    elec_size_color_params : dict
        A dictionary of keyword arguments to pass to the
        `get_elec_sizes_and_colors` function to load the electrode size and
        color information for the plot. The `weights` keyword argument for that
        function will be determined from the `elec_weights` argument to this
        function, overridding the value from this dictionary if one was
        provided.
    elec_weights : ndarray or None
        Electrode weights, which should be a float array of shape
        `(num_electrodes,)`. If this is `None`, an array of the appropriate
        shape will be created (informed by the loaded electrode coordinate
        locations) containing flat (equal) values.
    plot_colorbar : bool
        Specifies whether or not to include a colorbar plot. If the value
        mapped to the `"color_spec"` key in `elec_size_color_params` is not a
        valid colormap or a string specifying a colormap, then a colorbar will
        not be plotted (but, if this argument is `True`, an empty space in the
        figure will still be allocated to where the colorbar would have been).
        If "relative" colors are specified in  `elec_size_color_params`, then
        the full colormap will be plotted. Otherwise, a subset of the colormap
        from `max(0, min(elec_weights))` to `min(1, max(elec_weights))` will
        be plotted.
    x_shift : float
        The shift to apply (add) to each electrode X coordinate.
    y_shift : float
        The shift to apply (add) to each electrode Y coordinate.
    x_scale : float
        The scale to apply (multiply) to each electrode X coordinate.
    y_scale : float
        The scale to apply (multiply) to each electrode Y coordinate.
    add_width : bool
        Specifies whether or not to add the width of the first image to the
        electrode X coordinates, which can be useful if using an `x_scale` of
        `-1` to invert the plotted electrode X locations. If this evaluates to
        `True`, this transformation will be applied after applying the scale
        and shift.
    add_height : bool
        Specifies whether or not to add the height of the first image to the
        electrode Y coordinates, which can be useful if using an `y_scale` of
        `-1` to invert the plotted electrode Y locations. If this evaluates to
        `True`, this transformation will be applied after applying the scale
        and shift.
    elec_plot_params : dict
        Keyword arguments to pass to the `scatter` method used to plot the
        electrodes in the image. The `x`, `y`, `s`, and `color` keyword
        arguments are specified within this function, but they will be
        overridden by items in this dictionary if there is a conflict.
    figure_params : dict
        Keyword arguments to pass to the `pyplot.figure` function when creating
        the figure. This is only used if `fig` is `None`.
    apply_formatting : bool
        Specifies whether or not to format the plot by removing the axis
        borders and restricting the size to the image dimensions.
    colorbar_title : str
        The title to apply to the colorbar. This is only used if
        `plot_colorbar` evaluates to `True`.
    colorbar_orientation : str
        The orientation of the colorbar. This is only used if `plot_colorbar`
        evaluates to `True`.
    fig : Figure or None
        The `Figure` instance to use. If this is `None`, a new figure will be
        created.
    ax : AxesSubplot or None
        The `AxesSubplot` instance to use to plot the images and electrodes.
        If this is `None`, a new subplot will be created.
    ax_cbar : AxesSubplot or None
        The `AxesSubplot` instance to use to plot color bar (if one is
        desired). If this is `None`, a new subplot will be created for the
        colorbar (if desired).
    show_fig : bool
        Specifies whether or not to show the figure after plotting.
    save_figs_kwargs : dict
        Keyword arguments to pass to the `pyplot.savefig` function to save the
        generated figure. This is only used if `show_fig` evaluates to `True`
        and this argument contains at least one item.
    """

    # Converts a single image parameter dictionary into a 1-element list
    # containing the dictionary as the only element
    if isinstance(all_image_params, dict):
        all_image_params = [all_image_params]

    # Loads all of the image data
    all_image_data = [
        load_asset_image(**cur_params) for cur_params in all_image_params
    ]

    # Loads the electrode location coordinates and applies the desired
    # transformations
    elec_locs = np.load(elec_loc_file_path)#assets.get_asset_file_path(elec_loc_file_path))

    image_height, image_width = all_image_data[0].shape[:2]
    elec_locs[:, 0] = (elec_locs[:, 0] * x_scale) + x_shift
    elec_locs[:, 1] = (elec_locs[:, 1] * y_scale) + y_shift

    if add_width:
        elec_locs[:, 0] += image_width
    if add_height:
        elec_locs[:, 1] += image_height

    # Creates the figure (if one was not provided)
    if fig is None:
        fig = plt.figure(**figure_params)

    # Creates the image axis (if one was not provided) and, if desired, the
    # colorbar axis (if one was not provided)
    gridspec = fig.add_gridspec(20, 1)

    if ax is None:
        if plot_colorbar:
            ax = fig.add_subplot(gridspec[:18, :])
        else:
            ax = fig.add_subplot(111)

    if plot_colorbar and ax_cbar is None:
        ax_cbar = fig.add_subplot(gridspec[-1, :])

    # Plots the images
    for cur_image_data in all_image_data:
        ax.imshow(cur_image_data)

    # If weights were not provided, default flat weights are used
    if elec_weights is None:
        elec_weights = np.ones(elec_locs.shape[0], dtype=float)
        elec_weights /= np.sum(elec_weights)

    # Computes the electrode sizes and colors
    
    # Alex adding way to automatically pass in colors
    if(compute_color_size):
        cur_sizes, cur_colors, cur_elec_params = get_elec_sizes_and_colors(
            weights=elec_weights, return_elec_params=True, **elec_size_color_params
        )
        
    

    # Plots the electrodes
    actual_elec_plot_params = dict(
        x=elec_locs[:, 0], y=elec_locs[:, 1], s=cur_sizes, color=cur_colors,
    )
    actual_elec_plot_params.update(elec_plot_params)
    ax.scatter(**actual_elec_plot_params)

    # Formats the plot (if desired)
    if apply_formatting:
        ax.axis('off')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)

    # Adds a colorbar (if appropriate)
    if plot_colorbar and compute_color_size:
        try:
            cmap = elec_size_color_params['color_spec']
            if not isinstance(cmap, mpl.colors.Colormap):
                cmap = plt.get_cmap(cmap)
        except ValueError:
            ax_cbar.axis('off')
        else:

            min_val = np.min(cur_elec_params['colors'])
            max_val = np.max(cur_elec_params['colors'])

            if not elec_size_color_params.get('color_params', {}).get(
                    'relative', False
            ):
                min_val = max(0., min_val)
                max_val = min(1., max_val)

                cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    name='subset_cmap',
                    colors=cmap(np.linspace(min_val, max_val, 1000))
                )

            colorbar = mpl.colorbar.ColorbarBase(
                ax=ax_cbar, cmap=cmap, orientation=colorbar_orientation
            )
            colorbar.set_label(colorbar_title)

            colorbar.set_ticks(np.linspace(0., 1., 5, endpoint=True))
            colorbar.set_ticklabels(
                ['{:g}'.format(i) for i in
                 np.linspace(min_val, max_val, 5, endpoint=True)]
            )

    # Shows and saves the figure (as desired)
    if show_fig:
        plt.show()

        if len(save_fig_kwargs):
            plt.savefig(**save_fig_kwargs)

def load_asset_image(file_name, invert_y=False, alpha=None,
                     only_apply_alpha_to_nonzero=False,
                     mask_pixel_values=None):
    """
    Loads, processes, and returns pixel data associated with an asset image.

    Parameters
    ----------
    file_name : str
        The asset file name (the name of a file within the package's `assets`
        directory).
    invert_y : bool
        Specifies whether or not to invert the image along the y-axis, which
        can be useful when preparing the image data for a plot. Note that this
        will invert the first dimension of the pixel data.
    alpha : float or None
        Either a float in the range `[0, 1]` specifying the alpha value to
        apply to the pixels or `None`. If this is `None`, the image alpha
        values are not modified.
    only_apply_alpha_to_nonzero : bool
        Specifies whether or not to apply the alpha values to pixels that have
        a non-zero alpha value.
    mask_pixel_values : array-like or None
        If this is not `None`, it is assumed that the specified file is a
        `numpy` data file containing a 2-D boolean array. This array specifies
        which pixels are "inactive" and which are "active". A float array of
        shape `(x.shape[0], x.shape[1], 4)` will be created to represent the
        pixel data, where `x` is the boolean array loaded from the file. This
        new float array will contain the pixel data for the image. The two
        elements in this argument value should be the data for each "inactive"
        pixel and the data for each "active" pixel (in that order). The
        `matplotlib.colors.to_rgba` function will be used to convert the
        value of each element to the RGBA pixel values. If this argument value
        is `None`, it is assumed that the specified file is an image file, and
        the `matplotlib.image.imread` function is used to load the pixel data.

    Returns
    -------
    ndarray
        An array of shape `(height, width, 4)` containing RGBA values for each
        pixel of the image.
    """

    # If mask pixel values were not provided, the image is loaded directly
    if mask_pixel_values is None:
        image_data = mpimg.imread(file_name)

    # Otherwise, additional steps are taken to compute the image pixel values
    else:

        # Extracts the "active" and "inactive" pixel colors
        inactive_pixel_color, active_pixel_color = [
            mpl.colors.to_rgba(c) for c in mask_pixel_values
        ]

        # Loads the pixel boolean mask
        image_mask = np.load(file_name)

        # Creates the image pixel array
        image_data = np.empty(shape=(image_mask.shape + (4,)), dtype=float)
        image_data[~image_mask] = inactive_pixel_color
        image_data[image_mask]  = active_pixel_color

    # Inverts the y-axis (if desired)
    if invert_y:
        image_data = image_data[::-1, :, :]

    # If the desired alpha value is non-zero, additional steps are taken
    if alpha is not None:

        # If desired, the alpha value will only be applied to pixels with
        # non-zero alpha values
        if only_apply_alpha_to_nonzero:
            nonzero_inds = np.nonzero(image_data[:, :, -1])
            target_alpha_inds = nonzero_inds + (
                np.full(shape=nonzero_inds[0].shape, dtype=int, fill_value=-1),
            )
            image_data[target_alpha_inds] = alpha

        # Otherwise, the alpha value is applied to all pixels
        else:
            image_data[:, :, -1] = alpha

    # Returns the image data
    return image_data

def get_elec_sizes_and_colors(
        weights, color_spec, size_params=empty_dict(),
        color_params=empty_dict(),
        alpha_params=empty_dict(), return_elec_params=False
):
    """
    Computes electrode sizes and RGBA colors from weights values.

    Electrode sizes, colors, and alpha values are controlled by the weights
    and function parameters. These values are processed in the following order:
    1) Compute initial values from the weights (direct or relative).
    2) Shift the values.
    3) Scale the values.
    4) Exponentiate the values.
    5) Clip the values.

    If a colormap is specified, colors will be scaled such
    Parameters
    ----------
    weights : array-like
        A 1D array of shape `(num_electrodes,)` containing electrode weights
        (as floats). All values should lie in the range of `[0, 1]`.
    color_spec : Colormap or str or object
        Either an object recognizable by `matplotlib` as a color or a color
        map (or string specifying a color map). If it is a single color,
        it will be used for all electrodes. Otherwise, the provided weights
        will be used to obtain the color from the color map. These colors may
        still be altered using new alpha values.
    size_params : dict
        A dictionary that may contain any of the following items (the presence
        of one of these items in this dictionary argument will override the
        corresponding default value):
        - "min" : float
            The minimum electrode size. Default is `0.0`.
        - "max" : float
            The maximum electrode size. Default is `+inf`.
        - "relative" : bool
            Specifies whether or not to scale the sizes relative to the range
            of `weight` values (which would cause the electrode with the
            smallest weight to have a size of `0.0` and the electrode with the
            largest weight to have a size of `1.0`). Otherwise, the weights
            will be used as the sizes. All sizes are still shifted, scaled,
            exponentiated, and clipped before they are returned. Default is
            `True`.
        - "shift" : float
            All electrode sizes will be shifted (incremented) by this value.
            Default is `0.0`.
        - "scale" : float
            All electrode sizes will be scaled (multiplied) by this value.
            Default is `1.0`.
        - "exponent" : float
            All electrode sizes will be exponentiated by this value. Default is
            `1.0`.
    color_params : dict
        A dictionary that resembles the `size_params` dictionary except that it
        modifies the color values. Items in this dictionary will only be used
        if a color map was specified in `color_spec` (and not just a single
        color).
    alpha_params : dict
        A dictionary that resembles the `size_params` dictionary except that it
        modifies the alpha values.
    return_elec_params : bool
        Specifies whether or not to return the electrode-parameter dictionary.
        This dictionary contains the adjusted electrode-weight values for the
        sizes, colors, and alphas.

    Returns
    -------
    sizes : array-like
        A 1-D float array of shape `(num_electrodes,)` containing electrode
        sizes.
    colors : array-like
        A 2-D float array of shape `(num_electrodes, 4)` electrode RGBA values.
    elec_params : dict (optional)
        A dictionary with keys `"sizes"`, `"colors"`, and `"alphas"` and values
        of 1-D float arrays of shape `(num_electrodes,)` containing the
        adjusted electrode-weight values for the associated key. This
        dictionary is only returned if `return_elec_params` evaluates to
        `True`.
    """

    # Defines a local dictionary from which default values will be loaded for
    # any missing parameters
    _defaults = {
        "min"      : 0.,
        "max"      : float('inf'),
        "relative" : True,
        "shift"    : 0.,
        "scale"    : 1.,
        "exponent" : 1.
    }

    # Defines a local function to use to obtain parameter values
    def _get_spec(_cur_params, _cur_spec):
        return _cur_params.get(_cur_spec, _defaults[_cur_spec])

    # Determines the number of electrodes
    num_electrodes = len(weights)

    # Computes the minimum and maximum weight and the relative weights
    weights = np.asarray(weights)
    if np.all(weights == weights[0]):
        relative_weights = weights / np.sum(weights)
    else:
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        relative_weights = (weights - min_weight) / (max_weight - min_weight)

    # Iterates through the electrode sizes, colors, and alpha values
    elec_params = {}
    for cur_key, cur_params in zip(
            ('sizes', 'colors', 'alphas'),
            (size_params, color_params, alpha_params)
    ):

        # Obtains either the relative weights or original weights
        if _get_spec(cur_params, 'relative'):
            cur_vals = relative_weights.copy()
        else:
            cur_vals = weights.copy()

        # Applies operations to the values
        cur_vals = np.clip(
            (((cur_vals + _get_spec(cur_params, 'shift'))
              * _get_spec(cur_params, 'scale'))
             ** _get_spec(cur_params, 'exponent')),
            _get_spec(cur_params, 'min'), _get_spec(cur_params, 'max')
        )

        # Stores the current values
        elec_params[cur_key] = cur_vals

    # Initializes the color array, either using a static color or a colormap
    # (depending on what was provided)
    try:
        colors = mpl.colors.to_rgba_array([color_spec] * num_electrodes)
    except ValueError:
        if not isinstance(color_spec, mpl.colors.Colormap):
            color_spec = plt.get_cmap(color_spec)
        colors = color_spec(elec_params['colors'])

    # Adds the alpha values to the color array
    colors[:, -1] = elec_params['alphas']

    # Returns the sizes and colors (and, if desired, the electrode parameters)
    if return_elec_params:
        return (elec_params['sizes'], colors, elec_params)
    else:
        return (elec_params['sizes'], colors)

def format_values(values, label_format='{}'):
    """
    Creates LaTeX-friendly label strings.

    Parameters
    ----------
    values : array-like
        An array-like object containing the values to format.
    label_format : str
        A string that can be formatted with a value within the `values`
        argument.

    Returns
    -------
    list
        A list of strings containing the formatted labels.
    """

    # Computes the signs and magnitudes of the provided values
    signs      = np.sign(values)
    magnitudes = np.abs(values)

    # Computes and returns the labels, with special handling of negative values
    return [
        ('{}' + label_format).format(
            '\\textminus' if cur_sign == -1 else '', cur_magnitude
        ) for cur_magnitude, cur_sign in zip(magnitudes, signs)
    ]

def create_linear_colormap(start_color, end_color, name='Unnamed', **kwargs):
    """
    Creates a custom linear colormap.

    Parameters
    ----------
    start_color : object
        The colormap start color (associated with value `0.0`). This should be
        a valid color specification for use with `matplotlib`.
    end_color : object
        The colormap end color (associated with value `1.0`). This should be
        a valid color specification for use with `matplotlib`.
    name : str
        The colormap name.
    **kwargs
        Additional keyword arguments are accepted and passed to the
        `LinearSegmentedColormap`. If a `segmentdata` argument is present, it
        will override the segment data computed within this function.

    Returns
    -------
    LinearSegmentedColormap
        A colormap instance.
    """

    # Ensures that the provided colors are represented as RGB specifications
    start_color_rgb = mpl.colors.to_rgb(start_color)
    end_color_rgb   = mpl.colors.to_rgb(end_color)

    # Creates the segment data
    segmentdata = {
        cur_color_channel : [
            (x, cur_color[i], cur_color[i])
            for x, cur_color in zip((0., 1.), (start_color_rgb, end_color_rgb))
        ]
        for i, cur_color_channel in enumerate(('red', 'green', 'blue'))
    }

    # Creates and returns the colormap
    final_kwargs = dict(name=name, segmentdata=segmentdata)
    final_kwargs.update(kwargs)
    return mpl.colors.LinearSegmentedColormap(**final_kwargs)

def create_skewed_colormap(base_cmap, val_mapping, interpolation=0,
                           name='Skewed colormap', **kwargs):
    """
    Creates a skewed version of a specified colormap.

    Parameters
    ----------
    base_cmap : object
        Either a `Colormap` instance a string that can be used with the
        `pyplot.get_cmap` function to obtain the desired `Colormap` instance.
    val_mapping : list
        A list of 2-element tuples containing floats. The color specified by
        the base colormap at the first float in each tuple will be the color
        specified by the new colormap at the second float in that tuple.
    interpolation : int
        The number of data points to interpolate between the values in the
        value mapping. For example, if `val_mapping` was
        `((0., 0.), (0.4, 0.6), (1., 1.))` and `interpolation` was `1`, then
        the updated `val_mapping` value would be:
        `((0., 0.), (0.2, 0.3), (0.4, 0.6), (0.7, 0.8), (1., 1.))`.
    name : str
        The colormap name.
    **kwargs
        Additional keyword arguments are accepted and passed to the
        `LinearSegmentedColormap`. If a `segmentdata` argument is present, it
        will override the segment data computed within this function.

    Returns
    -------
    `LinearSegmentedColormap`
        The new skewed colormap.

    Raises
    ------
    ValueError
        If an invalid `val_mapping` argument value was provided.
    """

    # Converts the base colormap specification from a string to a colormap
    # instance (if necessary)
    if isinstance(base_cmap, str):
        base_cmap = plt.get_cmap(base_cmap)

    # Creates a list copy of the value mapping
    val_mapping = list(val_mapping)

    # Raises an exception if `val_mapping` has an invalid value
    if len(val_mapping) == 0:
        raise ValueError('The provided value mapping list was empty.')
    for i, (old_val, new_val) in enumerate(val_mapping):
        for cur_val in (old_val, new_val):
            if not (0. <= cur_val <= 1.):
               raise ValueError(
                   ('All value mapping elements should be in the range '
                    '[0, 1], but a value at index {} was {}.').format(
                       i, cur_val
                   )
               )

    # Defines the colors
    colors = ('red', 'green', 'blue')

    # If either endpoint value for the new colormap was not specified in the
    # value mapping, the value mapping is updated to contain a mapping using
    # the same endpoint value in both colormaps
    for endpoint in (0., 1.):
        if not any(endpoint == i[1] for i in val_mapping):
            val_mapping.append((endpoint, endpoint))

    # Sorts the value mapping
    val_mapping = sorted(val_mapping, key=operator.itemgetter(0))

    # Performs interpolation (if desired)
    if interpolation != 0:

        # Initializes the new value mapping list and determines the total
        # number of points that will be present for each interpolation segment
        # (including the endpoints of each segment)
        new_val_mapping = []
        num_points = interpolation + 2

        # Loops through each value mapping (except for the first one)
        for i in range(1, len(val_mapping)):

            # Gets the current and previous value mappings
            cur_mapping  = val_mapping[i]
            prev_mapping = val_mapping[i-1]

            # Creates a linear interpolation vector between the mappings
            old_vals = np.linspace(prev_mapping[0], cur_mapping[0], num_points)
            new_vals = np.linspace(prev_mapping[1], cur_mapping[1], num_points)

            # Adds the new value mappings to the new value mapping list (if
            # the list does not already contain this value)
            for old_val, new_val in zip(old_vals, new_vals):
                if (old_val, new_val) not in new_val_mapping:
                    new_val_mapping.append((old_val, new_val))

        # Stores the new value mapping as the overall value mapping
        val_mapping = new_val_mapping

    # Initializes the color dictionary
    color_dict = {c: [] for c in colors}

    # Loops through the value mappings
    for old_val, new_val in val_mapping:

        # Finds the color associated with the old value
        old_color = base_cmap(old_val)

        # Stores the color at the new value in the new colormap
        for i, cur_color in enumerate(colors):
            color_dict[cur_color].append((new_val, old_color[i], old_color[i]))

    # Sorts the color dictionary so that the value specifications are in
    # ascending order
    color_dict = {
        k: sorted(v, key=operator.itemgetter(0)) for k, v in color_dict.items()
    }

    # Creates and returns the skewed colormap
    final_kwargs = dict(name=name, segmentdata=color_dict)
    final_kwargs.update(kwargs)
    return mpl.colors.LinearSegmentedColormap(**final_kwargs)
