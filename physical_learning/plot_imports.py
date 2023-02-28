import numpy as np
import matplotlib as mpl
import matplotlib.ticker as mticker
import cmocean

def add_alpha(color, alpha):
    '''Add white to produce new color with the same effect as reducing opacity.
       
    Parameters
    ----------
    color : float array, shape (3,) or (4,)
        An RGB or RGBA color on 0 to 1 scale.
    alpha : float, 0 to 1
        The opacity to add.

    Returns
    -------
    color : float array, shape (3,) or (4,)
        The resulting color mixed with white.
    '''
    color = np.array(color)
    color[:3] = alpha*color[:3] + (1-alpha)*np.ones(color[:3].shape)
    return color

# pre-defined color palette
pal = {'red': np.array([238, 99, 82])/255., 'blue': np.array([63, 167, 214])/255.,
       'green': np.array([89, 205, 144])/255., 'yellow': np.array([250, 192, 94])/255.,
       'purple': np.array([106, 76, 224])/255.}

# pre-defined cyclic colormap
cyclic = add_alpha(cmocean.cm.phase(np.linspace(0,1,500)).T, 0.7).T
cyclic_cmap = mpl.colors.ListedColormap(cyclic)

def hex_to_rgb(color):
    '''Convert a hexadecimal color code to RGB.
       
    Parameters
    ----------
    color : string
        A hexadecimal color code beginning with #.
    
    Returns
    -------
    color : tuple of 3 ints
        The resulting RGB color on 0 to 255 scale.
    '''
    color = color.strip("#") # remove the # symbol
    n = len(color)
    return tuple(int(color[i:i+n//3],16) for i in range(0,n,n//3))

def rgb_to_dec(color):
    '''Convert an RGB color on (0,255) scale to (0,1) scale.
       
    Parameters
    ----------
    color : int array
        An RGB color with values on 0 to 255.
    
    Returns
    -------
    color : list
        The resulting RGB color with values on 0 to 1.
    '''
    return [c/255. for c in color]

def continuous_cmap(rgb_colors, float_values):
    '''Create a colormap from a list of colors and corresponding values on (0,1).
       
    Parameters
    ----------
    rgb_colors : float array, shape (n_values, 3)
        An array of RGB colors.
    float_values : float array or list, shape (n_values,)
        List of values on 0 to 1 at which the corresponding color should be placed.
    
    Returns
    -------
    cmap : matplotlib LinearSegmentedColormap object
        A callable colormap object with the specified colors.
    '''
    n = len(rgb_colors) # should be the same as float_values
    cdict = {}
    primary = ['red','green','blue']
    for i,p in enumerate(primary):
        cdict[p] = [[float_values[j], rgb_colors[j][i], rgb_colors[j][i]] for j in range(n)]
        cmap = mpl.colors.LinearSegmentedColormap('my_cmap', segmentdata=cdict, N=500)
    return cmap

def truncate_colormap(cmap, a=0.0, b=1.0, n=100):
    '''Truncate the color range of an existing colormap.
       
    Parameters
    ----------
    cmap : callable, matplotlib colormap
        A callable colormap object.
    a, b : floats, optional, 0 to 1
        Start and end points of the colormap range to keep. Default a=0, b=1
        means no change to the original colormap.
    n : int, optional
        Number of values to sample between a and b. Default 100.
    
    Returns
    -------
    my_cmap : matplotlib LinearSegmentedColormap object
        A callable colormap object with the truncated color range.
    '''
    my_cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',
               cmap(np.linspace(a,b,n)))
    return my_cmap

class MathTextSciFormatter(mticker.Formatter):
    '''Class for formatting strings in Latex-formatted scientific notation, taken from the following Stack Overflow reply:
    https://stackoverflow.com/questions/25750170/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    
    Attributes
    ----------
    fmt : string, optional
        The display format.
    '''
    
    def __init__(self, fmt="%1.2e"):
        '''
        Parameters
        ----------
        fmt : string, optional
            The display format.
        '''
        self.fmt = fmt
    def __call__(self, x):
        '''Formats a value x in the desired format.

        Parameters
        ----------
        x : float
            The value to display.

        Returns
        -------
        s : string
            The formatted string.
        '''
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

# Instance of formatter to 2 decimal places.
fsci = MathTextSciFormatter("%1.2e")