# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Physical Learning'
copyright = '2023'
author = 'Jovana Andrejevic'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Options for docstring
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
add_module_names = False

# -- Options for nbsphinx
nbsphinx_execute = 'never'

import os
import sys
#Location of Sphinx files
sys.path.insert(0, os.path.abspath('./../..'))

# to display docs when using C-based packages
import mock
from physical_learning import plot_imports, packing_utils, elastic_utils, allosteric_utils, lammps_utils
 
MOCK_MODULES = ['numpy', 'scipy', 'scipy.integrate', 'scipy.spatial', 'scipy.linalg', 'cmocean', 'networkx',
		'numba', 'pandas', 'poisson_disc', 'skimage', 'sklearn', 'tqdm',
		'matplotlib', 'matplotlib.pyplot', 'matplotlib.collections', 'matplotlib.animation', 'matplotlib.ticker']

REAL_MODULES = ['plot_imports', 'packing_utils', 'elastic_utils', 'allosteric_utils', 'lammps_utils']

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
for mod_name in REAL_MODULES:
	sys.modules[mod_name] = eval(mod_name)

#def setup(app):
    
    # need to assign the names here, otherwise autodoc won't document these classes,
    # and will instead just say 'alias of ...'
#    packing_utils.Packing.__name__ = 'Packing'
#    elastic_utils.Elastic__name__ = 'Elastic'
#    allosteric_utils.Allosteric.__name__ = 'Allosteric'
#    app.connect('build-finished', build_finished)

#def build_finished(app, exception):
#    pass

