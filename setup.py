# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['emgregs']

package_data = \
{'': ['*']}

install_requires = \
['numpy>=1.22.3,<2.0.0',
 'pandas>=1.4.2,<2.0.0',
 'scikit-learn>=1.0.2,<2.0.0',
 'scipy>=1.8.0,<2.0.0',
 'statsmodels>=0.13.2,<0.14.0',
 'tqdm>=4.64.1,<5.0.0']

setup_kwargs = {
    'name': 'emgregs',
    'version': '0.0.6',
    'description': "Python port to Yanxi Li's R emg flare code",
    'long_description': None,
    'author': 'jgori',
    'author_email': 'juliengori@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.11',
}


setup(**setup_kwargs)
