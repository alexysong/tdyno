from setuptools import setup

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name='tdyno',
    version='0.1.1',
    description='FDTD with dynamic modulations in the refractive index or the gain/loss in materials.',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/alexysong/tdyno',
    # url='http://',
    # check https://pypi.org/pypi?%3Aaction=list_classifiers for classifiers
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
    ],
    keywords=['fdtd', 'dynamic modulation'],
    author='Alex Y. Song',
    author_email='song.alexy@gmail.com',
    packages=['tdyno', 'examples'],
    install_requires=[
        'numpy >= 1.11.3',
        'scipy >= 0.19.0',
        'matplotlib >= 2.0.0'
    ],
    zip_safe=False)
