import codecs, glob, os, sys, re
from setuptools import setup, find_packages, Extension
from distutils import log

from setuptools.command.install import install as _install

add_pkg = ['arkjit']
# requirements = ['arkouda']
requirements = []
setup_requirements = ['wheel']

here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# https://packaging.python.org/guides/single-sourcing-package-version/
def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='arkjit',
    version=find_version('client', 'arkjit', '_version.py'),
    description='Numba-based JIT for Arkouda',
    long_description=long_description,

    url='https://bears-r-us.github.io/arkouda/',

    # Author details
    author='Wim Lavrijsen',
    author_email='WLavrijsen@lbl.gov',

    license='LBNL BSD',

    classifiers=[
        'Development Status :: 1 - Production/Stable',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3',

        'Natural Language :: English'
    ],

    setup_requires=setup_requirements,
    install_requires=requirements,

    keywords='HPC workflow exploratory analysis parallel distribute arrays Chapel',

    package_dir={'': 'client'},
    packages=find_packages('python', include=add_pkg),

    zip_safe=False,
)
