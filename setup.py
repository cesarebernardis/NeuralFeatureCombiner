import numpy, os, glob

from setuptools import Command, Extension, setup

from Cython.Build import cythonize
from Cython.Compiler.Options import _directive_defaults

_directive_defaults['linetrace'] = True
_directive_defaults['binding'] = True
compile_args = ['-Wno-unused-function', '-Wno-maybe-uninitialized', '-Wno-sign-compare', '-O3', '-ffast-math', '-fopenmp']
link_args = ['-fopenmp']

# Look for cython files to compile
cython_files = []
for path in glob.glob("RecSysFramework/**/*.pyx", recursive=True):
    cython_files.append((path.replace(".pyx", "").replace(os.sep, "."), path))

modules = [Extension(file[0], [file[1]], language='c++', extra_compile_args=compile_args, extra_link_args=link_args)
           for file in cython_files]
setup(
    name='RecSysFramework',
    version="1.0.0",
    description='Recommender System Framework',
    url='',
    author='Cesare Bernardis',
    author_email='cesare.bernardis@polimi.it',
    install_requires=['numpy',
                      'pandas>=1.1',
                      'scipy>=1.4',
                      'scikit-learn==0.22',
                      'matplotlib>=3.3',
                      'Cython>=0.29',
                      'nltk>=3.2.5',
                      'tensorflow-gpu==2.5',
                      'similaripy>=0.0.11'],
    packages=['RecSysFramework',
              'RecSysFramework.Evaluation',
              'RecSysFramework.DataManager',
              'RecSysFramework.DataManager.Reader',
              'RecSysFramework.DataManager.DatasetPostprocessing',
              'RecSysFramework.DataManager.Splitter',
              'RecSysFramework.Recommender',
              'RecSysFramework.Recommender.GraphBased',
              'RecSysFramework.Recommender.KNN',
              'RecSysFramework.Recommender.MatrixFactorization',
              'RecSysFramework.Recommender.SLIM',
              'RecSysFramework.Recommender.SLIM.BPR',
              'RecSysFramework.Recommender.SLIM.ElasticNet',
              'RecSysFramework.ParameterTuning',
              'RecSysFramework.Utils',
              'RecSysFramework.Utils.Similarity',
              ],
    setup_requires=["Cython >= 0.27"],
    ext_modules=cythonize(modules),
    include_dirs=[numpy.get_include()]
)
