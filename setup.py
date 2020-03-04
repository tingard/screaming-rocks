from setuptools import setup, find_packages

setup(name='screaming-rocks',
      version='0.1',
      description='YEAH ROCKS W00T',
      author='Tim Lingard, David Turner, Connor McIsaac',
      author_email='NaN',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'dask',
          'tqdm',
      ],
      zip_safe=False)
