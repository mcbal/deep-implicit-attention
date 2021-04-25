from setuptools import find_packages, setup

setup(
    name='deep-implicit-attention',
    packages=find_packages(exclude=['images']),
    version='0.0.0',
    license='MIT',
    description='Deep Implicit Attention',
    author='Matthias Bal',
    author_email='matthiascbal@gmail.com',
    url='https://github.com/mcbal/deep-implicit-attention',
    keywords=['artificial intelligence', 'attention mechanism'],
    install_requires=['einops>=0.3', 'numpy>=1.19', 'torch>=1.8'],
)
