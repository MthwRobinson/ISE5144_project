from distutils.core import setup
# To upload to PyPi run python setup.py sdist upload -r pypi
setup(
    name='energy_sim',
    version='0.0.1',
    author='Matthew Robinson',
    author_email='robinmw@vt.edu',
    packages=['energy_sim'],
    scripts=[],
    url='https://github.com/MthwRobinson/ISE5144_project',
    license='LICENSE.txt',
    description='software to support energy grid simulations',
    long_description=open('README.txt').read(),
    install_requires=[
        "simpy",
        "numpy",
        "pylab",
        "matplotlib",
        "seaborn",
        "pickle"
        ]
)