from setuptools import setup, find_packages

setup(
    name='kinematic_morphing_network',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='An example python package',
    long_description=open('README.md').read(),
    install_requires=['numpy','keras','tensorflow','texttable','matplotlib2tikz','matplotlib','Pillow','pytest'],
    url='https://github.com/etpr/kinematic_morphing_network',
    author='Peter Englert',
    author_email='englertpr@gmail.com',
    include_package_data=True
)