from setuptools import setup, find_packages
# import os
# from setuptools import setup, find_packages

#def is_package(path):
#    return (
#        os.path.isdir(path) and
#        os.path.isfile(os.path.join(path, '__init__.py'))
#        )


#def find_packages(path, base=""):
#    """ Find all packages in path """
#    packages = {}
#    for item in os.listdir(path):
#        dir = os.path.join(path, item)
#        if is_package(dir):
#            if base:
#                module_name = "%(base)s.%(item)s" % vars()
#            else:
#                module_name = item
#            packages[module_name] = dir
#            packages.update(find_packages(dir, module_name))
#    return packages

setup(
    name='econtools',
    version='0.1',
    author='Spencer Lyon',
    author_email='spencerlyon2@gmail.com',
    packages=find_packages(),
    url='https://github.com/spencerlyon2/econtools',
    description='Python tools used in economics courses',
    long_description=open('../README.md').read()
)
