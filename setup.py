from setuptools import setup
from setuptools import find_packages


setup(name="maskflow",
      version='0.1.2',
      author='Hadrien Mary',
      author_email='hadrien.mary@gmail.com',
      url='https://github.com/hadim/maskflow/',
      description='Object Detection and Segmentation for Cell Biology.',
      include_package_data=True,
      packages=find_packages(),
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Natural Language :: English',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3'])
