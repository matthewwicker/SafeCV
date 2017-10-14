from distutils.core import setup
setup(
  name = 'SafeCV',
  packages = ['SafeCV'], # this must be the same as the name above
  version = '0.0.2',
  description = 'Algorithms for blackbox falsification of convolutional neural networks',
  author = 'Matthew Wicker',
  author_email = 'mrw64879@uga.edu',
  url = 'https://github.com/matthewwicker/SafeCV', # use the URL to the github repo
  download_url = 'https://github.com/matthewwicker/SafeCV/archive/0.0.2.tar.gz', # I'll explain this in a second
  keywords = ['testing', 'safety',  'deep learning', 'computer vision'], # arbitrary keywords
  classifiers = [],
)
