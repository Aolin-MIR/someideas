from setuptools import setup, find_packages

def _requirements():
    return [name.rstrip() for name in open('requirements.txt').readlines()]

setup(name='ood_gan',
      version='0.0.1',
      description='ood_gan with keras',
      author='Aolin Li',
      install_requires=_requirements(),
      packages=find_packages(),
)
