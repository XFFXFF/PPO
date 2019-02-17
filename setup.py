from setuptools import setup


setup(
    name='ppo',
    py_modules=['ppo'],
    version='0.1',
    install_requires=[
        'gym[atari,box2d,classic_control]>=0.10.8',
        'joblib',
        'numpy',
        'psutil',
    ],
    author="XFFXFF",
)