from setuptools import setup
setup(
    name="deepmdbenchmark",
    install_requires=['numpy', 'py-cpuinfo', 'leancloud', 'nvgpu'],
    packages=['deepmdbenchmark'],
    entry_points={
        'console_scripts': ['deepmdbenchmark = deepmdbenchmark.benchmark:run']
    },
    package_data={
        'deepmdbenchmark': ['*.json','data/type.raw','data/set.*/*.npy'],
    },
)
