from setuptools import setup
setup(
    name="deepmdbenchmark",
    install_requires=['numpy', 'py-cpuinfo', 'leancloud'],
    packages=['deepmdbenchmark'],
    entry_points={
        'console_scripts': ['deepmdbenchmark = deepmdbenchmark.benchmark:run',
                            'uploaddpbench = deepmdbenchmark.benchmark:upload_dict',
        ]
    },
    package_data={
        'deepmdbenchmark': ['*.json','data/type.raw','data/set.*/*.npy'],
    },
    version="0.0.4",
)
