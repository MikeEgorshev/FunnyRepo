from glob import glob

from setuptools import setup

package_name = 'tram_odometry'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mike Egorshev',
    maintainer_email='MikeEgorshev@users.noreply.github.com',
    description='Backup odometry of a tram without GNSS',
    license='TBD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tram_odometry_node = tram_odometry.node:main',
        ],
    },
)
