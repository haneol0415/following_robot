from setuptools import find_packages, setup
import glob

package_name = 'follow_dl_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/img_saved/train/one_person', glob.glob('img_saved/train/one_person/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='haneol',
    maintainer_email='haneol0415@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'img_saver = follow_dl_pkg.img_saver:main',
            'tracker = follow_dl_pkg.tracker:main',
            'multi_person_tracker = follow_dl_pkg.multi_person_tracker:main',
            'hand_recognizer = follow_dl_pkg.hand_recognizer:main'
        ],
    },
)
