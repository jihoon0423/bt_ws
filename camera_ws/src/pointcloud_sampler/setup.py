from setuptools import find_packages, setup

package_name = 'pointcloud_sampler'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='choi',
    maintainer_email='c990305@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 기존에 다른 노드가 있으면 추가하고, 여기에 샘플러 노드 등록
            'pointcloud_sampler_for_nav2 = pointcloud_sampler.pointcloud_sampler_for_nav2:main',
            # 시간적 샘플러나 공간적 샘플러를 별도 스크립트로 만들었다면 여기에 등록
            # 'oakd_pc_temporal_sampler = pointcloud_sampler.pointcloud_temporal_sampler_oakd:main',
            # 'oakd_pc_spatial_sampler = pointcloud_sampler.pointcloud_spatial_sampler_oakd:main',
        ],
    },
)
