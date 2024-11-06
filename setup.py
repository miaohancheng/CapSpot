from setuptools import setup, find_packages

setup(
    name='CapSpot',  # 替换为您的包名
    version='0.1.0',
    description='An image matching package using ONNX models',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Miao Hancheng',  # 替换为您的姓名
    author_email='hanchengmiao@gmail.com',  # 替换为您的邮箱
    url='https://github.com/miaohancheng/CapSpot',  # 替换为您的GitHub仓库地址
    packages=find_packages(),
    include_package_data=True,  # 启用包含包数据
    package_data={
        'CapSpot': ['models/*.onnx'],  # 指定要包含的模型文件
    },
    install_requires=[
        'numpy',
        'opencv-python',
        'Pillow',
        'onnxruntime',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)