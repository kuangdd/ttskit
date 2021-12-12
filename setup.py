#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/15
"""
语音处理工具箱。
生成whl格式安装包：python setup.py bdist_wheel

直接上传pypi：python setup.py sdist upload

用twine上传pypi：
生成安装包：python setup.py sdist
上传安装包：twine upload [package path]

注意：需要在home目录下建立.pypirc配置文件，文件内容格式：
[distutils]
index-servers=pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username: kdd
password: admin
"""

from setuptools import setup, find_packages
import os
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

install_requires = [line.strip() for line in open('requirements.txt', encoding='utf8')
                    if (not line.startswith('#')) and line.strip()]

# install_requires = ['aukit>=1.4.4', 'inflect', 'cycler', 'librosa', 'numpy',
#                     'phkit>=0.2.7', 'pydub', 'PyYAML', 'scikit_learn', 'scipy', 'setproctitle', 'SIP',
#                     'tensorboardX', 'tqdm', 'Unidecode', 'visdom',
#                     'webrtcvad_wheels', 'xmltodict', 'flask', 'gevent']
# 'matplotlib>=3.1.1,<=3.4.2', torch>=1.6.0,<=1.7.1, librosa, numba==0.48 'umap_learn',
requires = [re.sub(r'[<>=].+', '', w) for w in install_requires]


def create_readme():
    from ttskit import sdk_api
    from ttskit import cli_api
    from ttskit import web_api
    from ttskit import http_server
    from ttskit import encoder
    from ttskit import mellotron
    from ttskit import waveglow
    from ttskit import melgan
    from ttskit import resource
    import ttskit
    version_doc = """
    ### 版本
    v{}
    """.format(ttskit.__version__)
    readme_docs = [
        ttskit.__doc__, version_doc,
        sdk_api.__doc__, cli_api.__doc__, web_api.__doc__, http_server.__doc__,
        resource.__doc__, encoder.__doc__, mellotron.__doc__, waveglow.__doc__, melgan.__doc__,
    ]

    docs = []
    with open("README.md", "wt", encoding="utf8") as fout:
        for doc in readme_docs:
            fout.write(doc.replace("\n", "\n"))
            docs.append(doc)
    return "".join(docs)


ttskit_doc = create_readme()
from ttskit import __version__ as ttskit_version

setup(
    name="ttskit",
    version=ttskit_version,
    author="kuangdd",
    author_email="kuangdd@foxmail.com",
    description="text to speech toolkit",
    long_description=ttskit_doc,
    long_description_content_type="text/markdown",
    url="https://github.com/KuangDD/ttskit",
    packages=find_packages(exclude=['contrib', 'docs', 'test*']),
    install_requires=install_requires,  # 指定项目最低限度需要运行的依赖项
    requires=requires,
    python_requires='>=3.6',  # python的依赖关系
    package_data={
        'ttskit': ['requirements.txt', 'README.md'],
        'ttskit.templates': ['index.html', 'index.png'],
        'ttskit.resource.model': ['*.json', '*.pt'],
    },  # 包数据，通常是与软件包实现密切相关的数据
    # include_package_data=True,
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'tkcli = ttskit.cli_api:tts_cli',
            'tkhttp = ttskit.http_server:start_sever',
        ]
    }
)

if __name__ == "__main__":
    logger.info(__file__)
