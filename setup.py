import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="draw_network",
    version="0.01",
    author="Mathias Gotsmy",
    author_email="mathias.gotsmy@univie.ac.at",
    description="Drawing networks for perfectionists.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gotsmy/draw_network",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNUv3 License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_namespace_packages(where="."),
    python_requires=">=3.10",
    include_package_data=True,
    package_data={'': ['data/*']},
    install_requires=['matplotlib','numpy']
)
