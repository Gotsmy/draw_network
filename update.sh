#!/bin/bash
pip uninstall draw_network -y
python setup.py install
cd docs
make html
# make latexpdf
cd ..
