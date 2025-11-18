#!/usr/bin/env bash

conda create -n immrax python=3.12
cd immrax && conda run -n immrax pip install -e .
conda run -n immrax conda install ipympl ipykernel --live-stream
