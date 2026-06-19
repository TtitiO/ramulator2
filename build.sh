mkdir -p build
cd build
cmake -DPython_EXECUTABLE=../../.venv/bin/python ..
make -j$(nproc)
cd ..
../.venv/bin/python -m pip install --no-build-isolation -e .
PYTHONPATH=python ../.venv/bin/python -m ramulator codegen
