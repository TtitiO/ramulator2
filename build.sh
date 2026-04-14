mkdir -p build
cd build
cmake -DPython_EXECUTABLE=../.venv/bin/python ..
make -j$(nproc)
cd ..
uv pip install --no-build-isolation -e .
PYTHONPATH=python python -m ramulator codegen
