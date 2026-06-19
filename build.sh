mkdir -p build
cd build
cmake -DPython_EXECUTABLE=../../.venv/bin/python ..
make -j$(nproc)
cd ..
uv pip install --python ../.venv/bin/python --no-build-isolation -e .
PYTHONPATH=python ../.venv/bin/python -m ramulator codegen
