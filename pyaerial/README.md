# pyAerial developer guide

The goal of pyAerial is to provide a documented Python API to everything in Aerial. Currently, it provides a Python API to a subset of cuPHY functionality.

pyAerial is in particular intended towards machine learning researchers, and can be used for example in model validation, model benchmarking, dataset generation and in generating ground truth signals for over the air collected datasets. Such ground truth signals can be for example transmitted MAC PDUs (obtained via detection), coded bits (via re-encoding the information bits) or the transmitted symbols.

## Building and running the pyAerial container

To use pyAerial, one needs its own container with machine learning tools such as NVIDIA Sionna, NVIDIA TensorRT and TensorFlow pre-installed.

The container is built and run using the following. Note that this requires an installation of [Docker](https://www.docker.com/) and [hpccm](https://github.com/NVIDIA/hpc-container-maker).
```
$cuBB_SDK/pyaerial/container/build.sh
$cuBB_SDK/pyaerial/container/run.sh
```

## Building and installing pyAerial
First, cuPHY and in particular the Python bindings under `pyaerial/pybind11` need to be built.
Run the following within the pyAerial container:
```
cd $cuBB_SDK
cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native -DNVIPC_FMTLOG_ENABLE=OFF -DASIM_CUPHY_SRS_OUTPUT_FP32=ON
cmake --build build -t _pycuphy pycuphycpp
```
Once cuPHY and the Python bindings have been built, the `pyaerial` package needs to be built. There is a script for this purpose:
```
$cuBB_SDK/pyaerial/scripts/install_dev_pkg.sh
```
The script defines a `BUILD_ID` if it does not exist, builds and installs the pip package in development mode (so that
after any changes to the Python code the package does not need to be rebuilt).

The installation can be tested by running
```
python3 -c "import aerial"
```
which should just succeed without errors. The installation can be further tested by running the unit tests, see below.

## Running pyAerial tests

pyAerial static and unit tests can be run in the pyAerial container (see above how to build the pyAerial container).

The unit tests are using (a subset of) the same cuPHY test vectors as cuPHY itself. These are assumed to be located under `/mnt/cicd_tvs/develop/GPU_test_input/`, or in another location set into environment variable `TEST_VECTOR_DIR`.

pyAerial contains static code analysis based tests as well as unit tests testing the functionality. The static tests can be run from like this:
```
$cuBB_SDK/pyaerial/scripts/run_static_tests.sh
```
This runs `flake8` and `pylint` for coding style, `mypy` for static type analysis and `interrogate` for checking the coverage of docstrings. For passing static tests, all these need to pass.

The unit tests can be run as:
```
$cuBB_SDK/pyaerial/scripts/run_unit_tests.sh
```

## Running pyAerial example notebooks
pyAerial contains a number of example notebooks in Jupyter notebook format. The Jupyter notebooks can be run interactively
within the pyAerial container. This is done by starting a JupyterLab server (within the pyAerial container) as
```
cd $cuBB_SDK/pyaerial/notebooks
jupyter lab --ip=0.0.0.0
```
and then pointing the browser to the given address.

All notebooks can be pre-executed by running the following (within the Docker container):
```
$cuBB_SDK/pyaerial/scripts/run_notebooks.sh
```

Note that running the example notebooks takes a rather long time as some of them are running full simulations and training ML models within the notebook.
Also note that the Aerial Data Lake example notebooks require the example database to be created first. Refer to Aerial Data Lake documentation on
how to start the clickhouse server and create the example database.
