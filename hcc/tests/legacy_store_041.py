"""Two small arrays written by healpix-compress 0.4.1 under the codec ids of
that release, to check that the ids kept as aliases still open them.

Made with the two 0.4.1 serializers, ``effort=1`` for the single-layer one and
``relative_step=0.5, predictor_size=3, effort=1`` for the pyramid one (Python
path, no native kernels), on 32 float32 values with one ``NaN``, in chunks of
16 cells (``INPUTS``); ``VALUES`` is what 0.4.1 read back.
"""

import base64
import json

import numpy as np

_FILES = {
    "pyramid/c/0": base64.b64decode(
        "SFMyUAECAwHTn/xC/P///6AAAADxAAAA8AAAAAoAAAAAAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA"
        "PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADEpYTCANCocKAAAAFGZ0eXBqeGwgAAAAAGp4bCAAAAAJ"
        "anhsbAoAAADIanhsY/8KCBD8AEWLCAIBAMgCSxibnHGEAziAAzggSsA5BQEAIESACBABIkCE//8/"
        "+P//oj7GGHOutffe55xzDgFFRERERGRmZt73flVVVRX8///B//8X9THGmHOtvfc+55xzCCgiIiIi"
        "IjMz8773q6qqquD//w/+/7+ojzHGnGvtvfc555xDQBERERERmZmZ971fVVVVBf//f/D//0V9jDHm"
        "XGvvvc855xwCioiIiIjIzMy87/2qqqqq+XDXVwZPxe8FAQAAAAxKWEwgDQqHCgAAABRmdHlwanhs"
        "IAAAAABqeGwgAAAACWp4bGwKAAAAx2p4bGP/ChgQ/ABFiwgCAQDEAksYm5xxhAM4gAM4IErAOQUB"
        "ACBEgAgQASJAhP//P/j//6I+xhhzrrX33ueccw4BRURERERkZmbe935VVVUV/P//wf//F/Uxxphz"
        "rb33PueccwgoIiIiIiIzM/O+96uqqqrg//8P/v+/qI8xxpxr7b33OeecQ0AREREREZmZmfe9X1VV"
        "VQX//3/w//9FfYwx5lxr773POeccAoqIiIiIyMzMvO/9qqqqqqGKTQzZQzJDGHja+/sfAAL7Af0="
    ),
    "pyramid/c/1": base64.b64decode(
        "SFMyUAECAwAbfwxD/f///6AAAADxAAAA8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA"
        "PwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADEpYTCANCocKAAAAFGZ0eXBqeGwgAAAAAGp4bCAAAAAJ"
        "anhsbAoAAADIanhsY/8KCBD8AEWLCAIBAMgCSxibnHGEAziAAzggSsA5BQEAIESACBABIkCE//8/"
        "+P//oj7GGHOutffe55xzDgFFRERERGRmZt73flVVVRX8///B//8X9THGmHOtvfc+55xzCCgiIiIi"
        "IjMz8773q6qqquD//w/+/7+ojzHGnGvtvfc555xDQBERERERmZmZ971fVVVVBf//f/D//0V9jDHm"
        "XGvvvc855xwCioiIiIjIzMy87/2qqqqq+TCitz7eynjPAgAAAAxKWEwgDQqHCgAAABRmdHlwanhs"
        "IAAAAABqeGwgAAAACWp4bGwKAAAAx2p4bGP/ChgQ/ABFiwgCAQDEAksYm5xxhAM4gAM4IErAOQUB"
        "ACBEgAgQASJAhP//P/j//6I+xhhzrrX33ueccw4BRURERERkZmbe935VVVUV/P//wf//F/Uxxphz"
        "rb33PueccwgoIiIiIiIzM/O+96uqqqrg//8P/v+/qI8xxpxr7b33OeecQ0AREREREZmZmfe9X1VV"
        "VQX//3/w//9FfYwx5lxr773POeccAoqIiIiIyMzMvO/9qqqqqqEwD8FdCgrmDg=="
    ),
    "pyramid/zarr.json": json.dumps(
        {
            "shape": [32],
            "data_type": "float32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [16]}},
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": "NaN",
            "codecs": [
                {
                    "name": "hcc-sentinel2-pyramid",
                    "configuration": {
                        "relative_step": 0.5,
                        "min_step": 1.0,
                        "predictor_size": 3,
                        "quantification_value": 10000.0,
                        "add_offset": -1000.0,
                        "effort": 1,
                    },
                }
            ],
            "attributes": {},
            "zarr_format": 3,
            "node_type": "array",
            "storage_transformers": [],
        }
    ).encode(),
    "single/c/0": base64.b64decode(
        "AAAADEpYTCANCocKAAAAFGZ0eXBqeGwgAAAAAGp4bCAAAAAJanhsbAoAAADfanhsY/8KGBD8AEWL"
        "CAIBACQDSxibnHGEAziAAzggSsA5BQEAIESACBABIkCE//8/+P//oj7GGHOutffe55xzDgFFRERE"
        "RGRmZt73flVVVRX8///B//8X9THGmHOtvfc+55xzCCgiIiIiIjMz8773q6qqquD//w/+/7+ojzHG"
        "nGvtvfc555xDQBERERERmZmZ971fVVVVBf//f/D//0V9jDHmXGvvvc855xwCioiIiIjIzMy87/2q"
        "qqqqeaJ/Gn1T54n0Qfpq+S2Nh+hLnbc0Xuo8jT5EH6QP0gcp"
    ),
    "single/c/1": base64.b64decode(
        "AAAADEpYTCANCocKAAAAFGZ0eXBqeGwgAAAAAGp4bCAAAAAJanhsbAoAAADfanhsY/8KGBD8AEWL"
        "CAIBACQDSxibnHGEAziAAzggSsA5BQEAIESACBABIkCE//8/+P//oj7GGHOutffe55xzDgFFRERE"
        "RGRmZt73flVVVRX8///B//8X9THGmHOtvfc+55xzCCgiIiIiIjMz8773q6qqquD//w/+/7+ojzHG"
        "nGvtvfc555xDQBERERERmZmZ971fVVVVBf//f/D//0V9jDHmXGvvvc855xwCioiIiIjIzMy87/2q"
        "qqqq+fDwp9E3dZ5IH6QP0gfpQ/SlztPoS52n0Yfog/RB+iAF"
    ),
    "single/zarr.json": json.dumps(
        {
            "shape": [32],
            "data_type": "float32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [16]}},
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": "NaN",
            "codecs": [
                {
                    "name": "hcc-sentinel2",
                    "configuration": {
                        "quantification_value": 10000.0,
                        "add_offset": -1000.0,
                        "effort": 1,
                    },
                }
            ],
            "attributes": {},
            "zarr_format": 3,
            "node_type": "array",
            "storage_transformers": [],
        }
    ).encode(),
}


def _inputs() -> dict[str, np.ndarray]:
    single = np.linspace(-0.05, 0.6, 32).astype("float32")
    single[3] = np.nan
    pyramid = (0.2 + 0.1 * np.sin(np.arange(32) / 3.0)).astype("float32")
    pyramid[9] = np.nan
    return {"single": single, "pyramid": pyramid}


#: The values written, by array.
INPUTS = _inputs()

#: The files of the store, by key.
FILES = _FILES

#: The values 0.4.1 read back, by array.
VALUES = {
    "single": np.array(
        [
            -0.05000000074505806,
            -0.028999999165534973,
            -0.008100000210106373,
            np.nan,
            0.033900000154972076,
            0.05480000004172325,
            0.07580000162124634,
            0.09679999947547913,
            0.1177000030875206,
            0.1386999934911728,
            0.15970000624656677,
            0.18060000240802765,
            0.20160000026226044,
            0.22259999811649323,
            0.2434999942779541,
            0.2644999921321869,
            0.2854999899864197,
            0.30649998784065247,
            0.32739999890327454,
            0.3483999967575073,
            0.3693999946117401,
            0.3903000056743622,
            0.41130000352859497,
            0.43230000138282776,
            0.45320001244544983,
            0.4742000102996826,
            0.4952000081539154,
            0.5160999894142151,
            0.5371000170707703,
            0.5580999851226807,
            0.5789999961853027,
            0.6000000238418579,
        ],
        dtype="float32",
    ),
    "pyramid": np.array(
        [
            0.1941751390695572,
            0.23206877708435059,
            0.2573312222957611,
            0.28259363770484924,
            0.3026312291622162,
            0.3026312291622162,
            0.28999999165534973,
            0.27736878395080566,
            0.24272486567497253,
            np.nan,
            0.17956878244876862,
            0.1543063521385193,
            0.12153121829032898,
            0.10890000313520432,
            0.09626878798007965,
            0.10890000313520432,
            0.11735104769468307,
            0.14545035362243652,
            0.17354965209960938,
            0.20164895057678223,
            0.24390070140361786,
            0.2720000147819519,
            0.28604966402053833,
            0.30009931325912476,
            0.30249929428100586,
            0.28844964504241943,
            0.274399995803833,
            0.24630069732666016,
            0.20514895021915436,
            0.1770496517419815,
            0.14895035326480865,
            0.1208510473370552,
        ],
        dtype="float32",
    ),
}
