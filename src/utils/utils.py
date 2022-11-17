import csv
import os
from pathlib import Path
from typing import (
	Any,
	Literal,
	TypeGuard,
	TypedDict,
)

import hydra
import networkx as nx
import numpy as np
from pydantic import BaseModel


class ShouldNotHappen(Exception):
	...


class CsvParams(BaseModel):
    delimiter: str = "\t"
    encoding: str = "utf8"
    quoting: Literal[0, 1, 2, 3] = csv.QUOTE_NONE



def path(path: str) -> Path:
	return Path(hydra.utils.to_absolute_path(path)).resolve()

def dataset_path(name: str, version: str) -> Path:
	root = os.getenv("DATA_DIR")
	if root is None:
		root = "wug"
	return path(root) / name / version

def xor(a: bool, b: bool):
	return (a and not b) or (not a and b)


def _check_nan_weights_exits(
	graph: nx.Graph
):
	"""Check if there are NaN weights in the graph.
	Parameters
	----------
	graph: networkx.Graph
		The graph to check NaN weights for
	Returns
	-------
	flag: bool
		True if there are NaN weights, False otherwise
	"""
	return any(np.isnan(graph.get_edge_data(*edge)['weight']) for edge in graph.edges)


def _negative_weights_exist(
	graph: nx.Graph
):
	"""Check if there are negative edges in the graph.
	Parameters
	----------
	graph: networkx.Graph
		The graph to check negative edges for
	Returns
	-------
	flag: bool
		True if there are negative edges, False otherwise
	"""
	for i, j in graph.edges():
		if graph[i][j]["weight"] < 0:
			return True
	return False

# Typeguards

def is_list(obj: Any) -> TypeGuard[list[Any]]:
	return hasattr(obj, "__len__")


def is_str_list(
	obj: Any
) -> TypeGuard[list[str]]:
	try:
		return obj is not None and all(isinstance(s, str) for s in obj)
	except TypeError:
		return False


def is_number(
	obj: Any
) -> TypeGuard[float | int]:
	try:
		return isinstance(obj, float) or isinstance(obj, int)
	except TypeError:
		return False


def is_int(
	obj: Any
) -> TypeGuard[int]:
	try:
		return isinstance(obj, int)
	except TypeError:
		return False
