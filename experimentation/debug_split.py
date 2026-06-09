#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import itertools
from pathlib import Path
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.callify import transform_to_call
from tinygrad.helpers import DEBUG_RANGEIFY, OPENPILOT_HACKS
from tinygrad.schedule.indexing import IndexingContext, pm_generate_realize_map
from tinygrad.schedule.multi import multi_pm
from tinygrad.schedule.rangeify import (
  earliest_rewrites,
  pm_add_buffers,
  pm_add_range_tags,
  pm_const_buffer_folding,
  pm_fold_moved_after,
  pm_limit_bufs,
  pm_mops,
  pm_reduce_simplify,
  pm_remove_bufferize,
  pm_syntactic_sugar,
  run_rangeify,
  split_kernels,
)
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.uop.render import pretty_print, pyrender
from tinygrad.uop.symbolic import symbolic


def example_two_matmuls() -> UOp:
  b, k1, h, o = 16, 32, 24, 20
  x = (Tensor.arange(b*k1, dtype=dtypes.float).reshape(b, k1) / 100).realize()
  w1 = (Tensor.arange(k1*h, dtype=dtypes.float).reshape(k1, h) / 100).realize()
  w2 = (Tensor.arange(h*o, dtype=dtypes.float).reshape(h, o) / 100).realize()
  y = x.matmul(w1).relu().matmul(w2).relu()
  return transform_to_call(UOp.sink(y.uop))[0].src[0]


def example_reduce_keepdim_add() -> UOp:
  x = Tensor.arange(16, dtype=dtypes.float).reshape(4, 4).realize()
  y = x + x.sum(axis=-1, keepdim=True)
  return transform_to_call(UOp.sink(y.uop))[0].src[0]


EXAMPLES: dict[str, tuple[str, callable[[], UOp]]] = {
  "two_matmuls": ("x.matmul(w1).relu().matmul(w2).relu()", example_two_matmuls),
  "reduce_keepdim_add": ("x + x.sum(axis=-1, keepdim=True)", example_reduce_keepdim_add),
}

def to_sink(value: Any) -> UOp:
  if isinstance(value, UOp):
    return value
  if isinstance(value, Tensor):
    return transform_to_call(UOp.sink(value.uop))[0].src[0]
  raise TypeError(f"build() must return a Tensor or UOp, got {type(value).__name__}")

def load_file_builder(path: str) -> tuple[str, callable[[], UOp]]:
  mod_path = Path(path).resolve()
  spec = importlib.util.spec_from_file_location("debug_split_input", mod_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load module from {mod_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  build = getattr(module, "build", None)
  if not callable(build):
    raise RuntimeError(f"{mod_path} must define a callable build()")
  desc = getattr(module, "DESCRIPTION", mod_path.name)
  return str(desc), lambda: to_sink(build())


def pipeline(sink: UOp) -> dict[str, UOp]:
  tsink = graph_rewrite(sink, multi_pm, name="multi_pm")
  if OPENPILOT_HACKS:
    tsink = graph_rewrite(tsink, pm_fold_moved_after, ctx={}, name="fold moved afters")
  early = graph_rewrite(tsink, pm_syntactic_sugar + pm_mops + earliest_rewrites, bottom_up=True, name="earliest rewrites")
  realize_seed: dict[UOp, None | list[int]] = {}
  graph_rewrite(early, pm_generate_realize_map, ctx=realize_seed, name="get realize")
  ranged, rctx = run_rangeify(early, bool(DEBUG_RANGEIFY))
  debuf = graph_rewrite(ranged, symbolic + pm_reduce_simplify + pm_const_buffer_folding + pm_remove_bufferize,
                        name="symbolic+reduce_collapse+debuf")
  limited = graph_rewrite(debuf, pm_limit_bufs, ctx=rctx, name="limit buffers")
  lunique_start = max([-1] + [x.arg for x in limited.toposort() if x.op is Ops.LUNIQUE]) + 1
  buffered = graph_rewrite(limited, pm_add_buffers + pm_add_range_tags, ctx=itertools.count(lunique_start),
                           bottom_up=True, name="stage to store")
  split = graph_rewrite(buffered, split_kernels, bottom_up=True, name="split kernels")
  return {
    "early": early,
    "realize_seed": realize_seed,
    "realize_final": rctx.realize_map,
    "ranged": ranged,
    "debuf": debuf,
    "buffered": buffered,
    "split": split,
  }


def iter_calls(u: UOp) -> list[UOp]:
  seen: list[UOp] = []
  for node in u.toposort():
    if node.op is Ops.CALL and node not in seen:
      seen.append(node)
  return seen


def iter_stages(u: UOp) -> list[UOp]:
  return [node for node in u.toposort() if node.op is Ops.STAGE]


def print_section(title: str, body: str) -> None:
  print(f"\n=== {title} ===")
  print(body)


def print_stages(title: str, u: UOp) -> None:
  stages = iter_stages(u)
  print(f"\n=== {title} ({len(stages)}) ===")
  if not stages:
    print("(none)")
    return
  for i, stage in enumerate(stages, start=1):
    print(f"\n--- stage {i} pretty ---")
    print(pretty_print(stage))
    print(f"\n--- stage {i} pyrender ---")
    print(pyrender(stage))


def print_realize_map(title: str, realize_map: dict[UOp, None | list[int]]) -> None:
  items = list(realize_map.items())
  print(f"\n=== {title} ({len(items)}) ===")
  if not items:
    print("(none)")
    return
  for i, (u, axes) in enumerate(items, start=1):
    print(f"\n--- realize {i} axes={axes} op={u.op} shape={u.shape} ---")
    print(pyrender(u))


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Inspect tinygrad rangeify, staging, bufferization, and kernel splitting.",
    epilog=(
      "Built-in examples:\n"
      "  %(prog)s two_matmuls\n"
      "  %(prog)s reduce_keepdim_add\n\n"
      "Custom file input:\n"
      "  %(prog)s --file my_example.py\n\n"
      "The file must define build(), returning either a Tensor or a UOp.\n"
      "Optional DESCRIPTION = '...'\n"
    ),
    formatter_class=argparse.RawTextHelpFormatter,
  )
  parser.add_argument("example", nargs="?", choices=sorted(EXAMPLES),
                      help="built-in example name")
  parser.add_argument("--file", dest="input_file",
                      help="python file defining build() -> Tensor | UOp")
  parser.add_argument("--list-examples", action="store_true",
                      help="print built-in example names and exit")
  args = parser.parse_args()

  if args.list_examples:
    for name, (desc, _) in EXAMPLES.items():
      print(f"{name}: {desc}")
    return

  if args.input_file and args.example:
    parser.error("pass either an example name or --file, not both")

  if args.input_file:
    example_name = Path(args.input_file).name
    desc, builder = load_file_builder(args.input_file)
  else:
    example_name = args.example or "two_matmuls"
    desc, builder = EXAMPLES[example_name]

  graphs = pipeline(builder())

  print(f"example: {example_name}")
  print(f"expr: {desc}")

  print_section("Before Rangeify", pyrender(graphs["early"]))
  print_realize_map("Initial realize_map seed", graphs["realize_seed"])
  print_stages("After Rangeify / Before remove_bufferize", graphs["ranged"])
  print_realize_map("Final realize_map after range analysis", graphs["realize_final"])
  print_stages("After remove_bufferize", graphs["debuf"])

  print_section("After Stage To Store", pyrender(graphs["buffered"]))
  print_section("After Split Pretty", pretty_print(graphs["split"]))
  print(f"\n=== After Split Kernels ({len(iter_calls(graphs['split']))}) ===")
  for i, call in enumerate(iter_calls(graphs["split"]), start=1):
    print(f"\n--- kernel {i} ---")
    print(pyrender(call.src[0]))


if __name__ == "__main__":
  main()
