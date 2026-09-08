#!/usr/bin/env python3
"""Evaluate merged landmark nodes from nodes.parquet and node composition.

Reuses microcell purity statistics and plotting, with node IDs, node counts
and landmark labels. No assignments or input tables are changed.
"""
from evaluate_microcell_purity import main


if __name__ == '__main__':
    main(kind='node')
