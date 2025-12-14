#!/usr/bin/env python3
"""
Convert a simple Sniper-style text trace into a netrace .tra (optionally .tra.bz2) file.

Input format (whitespace-separated):
  cycle  src  dst  size_bytes  latency_ns  class

Mapping:
  - cycle      -> nt_packet_t.cycle
  - src/dst    -> nt_packet_t.src/dst
  - size_bytes -> nt_packet_t.addr (BookSim can read this with netrace_use_addr_size=1)
  - class      -> nt_packet_t.node_types (BookSim uses this as suggested class when
                 netrace_class_from_node_types=1)
  - type       -> constant (default 1 = ReadReq)
  - deps       -> none (num_deps = 0)

Usage:
  python3 convert_to_netrace.py --input workloads/fft_trace.log \
    --output workloads/fft_trace.tra.bz2 --nodes 64 --benchmark fft
"""

import argparse
import bz2
import os
import struct
from typing import Iterable, Tuple

NT_MAGIC = 0x484A5455
HEADER_FMT = "<If30sBBQQII8s"  # packed header layout used by netrace
PACKET_FMT = "<QII5B"          # cycle, id, addr, type, src, dst, node_types, num_deps


def _pack_header(num_nodes: int, num_cycles: int, num_packets: int,
                 benchmark: str, notes: str) -> bytes:
    name_bytes = benchmark.encode("ascii", errors="ignore")[:30]
    name_bytes = name_bytes.ljust(30, b"\0")
    notes_bytes = notes.encode("ascii", errors="ignore") + b"\0" if notes else b""
    notes_len = len(notes_bytes)
    return struct.pack(
        HEADER_FMT,
        NT_MAGIC,
        1.0,  # version
        name_bytes,
        num_nodes & 0xFF,
        0,  # pad
        num_cycles,
        num_packets,
        notes_len,
        0,  # num_regions
        b"\0" * 8,
    ) + notes_bytes


def _pack_packet(pkt_id: int, cycle: int, src: int, dst: int,
                 size_bytes: int, pkt_type: int, clazz: int) -> bytes:
    return struct.pack(
        PACKET_FMT,
        cycle,
        pkt_id,
        size_bytes,
        pkt_type & 0xFF,
        src & 0xFF,
        dst & 0xFF,
        clazz & 0xFF,
        0,  # num_deps
    )


def _parse_lines(lines: Iterable[str]) -> Iterable[Tuple[int, int, int, int, int]]:
    """Yield (cycle, src, dst, size_bytes, class) for each valid line."""
    for lineno, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) < 6:
            raise ValueError(f"Line {lineno}: expected 6 fields, got {len(parts)}")
        cycle = int(parts[0])
        src = int(parts[1])
        dst = int(parts[2])
        size_bytes = int(parts[3])
        clazz = int(parts[5])
        yield (cycle, src, dst, size_bytes, clazz)


def convert(input_path: str, output_path: str, nodes: int,
            benchmark: str, pkt_type: int, notes: str, compress: bool) -> None:
    packets = list(_parse_lines(open(input_path)))
    if not packets:
        raise ValueError("No packets parsed from input trace.")
    num_packets = len(packets)
    max_cycle = max(p[0] for p in packets)
    inferred_nodes = max(max(p[1], p[2]) for p in packets) + 1
    num_nodes = nodes or inferred_nodes

    header_bytes = _pack_header(num_nodes, max_cycle, num_packets, benchmark, notes)

    if compress:
        fp = bz2.open(output_path, "wb")
    else:
        fp = open(output_path, "wb")

    with fp:
        fp.write(header_bytes)
        for pkt_id, (cycle, src, dst, size_bytes, clazz) in enumerate(packets):
            fp.write(_pack_packet(pkt_id, cycle, src, dst, size_bytes, pkt_type, clazz))


def main():
    parser = argparse.ArgumentParser(description="Convert Sniper trace to netrace .tra/.tra.bz2")
    parser.add_argument("--input", required=True, help="Input trace file (cycle src dst size latency class)")
    parser.add_argument("--output", help="Output .tra or .tra.bz2 path (default: input + .tra.bz2)")
    parser.add_argument("--nodes", type=int, default=0,
                        help="Number of nodes; default infers max(src,dst)+1")
    parser.add_argument("--benchmark", default="sniper", help="Benchmark name for header (<=30 chars)")
    parser.add_argument("--type", type=int, default=1, help="Netrace packet type code (default 1=ReadReq)")
    parser.add_argument("--notes", default="", help="Optional notes string for header")
    parser.add_argument("--no-compress", action="store_true",
                        help="Write raw .tra instead of bzip2-compressed .tra.bz2")
    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        suffix = ".tra" if args.no_compress else ".tra.bz2"
        output_path = args.input + suffix

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    convert(
        input_path=args.input,
        output_path=output_path,
        nodes=args.nodes,
        benchmark=args.benchmark,
        pkt_type=args.type,
        notes=args.notes,
        compress=not args.no_compress,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
