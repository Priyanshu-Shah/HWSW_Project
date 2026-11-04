# HW/SW Partitioning Tool

## Purpose

Given a small task described in plain English, decide which parts should run in **hardware
(FPGA)** and which in **software (CPU)** on a chosen board, to minimize latency.

## Pipeline

```text
plain-English task
   │
   ▼  LLM pipeline: asks clarifying questions, builds a task dependency graph (TDG)
TDG   (tasks = nodes, data transfers = edges)
   │
   ▼  custom_hls (using board specs from the library): estimates HW and SW cost of each task
costed TDG
   │
   ▼  partitioner: assigns each task to HW or SW
HW/SW partition + metrics
```

## Repository layout

| Folder | Contents |
|---|---|
| `pre_TDG/` | English task → costed TDG (LLM pipeline + custom_hls) |
| `TDG/` | Shared TDG schema; the interface between the teams |
| `post_TDG/` | Costed TDG → HW/SW partition (partitioner team) |
| `library/` | Board specifications (library team) |
| `app/` | Web page: chat with the pipeline and view the task graph |
