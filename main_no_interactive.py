#!/usr/bin/env python3
"""
main_no_interactive.py
DEPRECATED: Merged into main_batch_cluster.py.

This file is maintained for backward compatibility. All actions, CLI arguments,
and execution workflows have been unified into main_batch_cluster.py.
"""
import sys
import main_batch_cluster

if __name__ == "__main__":
    print("[DEPRECATED] 'main_no_interactive.py' has been merged into 'main_batch_cluster.py'.")
    print("             Forwarding execution to main_batch_cluster.py...\n", flush=True)
    main_batch_cluster.main()
