#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grace Entry Point

This script initializes Grace's cognitive router and handles input commands.
"""

import json
import argparse
from typing import Dict, Any
from grace_core.router import LightningRouter
from grace_core.protocols import GraceResponse


def parse_cli_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Grace Router Command Line Interface")
    parser.add_argument("--signal", required=True, help="The signal/command to route")
    parser.add_argument("--payload", default="{}", help="JSON string payload for the signal")
    parser.add_argument("--source", default="cli", help="Source of the signal (default: cli)")

    args = parser.parse_args()
    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError:
        raise ValueError("Payload must be a valid JSON string.")

    return {
        "signal": args.signal,
        "payload": payload,
        "source": args.source
    }


def main():
    args = parse_cli_args()
    router = LightningRouter()
    response: GraceResponse = router.route_input(args["signal"], args["payload"], args["source"])
    
    print(json.dumps(response.to_dict(), indent=2))


if __name__ == "__main__":
    main()