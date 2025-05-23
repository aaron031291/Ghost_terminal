"""Command-line interface for Grace."""

import argparse
import sys
import json
import logging
from typing import Dict, Any, Optional, List

from grace.grace import Grace

logger = logging.getLogger("grace.cli")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Grace: Generative Reasoning and Coding Engine")
    
    # Main command
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute code")
    execute_parser.add_argument("--code", help="Code to execute")
    execute_parser.add_argument("--file", help="File containing code to execute")
    execute_parser.add_argument("--language", default="python", help="Programming language")
    execute_parser.add_argument("--mode", choices=["strict", "inline"], help="Execution mode")
    
    # Learn command
    learn_parser = subparsers.add_parser("learn", help="Learn from code")
    learn_parser.add_argument("--code", help="Code to learn from")
    learn_parser.add_argument("--file", help="File containing code to learn from")
    learn_parser.add_argument("--language", default="python", help="Programming language")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export project")
    export_parser.add_argument("project_path", help="Path to project directory")
    export_parser.add_argument("destination", choices=["github", "zip", "docker", "vercel"], help="Export destination")
    export_parser.add_argument("--name", help="Project name for export")
    export_parser.add_argument("--description", help="Project description")
    export_parser.add_argument("--private", action="store_true", help="Make repository private (for GitHub)")
    export_parser.add_argument("--output", help="Output path (for ZIP)")
    
    # Teach command
    teach_parser = subparsers.add_parser("teach", help="Teach Grace from a file")
    teach_parser.add_argument("file_path", help="Path to file")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train Grace on a directory")
    train_parser.add_argument("directory_path", help="Path to directory")
    train_parser.add_argument("--recursive", action="store_true", help="Recursively process subdirectories")
    
    # Knowledge commands
    knowledge_parser = subparsers.add_parser("knowledge", help="Knowledge base operations")
    knowledge_subparsers = knowledge_parser.add_subparsers(dest="knowledge_command", help="Knowledge command")
    
    save_parser = knowledge_subparsers.add_parser("save", help="Save knowledge base")
    save_parser.add_argument("output_path", help="Path to save knowledge base")
    
    load_parser = knowledge_subparsers.add_parser("load", help="Load knowledge base")
    load_parser.add_argument("input_path", help="Path to load knowledge base from")
    
    query_parser = knowledge_subparsers.add_parser("query", help="Query knowledge base")
    query_parser.add_argument("query", help="Query string")
    
    summary_parser = knowledge_subparsers.add_parser("summary", help="Get knowledge base summary")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Get Grace version")
    
    # Config argument for all commands
    parser.add_argument("--config", help="Path to configuration file")
    
    # Output format
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    return parser.parse_args()


def get_code_from_file(file_path: str) -> str:
    """Read code from file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Code as string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        sys.exit(1)


def format_output(result: Any, format_type: str) -> str:
    """Format output based on format type.
    
    Args:
        result: Result to format
        format_type: Format type (text or json)
        
    Returns:
        Formatted output
    """
    if format_type == "json":
        if hasattr(result, "to_dict"):
            return json.dumps(result.to_dict(), indent=2)
        elif isinstance(result, dict):
            return json.dumps(result, indent=2)
        else:
            return json.dumps({"result": str(result)}, indent=2)
    else:
        return str(result)


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Initialize Grace
    grace = Grace(args.config)
    
    # Execute command
    if args.command == "execute":
        if args.code:
            code = args.code
        elif args.file:
            code = get_code_from_file(args.file)
        else:
            logger.error("Either --code or --file must be specified")
            sys.exit(1)
            
        result = grace.execute_code(code, args.language, args.mode)
        print(format_output(result, args.format))
        
    # Learn command
    elif args.command == "learn":
        if args.code:
            code = args.code
        elif args.file:
            code = get_code_from_file(args.file)
        else:
            logger.error("Either --code or --file must be specified")
            sys.exit(1)
            
        result = grace.learn_from_code(code, args.language)
        print(format_output(result, args.format))
        
    # Export command
    elif args.command == "export":
        kwargs = {}
        
        if args.name:
            kwargs["repo_name" if args.destination == "github" else "tag" if args.destination == "docker" else "project_name"] = args.name
            
        if args.description and args.destination == "github":
            kwargs["description"] = args.description
            
        if args.private and args.destination == "github":
            kwargs["private"] = args.private
            
        if args.output and args.destination == "zip":
            kwargs["output_path"] = args.output
            
        result = grace.export_project(args.project_path, args.destination, **kwargs)
        print(format_output(result, args.format))
        
    # Teach command
    elif args.command == "teach":
        result = grace.teach(args.file_path)
        print(format_output(result, args.format
        print(format_output(result, args.format))
        
    # Train command
    elif args.command == "train":
        result = grace.train(args.directory_path, args.recursive)
        print(format_output(result, args.format))
        
    # Knowledge commands
    elif args.command == "knowledge":
        if args.knowledge_command == "save":
            result = grace.save_knowledge(args.output_path)
            print(format_output({"success": result}, args.format))
            
        elif args.knowledge_command == "load":
            result = grace.load_knowledge(args.input_path)
            print(format_output({"success": result}, args.format))
            
        elif args.knowledge_command == "query":
            result = grace.query_knowledge(args.query)
            print(format_output(result, args.format))
            
        elif args.knowledge_command == "summary":
            result = grace.get_knowledge_summary()
            print(format_output(result, args.format))
            
        else:
            logger.error("Unknown knowledge command")
            sys.exit(1)
            
    # Version command
    elif args.command == "version":
        version = grace.get_version()
        print(format_output({"version": version}, args.format))
        
    else:
        logger.error("Unknown command")
        sys.exit(1)


if __name__ == "__main__":
    main()
