"""Universal Export Layer for Grace.

This module handles exporting projects to various destinations.
"""

import os
import logging
import zipfile
import tempfile
import shutil
import subprocess
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import json
import requests
import git

from grace.config import Config

logger = logging.getLogger("grace.export")


class ExportResult:
    """Result of an export operation."""

    def __init__(
        self,
        success: bool,
        destination: str,
        url: Optional[str] = None,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize export result.
        
        Args:
            success: Whether export was successful
            destination: Destination type (github, zip, docker, vercel)
            url: URL or path to exported project
            error: Error message if export failed
                        details: Additional details about the export
        """
        self.success = success
        self.destination = destination
        self.url = url
        self.error = error
        self.details = details or {}
        
    def __str__(self) -> str:
        """String representation of export result."""
        status = "SUCCESS" if self.success else "FAILURE"
        result = f"Export to {self.destination}: {status}\n"
        
        if self.url:
            result += f"URL: {self.url}\n"
            
        if self.error:
            result += f"Error: {self.error}\n"
            
        if self.details:
            result += "Details:\n"
            for key, value in self.details.items():
                result += f"  {key}: {value}\n"
                
        return result
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "destination": self.destination,
            "url": self.url,
            "error": self.error,
            "details": self.details,
        }


class UniversalExportLayer:
    """Handles exporting projects to various destinations."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize export layer.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        
    def export_to_github(
        self,
        project_path: str,
        repo_name: Optional[str] = None,
        description: str = "Project created by Grace",
        private: bool = False,
    ) -> ExportResult:
        """Export project to GitHub.
        
        Args:
            project_path: Path to project directory
            repo_name: Name for GitHub repository. If None, uses directory name.
            description: Repository description
            private: Whether repository should be private
            
        Returns:
            ExportResult with GitHub repository details
        """
        # Get GitHub token from environment
        token_env = self.config.get("export", "github_token_env")
        github_token = os.environ.get(token_env)
        
        if not github_token:
            return ExportResult(
                success=False,
                destination="github",
                error=f"GitHub token not found in environment variable {token_env}",
            )
            
        # Determine repository name
        if not repo_name:
            repo_name = os.path.basename(os.path.abspath(project_path))
            # Sanitize repo name
            repo_name = repo_name.replace(" ", "-").lower()
            
        try:
            # Create repository on GitHub
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            
            data = {
                "name": repo_name,
                "description": description,
                "private": private,
                "auto_init": False,
            }
            
            response = requests.post(
                "https://api.github.com/user/repos",
                headers=headers,
                json=data,
            )
            
            if response.status_code != 201:
                return ExportResult(
                    success=False,
                    destination="github",
                    error=f"Failed to create GitHub repository: {response.json().get('message', 'Unknown error')}",
                )
                
            repo_info = response.json()
            repo_url = repo_info["html_url"]
            git_url = repo_info["clone_url"]
            
            # Initialize git repository locally if needed
            repo = None
            try:
                repo = git.Repo(project_path)
            except git.exc.InvalidGitRepositoryError:
                repo = git.Repo.init(project_path)
                
            # Add remote if not already present
            try:
                origin = repo.remote("origin")
                if origin.url != git_url:
                    repo.delete_remote("origin")
                    repo.create_remote("origin", git_url)
            except ValueError:
                repo.create_remote("origin", git_url)
                
            # Add all files
            repo.git.add(A=True)
            
            # Commit if there are changes
            if repo.is_dirty() or len(repo.untracked_files) > 0:
                repo.git.commit("-m", "Initial commit by Grace")
                
            # Set up authentication for push
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                f.write(f"url=https://x-access-token:{github_token}@github.com/\n")
                git_credentials = f.name
                
            try:
                # Set git credentials
                repo.git.config("credential.helper", f"store --file={git_credentials}")
                
                # Push to GitHub
                repo.git.push("--set-upstream", "origin", "master")
            finally:
                # Clean up credentials file
                os.unlink(git_credentials)
                
            return ExportResult(
                success=True,
                destination="github",
                url=repo_url,
                details={
                    "repo_name": repo_name,
                    "private": private,
                    "git_url": git_url,
                },
            )
            
        except Exception as e:
            logger.error(f"Error exporting to GitHub: {str(e)}")
            return ExportResult(
                success=False,
                destination="github",
                error=f"Error exporting to GitHub: {str(e)}",
            )
            
    def export_to_zip(
        self,
        project_path: str,
        output_path: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> ExportResult:
        """Export project to ZIP file.
        
        Args:
            project_path: Path to project directory
            output_path: Path for output ZIP file. If None, uses project_path.zip.
            exclude_patterns: List of glob patterns to exclude
            
        Returns:
            ExportResult with ZIP file details
        """
        try:
            # Determine output path
            if not output_path:
                output_path = f"{os.path.abspath(project_path)}.zip"
                
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Default exclude patterns
            if exclude_patterns is None:
                exclude_patterns = [
                    "**/__pycache__/**",
                    "**/.git/**",
                    "**/.DS_Store",
                    "**/node_modules/**",
                    "**/.env",
                    "**/*.pyc",
                ]
                
            # Create ZIP file
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(project_path):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not any(
                        Path(os.path.join(root, d)).match(pattern)
                        for pattern in exclude_patterns
                    )]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        # Skip excluded files
                        if any(Path(file_path).match(pattern) for pattern in exclude_patterns):
                            continue
                            
                        # Add file to ZIP
                        arcname = os.path.relpath(file_path, project_path)
                        zipf.write(file_path, arcname)
                        
            return ExportResult(
                success=True,
                destination="zip",
                url=output_path,
                details={
                    "file_size": os.path.getsize(output_path),
                    "file_count": len(zipf.namelist()),
                },
            )
            
        except Exception as e:
            logger.error(f"Error exporting to ZIP: {str(e)}")
            return ExportResult(
                success=False,
                destination="zip",
                error=f"Error exporting to ZIP: {str(e)}",
            )
            
    def export_to_docker(
        self,
        project_path: str,
        tag: Optional[str] = None,
        dockerfile_path: Optional[str] = None,
        build_args: Optional[Dict[str, str]] = None,
    ) -> ExportResult:
        """Export project to Docker image.
        
        Args:
            project_path: Path to project directory
            tag: Docker image tag. If None, uses default from config.
            dockerfile_path: Path to Dockerfile. If None, uses project_path/Dockerfile.
            build_args: Dictionary of build arguments
            
        Returns:
            ExportResult with Docker image details
        """
        try:
            # Check if docker is installed
            try:
                subprocess.run(
                    ["docker", "--version"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                return ExportResult(
                    success=False,
                    destination="docker",
                    error="Docker not installed or not in PATH",
                )
                
            # Determine tag
            if not tag:
                tag = self.config.get("export", "default_docker_tag")
                
            # Determine Dockerfile path
            if not dockerfile_path:
                dockerfile_path = os.path.join(project_path, "Dockerfile")
                
            # Check if Dockerfile exists
            if not os.path.exists(dockerfile_path):
                return ExportResult(
                    success=False,
                    destination="docker",
                    error=f"Dockerfile not found at {dockerfile_path}",
                )
                
            # Build Docker image
            cmd = ["docker", "build", "-t", tag, "-f", dockerfile_path]
            
            # Add build args
            if build_args:
                for key, value in build_args.items():
                    cmd.extend(["--build-arg", f"{key}={value}"])
                    
            cmd.append(project_path)
            
            process = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            # Get image details
            image_info = subprocess.run(
                ["docker", "image", "inspect", tag],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            image_details = json.loads(image_info.stdout)
            
            return ExportResult(
                success=True,
                destination="docker",
                url=f"docker://{tag}",
                details={
                    "tag": tag,
                    "image_id": image_details[0]["Id"] if image_details else None,
                    "size": image_details[0]["Size"] if image_details else None,
                },
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error exporting to Docker: {e.stderr}")
            return ExportResult(
                success=False,
                destination="docker",
                error=f"Error exporting to Docker: {e.stderr}",
            )
        except Exception as e:
            logger.error(f"Error exporting to Docker: {str(e)}")
            return ExportResult(
                success=False,
                destination="docker",
                error=f"Error exporting to Docker: {str(e)}",
            )
            
    def export_to_vercel(
        self,
        project_path: str,
        project_name: Optional[str] = None,
        team: Optional[str] = None,
    ) -> ExportResult:
        """Export project to Vercel.
        
        Args:
            project_path: Path to project directory
            project_name: Vercel project name. If None, uses directory name.
            team: Vercel team name or ID
            
        Returns:
            ExportResult with Vercel deployment details
        """
        try:
            # Check if vercel CLI is installed
            try:
                subprocess.run(
                    ["vercel", "--version"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                return ExportResult(
                    success=False,
                    destination="vercel",
                    error="Vercel CLI not installed or not in PATH",
                )
                
            # Determine project name
            if not project_name:
                project_name = os.path.basename(os.path.abspath(project_path))
                # Sanitize project name
                project_name = project_name.replace(" ", "-").lower()
                
            # Build vercel command
            cmd = ["vercel", "--confirm"]
            
            if project_name:
                cmd.extend(["--name", project_name])
                
            if team:
                cmd.extend(["--scope", team])
                
            # Deploy to Vercel
            process = subprocess.run(
                cmd,
                cwd=project_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            # Extract deployment URL from output
            output = process.stdout
            url_match = re.search(r"https://[^\s]+\.vercel\.app", output)
            deployment_url = url_match.group(0) if url_match else None
            
            return ExportResult(
                success=True,
                destination="vercel",
                url=deployment_url,
                details={
                    "project_name": project_name,
                    "team": team,
                    "output": output,
                },
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error exporting to Vercel: {e.stderr}")
            return ExportResult(
                success=False,
                destination="vercel",
                error=f"Error exporting to Vercel: {e.stderr}",
            )
        except Exception as e:
            logger.error(f"Error exporting to Vercel: {str(e)}")
            return ExportResult(
                success=False,
                destination="vercel",
                error=f"Error exporting to Vercel: {str(e)}",
            )
            
    def export(
        self,
        project_path: str,
        destination: str,
        **kwargs,
    ) -> ExportResult:
        """Export project to specified destination.
        
        Args:
            project_path: Path to project directory
            destination: Destination type (github, zip, docker, vercel)
            **kwargs: Additional arguments for specific destination
            
        Returns:
            ExportResult with export details
        """
        if destination == "github":
            return self.export_to_github(project_path, **kwargs)
        elif destination == "zip":
            return self.export_to_zip(project_path, **kwargs)
        elif destination == "docker":
            return self.export_to_docker(project_path, **kwargs)
        elif destination == "vercel":
            return self.export_to_vercel(project_path, **kwargs)
        else:
            return ExportResult(
                success=False,
                destination=destination,
                error=f"Unsupported destination: {destination}",
            )

