from invoke import task
import os
import subprocess
import sys

@task
def requirements(c):
    """Install Python Dependencies"""
    test_environment(c)
    c.run(f"{sys.executable} -m pip install -U pip setuptools wheel")
    c.run(f"{sys.executable} -m pip install -r requirements.txt")

@task
def data(c):
    """Make Dataset"""
    requirements(c)
    c.run(f"{sys.executable} src/data/make_dataset.py data/raw data/processed")

@task
def clean(c):
    """Delete all compiled Python files"""
    c.run("find . -type f -name '*.py[co]' -delete")
    c.run("find . -type d -name '__pycache__' -delete")

@task
def lint(c):
    """Lint using flake8"""
    c.run("flake8 src")

@task
def test_environment(c):
    """Test python environment is setup correctly"""
    c.run(f"{sys.executable} test_environment.py")
