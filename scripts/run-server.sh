#!/bin/bash

PYTHONPATH="src" uvicorn server.main:app --port 8000 --reload