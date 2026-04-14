#!/bin/bash
uvicorn titanic_api:app --host 0.0.0.0 --port $PORT
