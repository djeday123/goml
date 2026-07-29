#!/bin/bash
cd /data/lib/podman-data/projects/goml
export GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs
go test -run 'TestALLM_MatMulF32Ex_ZeroRepro' -timeout 5m -v ./internal/abjexam/ 2>&1
